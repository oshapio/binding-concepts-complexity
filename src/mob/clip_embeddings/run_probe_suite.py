from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import os
from dataset_io import load_dataset
import train_embeddings as train_embeddings_mod


def _repo_root():
    # src/mob/pipelines/run_probe_suite.py -> repo root
    return Path(__file__).resolve().parents[3]


def _now_tag():
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _unique_suffix():
    return f"{os.getpid()}-{time.time_ns()}"


def _read_pickle_embeddings(path: Path):
    with path.open("rb") as f:
        arr = pickle.load(f)
    if isinstance(arr, dict) and "embeddings" in arr:
        arr = arr["embeddings"]
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected embeddings pickle to be rank-2 [N,D], got {arr.shape} from {path}")
    return arr


def _save_pickle(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(arr, f)


def _build_object_id_map(dataset: list):
    object_to_id: Dict[Tuple[int, ...], int] = {}
    for scene in dataset:
        for obj in scene:
            key = tuple(int(v) for v in obj)
            if key not in object_to_id:
                object_to_id[key] = len(object_to_id)
    return object_to_id


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.float()
    y_pred = y_pred.float()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0, keepdim=True)) ** 2)
    return float((1.0 - ss_res / (ss_tot + 1e-12)).item())


class ObjectOnlyLinear(nn.Module):
    """Small model used only to generate subtraction reconstructions."""

    def __init__(
        self,
        *,
        num_objects: int,
        num_possible_objects: int,
        output_dim: int,
        positional_concepts: bool,
        positional_objects: bool,
        num_concepts: int,
        num_values_per_concept: List[int],
        pool: str,
        use_objects: bool,
        use_concepts: bool,
    ):
        super().__init__()
        if pool not in {"sum", "mlp"}:
            raise ValueError("pool must be one of {'sum','mlp'}")
        if not (use_objects or use_concepts):
            raise ValueError("At least one of use_objects/use_concepts must be True")

        self.num_objects = int(num_objects)
        self.output_dim = int(output_dim)
        self.positional_concepts = bool(positional_concepts)
        self.positional_objects = bool(positional_objects)
        self.use_objects = bool(use_objects)
        self.use_concepts = bool(use_concepts)
        self.pool = pool
        self.num_concepts = int(num_concepts)
        self.num_values_per_concept = [int(v) for v in num_values_per_concept]

        if self.use_objects:
            if self.positional_objects:
                self.table_objects = nn.ModuleList(
                    [nn.Embedding(int(num_possible_objects), self.output_dim) for _ in range(self.num_objects)]
                )
            else:
                self.table_objects = nn.Embedding(int(num_possible_objects), self.output_dim)

        if self.use_concepts:
            if self.positional_concepts:
                self.table_concepts = nn.ModuleList(
                    [
                        nn.Embedding(self.num_values_per_concept[c], self.output_dim)
                        for c in range(self.num_concepts)
                        for _ in range(self.num_objects)
                    ]
                )
            else:
                self.table_concepts = nn.ModuleList(
                    [nn.Embedding(self.num_values_per_concept[c], self.output_dim) for c in range(self.num_concepts)]
                )

        if self.pool == "mlp":
            in_dim = 0
            if self.use_objects:
                in_dim += self.num_objects * self.output_dim
            if self.use_concepts:
                in_dim += self.num_objects * self.num_concepts * self.output_dim
            self.W = nn.Sequential(nn.Linear(in_dim, self.output_dim), nn.ReLU(), nn.Linear(self.output_dim, self.output_dim))
        else:
            self.W = None

    def forward(self, object_ids: Optional[torch.Tensor], scene_concepts: Optional[torch.Tensor]):
        vecs_objects: Optional[List[torch.Tensor]] = None
        vec_concepts: Optional[List[torch.Tensor]] = None

        if self.use_objects:
            if object_ids is None:
                raise ValueError("object_ids is required when use_objects=True")
            if object_ids.shape[1] != self.num_objects:
                raise ValueError(f"Expected object_ids shape [B,{self.num_objects}], got {list(object_ids.shape)}")
            if self.positional_objects:
                vecs_objects = [self.table_objects[o](object_ids[:, o]) for o in range(self.num_objects)]
            else:
                vecs_objects = [self.table_objects(object_ids[:, o]) for o in range(self.num_objects)]

        if self.use_concepts:
            if scene_concepts is None:
                raise ValueError("scene_concepts is required when use_concepts=True")
            if scene_concepts.dim() != 3 or scene_concepts.shape[1] != self.num_objects or scene_concepts.shape[2] != self.num_concepts:
                raise ValueError(
                    f"Expected scene_concepts shape [B,{self.num_objects},{self.num_concepts}], got {list(scene_concepts.shape)}"
                )
            if self.positional_concepts:
                vec_concepts = []
                for o in range(self.num_objects):
                    for c in range(self.num_concepts):
                        tok = scene_concepts[:, o, c]
                        vec_concepts.append(self.table_concepts[c * self.num_objects + o](tok))
            else:
                vec_concepts = [
                    self.table_concepts[c](scene_concepts[:, o, c]) for o in range(self.num_objects) for c in range(self.num_concepts)
                ]

        if self.pool == "mlp":
            parts: List[torch.Tensor] = []
            if vecs_objects is not None:
                parts.append(torch.cat(vecs_objects, dim=-1))
            if vec_concepts is not None:
                parts.append(torch.cat(vec_concepts, dim=-1))
            return self.W(torch.cat(parts, dim=-1))

        out: torch.Tensor = torch.tensor(0.0, device=(vecs_objects[0].device if vecs_objects else vec_concepts[0].device))  # type: ignore[index]
        if vecs_objects is not None:
            out = out + torch.sum(torch.stack(vecs_objects, dim=1), dim=1)
        if vec_concepts is not None:
            out = out + torch.sum(torch.stack(vec_concepts, dim=1), dim=1)
        return out


def fit_model(
    model: ObjectOnlyLinear,
    X_obj_tr: torch.Tensor,
    X_con_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_obj_te: torch.Tensor,
    X_con_te: torch.Tensor,
    Y_te: torch.Tensor,
    *,
    steps: int,
    lr: float,
    batch_size: int = 2048,
    weight_decay: float = 0.0,
    seed: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    use_obj = bool(model.use_objects)
    use_con = bool(model.use_concepts)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    N = int(Y_tr.shape[0])
    g = torch.Generator().manual_seed(int(seed))

    model.train()
    for step in range(int(steps)):
        idx = torch.randint(0, N, (int(batch_size),), generator=g)
        xb_obj = X_obj_tr[idx].to(device) if use_obj else None
        xb_con = X_con_tr[idx].to(device) if use_con else None
        yb = Y_tr[idx].to(device)

        pred = model(xb_obj, xb_con)
        loss = loss_fn(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                sl_te = slice(0, min(8192, int(Y_te.shape[0])))
                xo = X_obj_te[sl_te].to(device) if use_obj else None
                xc = X_con_te[sl_te].to(device) if use_con else None
                yt = Y_te[sl_te].to(device)
                yp = model(xo, xc)
                r2_te = r2_score(yt.cpu(), yp.cpu())

                sl_tr = slice(0, min(8192, int(Y_tr.shape[0])))
                xo = X_obj_tr[sl_tr].to(device) if use_obj else None
                xc = X_con_tr[sl_tr].to(device) if use_con else None
                yt = Y_tr[sl_tr].to(device)
                yp = model(xo, xc)
                r2_tr = r2_score(yt.cpu(), yp.cpu())

            model.train()
            print(f"step {step+1:5d} | loss {loss.item():.4f} | approx R2 train/test {r2_tr:.4f}/{r2_te:.4f}")

    model.eval()
    with torch.no_grad():
        xo = X_obj_tr.to(device) if use_obj else None
        xc = X_con_tr.to(device) if use_con else None
        yt = Y_tr.to(device)
        train_r2 = r2_score(yt.cpu(), model(xo, xc).cpu())

        xo = X_obj_te.to(device) if use_obj else None
        xc = X_con_te.to(device) if use_con else None
        yt = Y_te.to(device)
        test_r2 = r2_score(yt.cpu(), model(xo, xc).cpu())

    return train_r2, test_r2


@dataclass
class RunRecord:
    embedding_path: str
    dataset_path: str
    variant: str  # base | concepts_no_perm | concepts_perm | objects
    produced_embedding_path: Optional[str]
    summary_metrics: Optional[Dict[str, Any]]
    dataset_metadata: Optional[Dict[str, Any]]
    returncode: int
    duration_sec: float
    stdout: str
    stderr: str


@dataclass
class SuiteResult:
    created_at: str
    out_dir: str
    runs: List[RunRecord]
    created_subtractions: List[str]
    subtraction_fits: List[Dict[str, Any]]


def _subtraction_dir(out_dir: Path, embedding_pkl: Path):
    # Create a per-embedding folder so filenames don't collide.
    safe = embedding_pkl.stem.replace(os.sep, "_")
    return out_dir / "subtractions" / safe


def _inputs_from_path(
    path: Path,
    *,
    dataset_path: Path,
):
    """Return list of (embedding_pkl, dataset_path) to run the suite on.

    Supports only base embeddings in `.pkl` format.
    """
    if not path.exists():
        raise FileNotFoundError(f"Embedding path not found: {path}")

    if path.is_dir():
        candidates = sorted(
            p for p in path.glob("*_embeddings.pkl") if p.is_file() and p.name != "dataset.pkl"
        )
        if not candidates:
            raise FileNotFoundError(
                f"No *_embeddings.pkl files found in directory: {path}"
            )
        if len(candidates) > 1:
            names = "\n".join(str(p) for p in candidates)
            raise ValueError(
                "--embedding-path points to a directory with multiple embedding files. "
                "Pass a specific file path instead. Candidates:\n"
                f"{names}"
            )
        path = candidates[0]

    if path.suffix != ".pkl":
        raise ValueError(f"--embedding-path must be a .pkl file, got: {path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.pkl not found: {dataset_path}")
    return [(path, dataset_path)]


def make_subtractions(
    *,
    dataset: list,
    metadata: dict,
    base_embeddings_pkl: Path,
    out_dir: Path,
    steps: int,
    lr: float,
    seed: int,
    do_concepts: bool,
    do_objects: bool,
    positional_concepts: bool,
    positional_objects: bool,
):
    base_arr = _read_pickle_embeddings(base_embeddings_pkl)

    num_objects = int(metadata.get("num_objects", 2))
    if num_objects != 2:
        raise ValueError(f"This suite currently assumes num_objects==2, got {num_objects}")

    values = metadata.get("values_per_concept", metadata.get("num_values_per_concept"))
    if values is None:
        raise KeyError("metadata.json must include values_per_concept or num_values_per_concept")
    if isinstance(values, int):
        values_per_concept = [int(values)] * int(metadata["num_concepts"])
    else:
        values_per_concept = [int(v) for v in values]

    # select 2-object scenes (this is what your notebook does)
    two_ids = [i for i, scene in enumerate(dataset) if len(scene) == 2]
    if not two_ids:
        raise ValueError("No 2-object scenes found in dataset.")

    # clip_embeddings pipeline is now strict: embeddings must already be
    # aligned to 2-object scenes (same row count as two_ids).
    if base_arr.shape[0] != len(two_ids):
        raise ValueError(
            "Expected embeddings aligned to 2-object scenes only. "
            f"Got embedding rows={base_arr.shape[0]}, dataset scenes={len(dataset)}, two-object scenes={len(two_ids)}. "
            "Regenerate embeddings with clip_embeddings/train_embeddings.py."
        )
    emb_two = base_arr

    Y = torch.as_tensor(emb_two, dtype=torch.float32)
    Y = Y - Y.mean(dim=0, keepdim=True)

    object_to_id = _build_object_id_map(dataset)
    num_possible_objects = len(object_to_id)
    two_scenes = [dataset[i] for i in two_ids]
    object_ids = np.array([[object_to_id[tuple(map(int, obj))] for obj in scene] for scene in two_scenes], dtype=np.int64)
    X_obj = torch.as_tensor(object_ids, dtype=torch.long)
    X_con = torch.as_tensor(np.array(two_scenes, dtype=np.int64), dtype=torch.long)

    # train/test split
    N = int(Y.shape[0])
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(N, generator=g)
    train_n = int(0.99 * N)
    tr_idx, te_idx = perm[:train_n], perm[train_n:]
    X_obj_tr, X_obj_te = X_obj[tr_idx], X_obj[te_idx]
    X_con_tr, X_con_te = X_con[tr_idx], X_con[te_idx]
    Y_tr, Y_te = Y[tr_idx], Y[te_idx]

    created: Dict[str, Path] = {}
    created_list: List[Path] = []
    subtraction_fits: List[Dict[str, Any]] = []

    sub_dir = _subtraction_dir(out_dir, base_embeddings_pkl)
    sub_dir.mkdir(parents=True, exist_ok=True)

    def write_variant(name: str, recon: np.ndarray):
        out_arr = base_arr.copy()
        out_arr[: recon.shape[0]] = out_arr[: recon.shape[0]] - recon
        # Include subtraction-model config in filename to avoid overwrites.
        out_path = sub_dir / f"{name}_posC{int(positional_concepts)}_posO{int(positional_objects)}_{base_embeddings_pkl.name}"
        _save_pickle(out_path, out_arr)
        return out_path

    if do_objects:
        model_obj = ObjectOnlyLinear(
            num_objects=2,
            num_possible_objects=num_possible_objects,
            output_dim=int(Y.shape[1]),
            positional_concepts=False,
            positional_objects=bool(positional_objects),
            num_concepts=int(metadata["num_concepts"]),
            num_values_per_concept=values_per_concept,
            pool="sum",
            use_objects=True,
            use_concepts=False,
        )
        tr, te = fit_model(model_obj, X_obj_tr, X_con_tr, Y_tr, X_obj_te, X_con_te, Y_te, steps=steps, lr=lr, seed=seed)
        print(f"[subtraction objects] R2 train/test: {tr:.4f}/{te:.4f}")
        subtraction_fits.append(
            {
                "base_embeddings_pkl": str(base_embeddings_pkl),
                "mode": "objects",
                "r2_train": float(tr),
                "r2_test": float(te),
                "steps": int(steps),
                "lr": float(lr),
                "seed": int(seed),
                "positional_objects": bool(positional_objects),
                "index_mode": "two_only",
                "num_two_object_scenes": int(len(two_ids)),
                "embedding_dim": int(Y.shape[1]),
                "perm_variants": ["no_obj", "perm_no_obj"],
            }
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            recon = model_obj.to(device)(X_obj.to(device), None).to("cpu").float().numpy()
        p = write_variant("no_obj", recon)
        created["objects_no_perm"] = p
        created_list.append(p)

        g_perm_obj = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(int(num_possible_objects), generator=g_perm_obj)
        X_obj_perm = perm[X_obj]
        with torch.no_grad():
            recon_perm = model_obj.to(device)(X_obj_perm.to(device), None).to("cpu").float().numpy()
        p_perm = write_variant("perm_no_obj", recon_perm)
        created["objects_perm"] = p_perm
        created_list.append(p_perm)

    if do_concepts:
        model_con = ObjectOnlyLinear(
            num_objects=2,
            num_possible_objects=num_possible_objects,
            output_dim=int(Y.shape[1]),
            positional_concepts=bool(positional_concepts),
            positional_objects=False,
            num_concepts=int(metadata["num_concepts"]),
            num_values_per_concept=values_per_concept,
            pool="sum",
            use_objects=False,
            use_concepts=True,
        )
        tr, te = fit_model(model_con, X_obj_tr, X_con_tr, Y_tr, X_obj_te, X_con_te, Y_te, steps=steps, lr=lr, seed=seed)
        print(f"[subtraction concepts] R2 train/test: {tr:.4f}/{te:.4f}")
        subtraction_fits.append(
            {
                "base_embeddings_pkl": str(base_embeddings_pkl),
                "mode": "concepts",
                "r2_train": float(tr),
                "r2_test": float(te),
                "steps": int(steps),
                "lr": float(lr),
                "seed": int(seed),
                "positional_concepts": bool(positional_concepts),
                "index_mode": "two_only",
                "num_two_object_scenes": int(len(two_ids)),
                "embedding_dim": int(Y.shape[1]),
                "perm_variants": ["False_no_concepts", "True_no_concepts"],
            }
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            recon = model_con.to(device)(None, X_con.to(device)).to("cpu").float().numpy()
        p = write_variant("False_no_concepts", recon)
        created["concepts_no_perm"] = p
        created_list.append(p)

        # permute each concept's token ids (deterministic)
        g_perm = torch.Generator().manual_seed(int(seed))
        num_vals_here = [int(X_con[:, :, i].max().item() + 1) for i in range(int(metadata["num_concepts"]))]
        perms = [torch.randperm(n, generator=g_perm) for n in num_vals_here]
        X_con_perm = torch.empty_like(X_con)
        for i in range(int(metadata["num_concepts"])):
            X_con_perm[:, :, i] = perms[i][X_con[:, :, i]]

        with torch.no_grad():
            recon = model_con.to(device)(None, X_con_perm.to(device)).to("cpu").float().numpy()
        p = write_variant("True_no_concepts", recon)
        created["concepts_perm"] = p
        created_list.append(p)

    return created, created_list, subtraction_fits


def run_train_embeddings(dataset_path: Path, embedding_path: Path):
    t0 = time.time()
    out = train_embeddings_mod.main(
        [
            "--dataset-path",
            str(dataset_path),
            "--embedding_path",
            str(embedding_path),
        ]
    )
    if not isinstance(out, dict):
        raise TypeError(f"train_embeddings.main must return a dict, got: {type(out)}")

    if out.get("training_skipped", False):
        reason = str(out.get("reason", "training_skipped"))
        raise RuntimeError(f"train_embeddings skipped for {embedding_path}: {reason}")

    produced = out.get("embeddings_saved")
    summary_metrics: Optional[Dict[str, Any]] = out.get("summary_metrics")
    dt = time.time() - t0

    return RunRecord(
        embedding_path=str(embedding_path),
        dataset_path=str(dataset_path),
        variant="base",
        produced_embedding_path=(None if produced is None else str(produced)),
        summary_metrics=summary_metrics,
        dataset_metadata=None,
        returncode=0,
        duration_sec=float(dt),
        stdout="",
        stderr="",
    )


def run_suite_for_embedding(
    *,
    embedding_pkl: Path,
    dataset_path: Path,
    out_dir: Path,
    steps: int,
    lr: float,
    seed: int,
    positional_concepts: bool,
    positional_objects: bool,
):
    dataset, metadata = load_dataset(dataset_path)
    if metadata is None:
        raise ValueError(f"metadata.json not found next to {dataset_path} (required for the subtraction suite).")
    run_dataset_metadata = {
        "num_scenes": int(len(dataset)),
        "num_objects": metadata.get("num_objects"),
        "num_concepts": metadata.get("num_concepts"),
        "values_per_concept": metadata.get("values_per_concept", metadata.get("num_values_per_concept")),
    }

    runs: List[RunRecord] = []
    created_subtractions: List[Path] = []
    subtraction_fits: List[Dict[str, Any]] = []

    rr = run_train_embeddings(dataset_path=dataset_path, embedding_path=embedding_pkl)
    rr.variant = "base"
    rr.dataset_metadata = run_dataset_metadata
    runs.append(rr)

    created_map, created_subtractions, subtraction_fits = make_subtractions(
        dataset=dataset,
        metadata=metadata,
        base_embeddings_pkl=embedding_pkl,
        out_dir=out_dir,
        steps=steps,
        lr=lr,
        seed=seed,
        do_concepts=True,
        do_objects=True,
        positional_concepts=positional_concepts,
        positional_objects=positional_objects,
    )

    p1 = created_map["concepts_no_perm"]
    r1 = run_train_embeddings(dataset_path=dataset_path, embedding_path=p1)
    r1.variant = "concepts_no_perm"
    r1.dataset_metadata = run_dataset_metadata
    runs.append(r1)

    p2 = created_map["concepts_perm"]
    r2 = run_train_embeddings(dataset_path=dataset_path, embedding_path=p2)
    r2.variant = "concepts_perm"
    r2.dataset_metadata = run_dataset_metadata
    runs.append(r2)

    p = created_map["objects_no_perm"]
    r = run_train_embeddings(dataset_path=dataset_path, embedding_path=p)
    r.variant = "objects"
    r.dataset_metadata = run_dataset_metadata
    runs.append(r)

    pp = created_map["objects_perm"]
    rr = run_train_embeddings(dataset_path=dataset_path, embedding_path=pp)
    rr.variant = "objects_perm"
    rr.dataset_metadata = run_dataset_metadata
    runs.append(rr)

    return runs, created_subtractions, subtraction_fits


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Run probe suite and save results.json.")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory.")
    p.add_argument("--steps", type=int, default=10_000, help="Subtraction fit steps.")
    p.add_argument("--lr", type=float, default=1e-2, help="Subtraction fit LR.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--dataset-path", type=str, required=True, help="Path to dataset.pkl.")
    p.add_argument(
        "--embedding-path",
        type=str,
        action="append",
        required=True,
        help="Path to base embeddings .pkl, or directory with exactly one *_embeddings.pkl (repeat to provide multiple).",
    )

    args = p.parse_args(argv)

    if args.out_dir:
        requested_out_dir = Path(args.out_dir)
        out_dir = requested_out_dir.parent / f"{requested_out_dir.name}_{_unique_suffix()}"
        print({"requested_out_dir": str(requested_out_dir), "resolved_out_dir": str(out_dir)})
    else:
        first_embedding_path = Path(args.embedding_path[0])
        out_dir = first_embedding_path.parent / f"probe_suite_{_now_tag()}"
        print({"resolved_out_dir": str(out_dir), "mode": "next_to_embedding"})
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "results.json"

    all_runs: List[RunRecord] = []
    all_created: List[Path] = []
    all_subtraction_fits: List[Dict[str, Any]] = []
    dataset_path = Path(args.dataset_path)

    for input_str in args.embedding_path:
        input_path = Path(input_str)
        pairs = _inputs_from_path(input_path, dataset_path=dataset_path)
        for emb_pkl, ds in pairs:
            runs, created, subtraction_fits = run_suite_for_embedding(
                embedding_pkl=emb_pkl,
                dataset_path=ds,
                out_dir=out_dir,
                steps=int(args.steps),
                lr=float(args.lr),
                seed=int(args.seed),
                positional_concepts=True,
                positional_objects=True,
            )
            all_runs.extend(runs)
            all_created.extend(created)
            all_subtraction_fits.extend(subtraction_fits)

    suite = SuiteResult(
        created_at=datetime.now().isoformat(),
        out_dir=str(out_dir),
        runs=all_runs,
        created_subtractions=[str(p) for p in all_created],
        subtraction_fits=all_subtraction_fits,
    )
    if suite.created_subtractions and not suite.subtraction_fits:
        raise RuntimeError(
            "Internal error: subtraction files were created but subtraction_fits is empty."
        )

    results_json.parent.mkdir(parents=True, exist_ok=True)
    with results_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": suite.created_at,
                "out_dir": suite.out_dir,
                "created_subtractions": suite.created_subtractions,
                "subtraction_fits": suite.subtraction_fits,
                "runs": [asdict(r) for r in suite.runs],
            },
            f,
            indent=2,
        )

    print({"results_json": str(results_json), "num_runs": len(all_runs), "num_subtractions": len(all_created)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
