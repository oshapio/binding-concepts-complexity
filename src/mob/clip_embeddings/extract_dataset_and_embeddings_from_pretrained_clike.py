#!/usr/bin/env python3
"""Extract dataset.pkl + scene_embeddings.pkl from amortization checkpoints.

This converts an amortization checkpoint (e.g. model_best_test_objects.pt) into:
- dataset.pkl
- metadata.json
- scene_embeddings.pkl

The output can be used directly with approximate_complexity_scenes.py.
"""

import argparse
import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from amortization.models import SceneEncoder
from amortization.scenes import tokenize_scenes


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _rename_positional_key(state_dict):
    sd = dict(state_dict)
    if "positional_encoding.pe" in sd and "positional_embedding.pe" not in sd:
        sd["positional_embedding.pe"] = sd.pop("positional_encoding.pe")
    return sd


def _find_splits_npz(model_path: Path, model_cfg: dict):
    split_path = model_cfg.get("object_split_path")
    if split_path:
        p = Path(split_path)
        if p.exists():
            return p

    split_dir = model_path.parent.parent / "splits"
    if not split_dir.exists():
        return None
    split_files = sorted(split_dir.glob("*.npz"))
    if not split_files:
        return None
    return split_files[0]


def _tokenize_probe_concepts(
    *,
    num_concepts: int,
    num_vals_per_concept: int,
    eos_id: int,
    pad_id: int,
    max_scene_size: int,
    device: torch.device,
):
    # One token + EOS per query, where token id is concept-offset value id.
    pairs = [(c, v) for c in range(num_concepts) for v in range(num_vals_per_concept)]
    n = len(pairs)
    out = torch.full((n, max_scene_size), int(pad_id), dtype=torch.long, device=device)
    token_ids = [int(c * num_vals_per_concept + v) for c, v in pairs]
    out[:, 0] = torch.tensor(token_ids, dtype=torch.long, device=device)
    out[:, 1] = int(eos_id)
    return out


def _tokenize_probe_objects(
    *,
    objects: list[list[int]],
    soo_id: int,
    eoo_id: int,
    eos_id: int,
    pad_id: int,
    num_concepts: int,
    num_vals_per_concept: int,
    max_scene_size: int,
    device: torch.device,
):
    n = len(objects)
    out = torch.full((n, max_scene_size), int(pad_id), dtype=torch.long, device=device)
    concept_shift = np.arange(num_concepts, dtype=np.int64) * int(num_vals_per_concept)
    out[:, 0] = int(soo_id)
    out[:, 1 + num_concepts] = int(eoo_id)
    out[:, 1 + num_concepts + 1] = int(eos_id)
    for i, obj in enumerate(objects):
        obj_arr = np.asarray(obj, dtype=np.int64)
        shifted = obj_arr + concept_shift
        out[i, 1 : 1 + num_concepts] = torch.tensor(shifted, dtype=torch.long, device=device)
    return out


def extract_dataset_and_embeddings(
    *,
    model_path: Path,
    output_root: Path,
    objects_source: str,
    batch_size: int,
    normalize_embeddings: bool,
    max_objs: int,
    output_name: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if "config" not in model:
        raise KeyError(f"Checkpoint missing 'config': {model_path}")
    cfg = model["config"]

    scene_state = model.get("scene_encoder")
    probe_state = model.get("probe_encoder")
    if scene_state is None or probe_state is None:
        raise KeyError("Checkpoint must include scene_encoder and probe_encoder state dicts.")

    use_cliplike = bool(cfg.get("use_cliplike_text_encoder", False))
    if not use_cliplike:
        params = cfg.get("params")
        if isinstance(params, dict):
            use_cliplike = bool(params.get("use_cliplike_text_encoder", False))
    if not use_cliplike:
        if any(k.startswith("transformer.") for k in scene_state) or "text_projection" in scene_state:
            use_cliplike = True

    num_concepts = int(cfg["num_concepts"])
    raw_vals = cfg["num_vals_per_concept"]
    if isinstance(raw_vals, (list, tuple, np.ndarray)):
        vals_set = {int(v) for v in raw_vals}
        if len(vals_set) != 1:
            raise ValueError("Expected same num_vals_per_concept for all concepts.")
        num_vals_per_concept = int(next(iter(vals_set)))
    else:
        num_vals_per_concept = int(raw_vals)

    max_num_objects = int(cfg["max_num_objects"])
    if max_num_objects < 2:
        raise ValueError(f"max_num_objects must be >=2, got {max_num_objects}")

    soo_id = int(cfg["SOO_id"])
    eoo_id = int(cfg["EOO_id"])
    eos_id = int(cfg["EOS_id"])
    pad_id = int(cfg["PAD_token"])
    vocab_size = int(cfg["vocab_size"])
    max_scene_size = int(cfg["max_scene_size"])

    scene_encoder = SceneEncoder(
        d_model=int(cfg["d_model"]),
        d_out_size=int(cfg["d_out_size"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["num_layers"]),
        vocab_size=vocab_size,
        max_scene_size=max_scene_size,
        pad_token=pad_id,
        use_cliplike_text_encoder=use_cliplike,
    )
    probe_encoder = SceneEncoder(
        d_model=int(cfg["d_model"]),
        d_out_size=int(cfg.get("d_out_size_probe", cfg["d_out_size"])),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["num_layers"]),
        vocab_size=vocab_size,
        max_scene_size=max_scene_size,
        pad_token=pad_id,
        use_cliplike_text_encoder=use_cliplike,
    )

    scene_encoder.load_state_dict(_rename_positional_key(scene_state), strict=True)
    probe_encoder.load_state_dict(_rename_positional_key(probe_state), strict=True)
    scene_encoder = scene_encoder.to(device).eval()
    probe_encoder = probe_encoder.to(device).eval()

    num_vals_take = num_vals_per_concept
    if objects_source in {"train", "test"}:
        split_path = _find_splits_npz(model_path, cfg)
        if split_path is None:
            raise FileNotFoundError("No split .npz found for objects_source=train/test.")
        splits = np.load(str(split_path))
        key = "train_objects" if objects_source == "train" else "test_objects"
        objects = splits[key].astype(np.int64).tolist()
    elif objects_source == "all":
        num_vals_take = min(
            num_vals_per_concept,
            int(math.ceil(int(max_objs) ** (1.0 / float(num_concepts)))),
        )
        objects = [list(vals) for vals in itertools.product(range(num_vals_take), repeat=num_concepts)]
    else:
        raise ValueError(f"Unknown objects_source: {objects_source}")

    if not objects:
        raise ValueError("No objects available for scene construction.")

    scenes = [[list(a), list(b)] for a in objects for b in objects]

    out_dir = output_root / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "dataset.pkl"
    with dataset_path.open("wb") as f:
        pickle.dump(scenes, f, protocol=pickle.HIGHEST_PROTOCOL)

    with torch.inference_mode():
        emb_chunks = []
        for s in tqdm(range(0, len(scenes), int(batch_size)), desc="Scene embeddings"):
            batch = scenes[s : s + int(batch_size)]
            tok = tokenize_scenes(
                batch,
                soo_id,
                eoo_id,
                eos_id,
                pad_id,
                num_concepts,
                num_vals_per_concept,
                max_scene_size,
            ).to(device)
            emb = scene_encoder(tok)
            if normalize_embeddings:
                emb = emb / (1e-6 + torch.norm(emb, dim=-1, keepdim=True))
            emb_chunks.append(emb.cpu())
        scene_embeddings = torch.cat(emb_chunks, dim=0)

        concept_tok = _tokenize_probe_concepts(
            num_concepts=num_concepts,
            num_vals_per_concept=num_vals_per_concept,
            eos_id=eos_id,
            pad_id=pad_id,
            max_scene_size=max_scene_size,
            device=device,
        )
        concept_vecs = probe_encoder(concept_tok)
        concept_vecs = concept_vecs / (1e-6 + torch.norm(concept_vecs, dim=-1, keepdim=True))

        object_tok = _tokenize_probe_objects(
            objects=objects,
            soo_id=soo_id,
            eoo_id=eoo_id,
            eos_id=eos_id,
            pad_id=pad_id,
            num_concepts=num_concepts,
            num_vals_per_concept=num_vals_per_concept,
            max_scene_size=max_scene_size,
            device=device,
        )
        object_vecs = probe_encoder(object_tok)
        object_vecs = object_vecs / (1e-6 + torch.norm(object_vecs, dim=-1, keepdim=True))

    scene_dim = int(scene_embeddings.shape[1])
    if int(concept_vecs.shape[1]) != scene_dim or int(object_vecs.shape[1]) != scene_dim:
        raise ValueError(
            "Probe embedding dim does not match scene embedding dim. "
            f"scene={scene_dim}, concept_probe={int(concept_vecs.shape[1])}, object_probe={int(object_vecs.shape[1])}"
        )

    concept_probes = nn.ModuleDict({})
    for c in range(num_concepts):
        lin = nn.Linear(scene_dim, num_vals_per_concept, bias=False)
        lo = c * num_vals_per_concept
        hi = (c + 1) * num_vals_per_concept
        lin.weight.data = concept_vecs[lo:hi].cpu()
        concept_probes[f"{c}"] = lin

    object_probes = nn.ModuleDict({})
    lin_obj = nn.Linear(scene_dim, len(objects), bias=False)
    lin_obj.weight.data = object_vecs.cpu()
    object_probes["1"] = lin_obj

    embeddings_obj = {
        "embeddings": scene_embeddings,
        "probes_single_concept": concept_probes,
        "probes_object": object_probes,
    }

    emb_path = out_dir / "scene_embeddings.pkl"
    with emb_path.open("wb") as f:
        pickle.dump(embeddings_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "num_objects": 2,
        "num_concepts": num_concepts,
        "values_per_concept": [int(num_vals_take)] * num_concepts,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print({"dataset_path": str(dataset_path), "embeddings_path": str(emb_path), "output_dir": str(out_dir)})


def parse_args():
    repo_root = _default_repo_root()
    parser = argparse.ArgumentParser(
        description="Extract dataset.pkl + scene_embeddings.pkl from amortization checkpoint."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to checkpoint .pt (e.g. model_best_test_objects.pt).")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(repo_root / "data" / "clip_checks_public" / "amortization"),
        help="Output root for extracted dataset/embeddings.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="",
        help="Optional output subfolder name. Default: parent run folder name from model path.",
    )
    parser.add_argument(
        "--objects-source",
        type=str,
        choices=("all", "train", "test"),
        default="all",
        help="Which object set to use for scene construction.",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for encoding scenes.")
    parser.add_argument("--max-objs", type=int, default=400, help="Max objects when objects-source=all.")
    parser.add_argument(
        "--normalize-embeddings",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="L2-normalize scene embeddings before saving.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model-path not found: {model_path}")
    if model_path.suffix != ".pt":
        raise ValueError(f"model-path must be a .pt checkpoint: {model_path}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_name = args.output_name.strip() if args.output_name else model_path.parent.parent.name.replace("-", "_")

    extract_dataset_and_embeddings(
        model_path=model_path,
        output_root=output_root,
        output_name=output_name,
        objects_source=args.objects_source,
        batch_size=int(args.batch_size),
        normalize_embeddings=bool(args.normalize_embeddings),
        max_objs=int(args.max_objs),
    )


if __name__ == "__main__":
    main()
