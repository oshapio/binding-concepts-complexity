from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import tempfile
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dataset_io import load_dataset
import train_embeddings as train_embeddings_mod

# High-level flow in this script:
# 1) Load scene embeddings + labels from dataset.pkl
# 2) Build object embeddings according to selected mode
# 3) Construct steering vectors and intervened scene embeddings
# 4) Evaluate via retrieval (and optionally probes)
# 5) Save summary + per-case outputs


@dataclass
class PairRecord:
    attr1: int
    obj1: int
    attr2: int
    obj2: int
    pos1: str
    pos2: str


@dataclass
class RawPairRecord:
    attr1: str
    obj1: str
    attr2: str
    obj2: str
    pos1: str
    pos2: str


@dataclass
class InterventionCase:
    base_pair: Tuple[int, int, int, int]
    target_attr: int
    is_left: bool
    same_color: bool
    control_pair: Tuple[int, int, int, int]
    control_index: int
    predicted_index: int
    retrieval_top1_hit: bool
    probe_joint_hit: Optional[bool]


@dataclass
class LoadedData:
    records: List[PairRecord]
    embeddings: np.ndarray
    attr_to_idx: Dict[str, int]
    obj_to_idx: Dict[str, int]
    attr_concept_idx: int
    obj_concept_idx: int


@dataclass
class ProbeArtifacts:
    probes_single_concept: nn.ModuleDict
    probes_object: Optional[nn.ModuleDict]
    similarity: str
    sim_scale: float
    source: str


def _norm_token(value: object) -> str:
    return str(value).strip().lower()


def _load_embedding_array(path: Path) -> np.ndarray:
    """Load scene embeddings and normalize to a 2D float array [N, D]."""
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    elif suffix == ".npy":
        payload = np.load(path)
    else:
        with path.open("rb") as f:
            payload = pickle.load(f)

    if isinstance(payload, dict) and "embeddings" in payload:
        payload = payload["embeddings"]
    if isinstance(payload, torch.Tensor):
        payload = payload.detach().cpu().numpy()

    arr = np.asarray(payload)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 embeddings [N,D], got shape={arr.shape} from {path}")
    return arr.astype(np.float32)


def _load_embedding_payload(path: Path) -> object:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    if suffix == ".npy":
        return np.load(path)
    with path.open("rb") as f:
        return pickle.load(f)


def _extract_pair_from_text(label: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse a text pair like 'red cube and blue sphere' into (a1,o1,a2,o2)."""
    parts = [p.strip() for p in re.split(r",|\band\b", label, flags=re.IGNORECASE) if p.strip()]
    if len(parts) < 2:
        return None

    def _parse_attr_obj(chunk: str) -> Optional[Tuple[str, str]]:
        tokens = [t for t in chunk.split() if t]
        if len(tokens) < 2:
            return None
        return _norm_token(tokens[0]), _norm_token(tokens[1])

    left = _parse_attr_obj(parts[0])
    right = _parse_attr_obj(parts[1])
    if left is None or right is None:
        return None
    return left[0], left[1], right[0], right[1]


def _extract_meta_value_names(meta: dict, concept_idx: int) -> Optional[Sequence[object]]:
    candidates = [
        meta.get("concept_value_names"),
        meta.get("concept_values"),
        meta.get("values_per_concept"),
        meta.get("value_names"),
    ]
    for cand in candidates:
        if isinstance(cand, Sequence) and len(cand) > concept_idx:
            values = cand[concept_idx]
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                return values
    return None


def _infer_concept_idx(meta: dict, fallback: int, candidates: Sequence[str]) -> int:
    concept_names = meta.get("concept_names")
    if not isinstance(concept_names, Sequence):
        return int(fallback)
    for i, name in enumerate(concept_names):
        name_norm = _norm_token(name)
        if any(c in name_norm for c in candidates):
            return i
    return int(fallback)


def _build_vocab(records: Sequence[RawPairRecord], side: str, names_from_meta: Optional[Sequence[object]]) -> Dict[str, int]:
    value_to_idx: Dict[str, int] = {}
    if names_from_meta is not None:
        for i, value in enumerate(names_from_meta):
            value_to_idx[_norm_token(value)] = int(i)

    values_set = ({r.attr1 for r in records} | {r.attr2 for r in records}) if side == "attr" else ({r.obj1 for r in records} | {r.obj2 for r in records})

    def _is_int_token(s: str) -> bool:
        return bool(re.fullmatch(r"-?\d+", s))

    if values_set and all(_is_int_token(v) for v in values_set):
        values = sorted(values_set, key=lambda s: int(s))
    else:
        values = sorted(values_set)

    for value in values:
        if value not in value_to_idx:
            value_to_idx[value] = len(value_to_idx)
    return value_to_idx


def _record_from_mapping(row: Dict[str, object]) -> Optional[RawPairRecord]:
    keys = {k.strip().lower(): k for k in row.keys()}

    if all(name in keys for name in ["attr1", "obj1", "attr2", "obj2"]):
        pos1_key = keys.get("pos1")
        pos2_key = keys.get("pos2")
        return RawPairRecord(
            attr1=_norm_token(row[keys["attr1"]]),
            obj1=_norm_token(row[keys["obj1"]]),
            attr2=_norm_token(row[keys["attr2"]]),
            obj2=_norm_token(row[keys["obj2"]]),
            pos1=_norm_token(row[pos1_key]) if pos1_key else "left",
            pos2=_norm_token(row[pos2_key]) if pos2_key else "right",
        )

    pair_key = None
    for candidate in ["pair_label", "pair", "caption", "text", "label"]:
        if candidate in keys:
            pair_key = keys[candidate]
            break
    if pair_key is None:
        return None

    parsed = _extract_pair_from_text(str(row[pair_key]))
    if parsed is None:
        return None
    return RawPairRecord(
        attr1=parsed[0],
        obj1=parsed[1],
        attr2=parsed[2],
        obj2=parsed[3],
        pos1="left",
        pos2="right",
    )


def _load_dataset_records(
    *,
    dataset_path: Path,
    embeddings: np.ndarray,
    adapter: str,
    world_name: Optional[str],
    require_character_pos_null: bool,
) -> LoadedData:
    """Load and align pair labels with scene embeddings from dataset.pkl.

    Supports three shapes inside dataset.pkl:
    - numeric 2-object scenes (CLEVR-like)
    - dict-like rows with attr/object fields
    - text labels that can be parsed into two attr-object pairs
    """
    data, meta = load_dataset(dataset_path)
    if meta is None:
        meta = {}

    default_attr_fallback = 1 if adapter == "pug_spare" else 0
    default_obj_fallback = 0 if adapter == "pug_spare" else 1

    attr_concept_idx = _infer_concept_idx(meta, fallback=default_attr_fallback, candidates=["color", "attr", "attribute", "texture"])
    obj_concept_idx = _infer_concept_idx(meta, fallback=default_obj_fallback, candidates=["shape", "object", "obj", "name"])

    raw_records: List[RawPairRecord] = []
    selected_indices: List[int] = []

    for i, sample in enumerate(data):
        # Align strictly by row index with embeddings, then filter by parseability.
        if i >= embeddings.shape[0]:
            break

        parsed: Optional[RawPairRecord] = None
        row_like: Optional[Dict[str, object]] = None

        if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)) and len(sample) == 2:
            left = sample[0]
            right = sample[1]
            if isinstance(left, Sequence) and isinstance(right, Sequence):
                max_idx = max(attr_concept_idx, obj_concept_idx)
                if len(left) > max_idx and len(right) > max_idx:
                    parsed = RawPairRecord(
                        attr1=str(int(left[attr_concept_idx])),
                        obj1=str(int(left[obj_concept_idx])),
                        attr2=str(int(right[attr_concept_idx])),
                        obj2=str(int(right[obj_concept_idx])),
                        pos1="left",
                        pos2="right",
                    )

        elif isinstance(sample, dict):
            row_like = sample
            parsed = _record_from_mapping(sample)

        elif isinstance(sample, str):
            p = _extract_pair_from_text(sample)
            if p is not None:
                parsed = RawPairRecord(attr1=p[0], obj1=p[1], attr2=p[2], obj2=p[3], pos1="left", pos2="right")

        if parsed is None:
            continue

        if adapter == "pug_spare" and row_like is not None:
            # PUG-specific filtering preserved from earlier workflows.
            world_val = _norm_token(row_like.get("world_name", ""))
            char_pos = str(row_like.get("character_pos", "")).strip()
            keep_world = not world_name or world_val == _norm_token(world_name)
            keep_pos = (not require_character_pos_null) or (char_pos == "" or char_pos.lower() == "nan")
            if not (keep_world and keep_pos):
                continue

        raw_records.append(parsed)
        selected_indices.append(i)

    if not selected_indices:
        raise RuntimeError("No compatible records found in dataset.pkl for intervention evaluation.")

    emb = embeddings[np.array(selected_indices, dtype=np.int64)]
    attr_names = _extract_meta_value_names(meta, attr_concept_idx)
    obj_names = _extract_meta_value_names(meta, obj_concept_idx)

    attr_to_idx = _build_vocab(raw_records, "attr", attr_names)
    obj_to_idx = _build_vocab(raw_records, "obj", obj_names)

    records: List[PairRecord] = []
    for rec in raw_records:
        try:
            records.append(
                PairRecord(
                    attr1=int(attr_to_idx[rec.attr1]),
                    obj1=int(obj_to_idx[rec.obj1]),
                    attr2=int(attr_to_idx[rec.attr2]),
                    obj2=int(obj_to_idx[rec.obj2]),
                    pos1=rec.pos1,
                    pos2=rec.pos2,
                )
            )
        except KeyError:
            continue

    return LoadedData(
        records=records,
        embeddings=emb,
        attr_to_idx=attr_to_idx,
        obj_to_idx=obj_to_idx,
        attr_concept_idx=attr_concept_idx,
        obj_concept_idx=obj_concept_idx,
    )


def _load_single_object_dict(
    path: Path,
    attr_to_idx: Dict[str, int],
    obj_to_idx: Dict[str, int],
) -> Dict[Tuple[int, int, str], np.ndarray]:
    """Load single-object embeddings from dict[(attr,obj,pos)] -> 1D vector.

    Expected pickle payload format:
    - key: (attr, obj, pos)
    - value: embedding vector (1D)
    """
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    else:
        with path.open("rb") as f:
            raw = pickle.load(f)

    if not isinstance(raw, dict):
        raise ValueError("single-object embeddings file must contain a dict[(attr,obj,pos)] -> embedding")

    out: Dict[Tuple[int, int, str], np.ndarray] = {}

    def _to_idx(token: object, table: Dict[str, int], label: str) -> int:
        if isinstance(token, (int, np.integer)):
            return int(token)
        token_norm = _norm_token(token)
        if re.fullmatch(r"-?\d+", token_norm):
            return int(token_norm)
        if token_norm in table:
            return int(table[token_norm])
        raise ValueError(f"Unknown {label} token in single-object key: {token}")

    for k, v in raw.items():
        if not isinstance(k, tuple) or len(k) != 3:
            raise ValueError("single-object embeddings keys must be 3-tuples: (attr,obj,pos)")
        a = _to_idx(k[0], attr_to_idx, "attr")
        o = _to_idx(k[1], obj_to_idx, "obj")
        p = _norm_token(k[2])
        vec = np.asarray(v, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(f"single-object embedding for key {(a, o, p)} must be 1D, got shape={vec.shape}")
        out[(a, o, p)] = vec

    if not out:
        raise ValueError("single-object embeddings dictionary is empty")

    return out


def _build_object_embedding_bank(
    *,
    records: Sequence[PairRecord],
    scene_embeddings: np.ndarray,
    mode: str,
    adapter: str,
    single_object_embeddings_path: Optional[str],
    attr_to_idx: Dict[str, int],
    obj_to_idx: Dict[str, int],
) -> Dict[Tuple[int, int, str], np.ndarray]:
    """Build attr-object embedding lookup used for steering vectors.

    Keys are (attr, obj, pos_tag), where pos_tag is:
    - '*' for position-independent modes
    - concrete position token for position-dependent mode
    """
    if mode == "single_object":
        if adapter == "pug_spare":
            raise ValueError("single_object mode is not supported for pug_spare.")
        if not single_object_embeddings_path:
            raise ValueError("single_object mode requires --single-object-embeddings-path")
        return _load_single_object_dict(Path(single_object_embeddings_path), attr_to_idx, obj_to_idx)

    sums: Dict[Tuple[int, int, str], np.ndarray] = {}
    counts: Dict[Tuple[int, int, str], int] = {}

    for rec, emb in zip(records, scene_embeddings):
        # Each scene contributes both object slots to the selected bank.
        keys = []
        if mode == "avg_scene_position_independent":
            keys.append((rec.attr1, rec.obj1, "*"))
            keys.append((rec.attr2, rec.obj2, "*"))
        elif mode == "avg_scene_position_dependent":
            keys.append((rec.attr1, rec.obj1, rec.pos1))
            keys.append((rec.attr2, rec.obj2, rec.pos2))
        else:
            raise ValueError(f"Unknown object embedding mode: {mode}")

        for key in keys:
            if key not in sums:
                sums[key] = np.zeros_like(emb)
                counts[key] = 0
            sums[key] += emb
            counts[key] += 1

    return {k: sums[k] / max(counts[k], 1) for k in sums}


def _lookup_object_embedding(
    bank: Dict[Tuple[int, int, str], np.ndarray],
    attr: int,
    obj: int,
    pos: str,
    mode: str,
) -> Optional[np.ndarray]:
    """Fetch one attr-object embedding for the requested mode and position."""
    if mode in {"avg_scene_position_dependent", "single_object"}:
        exact = bank.get((attr, obj, pos))
        if exact is not None:
            return exact
        if mode == "single_object":
            return bank.get((attr, obj, "*"))
        return bank.get((attr, obj, pos))
    return bank.get((attr, obj, "*"))


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_safe = np.nan_to_num(a.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    b_safe = np.nan_to_num(b.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    a_norm = np.linalg.norm(a_safe, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b_safe, axis=1, keepdims=True)
    a_norm = np.where(a_norm < 1e-12, 1.0, a_norm)
    b_norm = np.where(b_norm < 1e-12, 1.0, b_norm)

    a_n = np.nan_to_num(a_safe / a_norm, nan=0.0, posinf=0.0, neginf=0.0)
    b_n = np.nan_to_num(b_safe / b_norm, nan=0.0, posinf=0.0, neginf=0.0)

    ta = torch.from_numpy(a_n.astype(np.float32))
    tb = torch.from_numpy(b_n.astype(np.float32))
    sims = (ta @ tb.T).cpu().numpy().astype(np.float32)
    return np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_embeddings(x: np.ndarray) -> np.ndarray:
    x_safe = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(x_safe, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    x_n = x_safe / n
    return np.nan_to_num(x_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _extract_probe_artifacts(raw: object, source: str) -> Optional[ProbeArtifacts]:
    """Extract train_embeddings-style probes from a saved artifact dict."""
    if not isinstance(raw, dict):
        return None
    probes_single_concept = raw.get("probes_single_concept")
    if not isinstance(probes_single_concept, nn.ModuleDict):
        return None

    probes_single_concept = probes_single_concept.to("cpu").eval()
    probes_object = raw.get("probes_object")
    if isinstance(probes_object, nn.ModuleDict):
        probes_object = probes_object.to("cpu").eval()
    else:
        probes_object = None

    similarity = str(raw.get("similarity", "dot"))
    sim_scale = float(raw.get("sim_scale", 1.0))
    return ProbeArtifacts(
        probes_single_concept=probes_single_concept,
        probes_object=probes_object,
        similarity=similarity,
        sim_scale=sim_scale,
        source=source,
    )


def _train_probe_artifacts_with_train_embeddings(
    *,
    dataset_path: Path,
    embedding_path: Path,
    epochs: int,
) -> Optional[ProbeArtifacts]:
    """Train probes via train_embeddings.main and return saved probe modules.

    train_embeddings expects pickle embeddings in some paths, so .pt/.npy inputs
    are converted to a temporary pickle first.
    """
    temp_pickle_path: Optional[Path] = None
    train_embedding_path = embedding_path
    if embedding_path.suffix.lower() in {".pt", ".pth", ".npy"}:
        arr = _load_embedding_array(embedding_path)
        fd, tmp_path = tempfile.mkstemp(prefix="interventions_probe_emb_", suffix=".pkl")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(arr, f)
        temp_pickle_path = Path(tmp_path)
        train_embedding_path = temp_pickle_path

    out = train_embeddings_mod.main(
        [
            "--dataset-path",
            str(dataset_path),
            "--embedding_path",
            str(train_embedding_path),
            "--epochs",
            str(int(epochs)),
        ]
    )

    try:
        if not isinstance(out, dict):
            return None
        produced = out.get("embeddings_saved")
        if produced is None:
            return None
        trained_payload = torch.load(str(produced), map_location="cpu", weights_only=False)
        return _extract_probe_artifacts(trained_payload, source="trained_with_train_embeddings")
    finally:
        if temp_pickle_path is not None:
            temp_pickle_path.unlink(missing_ok=True)


def _resolve_probe_artifacts(
    *,
    use_probe: bool,
    probe_path: Optional[str],
    embedding_path: Path,
    dataset_path: Path,
    probe_epochs: int,
) -> Optional[ProbeArtifacts]:
    """Resolve probes with precedence: --probe-path > embedding file > train fallback."""
    if not use_probe:
        return None

    if probe_path:
        payload = _load_embedding_payload(Path(probe_path))
        out = _extract_probe_artifacts(payload, source="probe_path")
        if out is not None:
            return out
        raise RuntimeError("--probe-path was provided, but no train_embeddings-style probes were found.")

    payload = _load_embedding_payload(embedding_path)
    out = _extract_probe_artifacts(payload, source="embedding_path")
    if out is not None:
        return out

    out = _train_probe_artifacts_with_train_embeddings(
        dataset_path=dataset_path,
        embedding_path=embedding_path,
        epochs=int(probe_epochs),
    )
    if out is not None:
        return out
    raise RuntimeError("Failed to load/train probes via train_embeddings.py")


def _compute_logits(probe: nn.Linear, x: torch.Tensor, similarity: str, sim_scale: float) -> torch.Tensor:
    """Mirror train_embeddings logit behavior for dot/cos modes."""
    if similarity == "cos":
        xx = F.normalize(x, dim=-1)
        ww = F.normalize(probe.weight, dim=-1)
        return float(sim_scale) * (xx @ ww.t())
    return probe(x)


def _probe_set_hit(logits_1d: torch.Tensor, positives: Sequence[int]) -> Optional[bool]:
    if logits_1d.ndim != 1 or len(positives) == 0:
        return None

    c = int(logits_1d.shape[0])
    pos = sorted({int(p) for p in positives})
    if any(p < 0 or p >= c for p in pos):
        return None

    neg = [i for i in range(c) if i not in pos]
    pos_min = torch.min(logits_1d[pos])
    if len(neg) == 0:
        return bool(torch.isfinite(pos_min).item())
    neg_max = torch.max(logits_1d[neg])
    return bool((pos_min > neg_max).item())


def _eval_probe_case(
    *,
    probe_artifacts: ProbeArtifacts,
    embedding: np.ndarray,
    control_pair: Tuple[int, int, int, int],
    attr_concept_idx: int,
    obj_concept_idx: int,
) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    """Score one intervened embedding with attribute/object probes.

    Returns:
    - attr hit (set-separation criterion)
    - object hit (set-separation criterion)
    - joint hit (both true)
    """
    x = torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0)
    attr_key = str(attr_concept_idx)
    obj_key = str(obj_concept_idx)
    attr_probe = probe_artifacts.probes_single_concept[attr_key] if attr_key in probe_artifacts.probes_single_concept else None
    obj_probe = probe_artifacts.probes_single_concept[obj_key] if obj_key in probe_artifacts.probes_single_concept else None

    if not isinstance(attr_probe, nn.Linear) or not isinstance(obj_probe, nn.Linear):
        return None, None, None

    logits_a = _compute_logits(attr_probe, x, probe_artifacts.similarity, probe_artifacts.sim_scale)[0]
    logits_o = _compute_logits(obj_probe, x, probe_artifacts.similarity, probe_artifacts.sim_scale)[0]

    attr_hit = _probe_set_hit(logits_a, [int(control_pair[0]), int(control_pair[2])])
    obj_hit = _probe_set_hit(logits_o, [int(control_pair[1]), int(control_pair[3])])
    if attr_hit is None or obj_hit is None:
        return attr_hit, obj_hit, None
    return attr_hit, obj_hit, bool(attr_hit and obj_hit)


def _metric_ratio(numer: int, denom: int) -> Optional[float]:
    if denom <= 0:
        return None
    return float(numer) / float(denom)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    slug = slug.strip("._-")
    return slug or "value"


def _infer_model_token_from_embedding_path(embedding_path: str) -> str:
    stem = Path(embedding_path).stem
    stem_lower = stem.lower()

    model_markers = [
        "clip-",
        "dinov2-",
        "siglip-",
        "openclip-",
        "eva-",
        "vit-",
        "resnet-",
    ]

    for marker in model_markers:
        idx = stem_lower.find(marker)
        if idx >= 0:
            return _safe_slug(stem[idx:])

    # Fallback: remove common trailing artifact tokens and use what's left.
    fallback = re.sub(r"(_?embeddings.*)$", "", stem, flags=re.IGNORECASE)
    if fallback and fallback != stem:
        return _safe_slug(fallback)

    return "unknown_model"


def _auto_output_json_path(args: argparse.Namespace) -> Path:
    model_token = _infer_model_token_from_embedding_path(str(args.embedding_path))

    base_name = "__".join(
        [
            "interventions",
            _safe_slug(str(args.dataset)),
            _safe_slug(model_token),
            _safe_slug(str(args.object_embedding_mode)),
            f"k{float(args.steering_strength):g}",
            f"probe{int(bool(args.probe))}",
            f"seed{int(args.seed)}",
        ]
    )

    out_dir = Path(args.dataset_path) / "interventions"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = out_dir / f"{base_name}.json"
    if not candidate.exists():
        return candidate

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{base_name}__{ts}.json"


def run_interventions(
    *,
    records: Sequence[PairRecord],
    embeddings: np.ndarray,
    reference_records: Optional[Sequence[PairRecord]],
    reference_embeddings: Optional[np.ndarray],
    object_embedding_mode: str,
    object_embedding_bank: Dict[Tuple[int, int, str], np.ndarray],
    max_targets_per_pair: int,
    steering_strength: float,
    seed: int,
    retrieval_batch_size: int,
    max_interventions: Optional[int],
    probe_artifacts: Optional[ProbeArtifacts],
    attr_concept_idx: int,
    obj_concept_idx: int,
    split_same_diff: bool,
) -> Tuple[dict, List[InterventionCase]]:
    """Core intervention loop: steer each base scene toward target attributes.

    Intervened embedding is computed as:
        intervened = base - steering_strength * steering_vector
    """
    if len(records) != embeddings.shape[0]:
        raise ValueError(f"records ({len(records)}) and embeddings ({embeddings.shape[0]}) must align.")

    ref_records = reference_records if reference_records is not None else records
    ref_embeddings = reference_embeddings if reference_embeddings is not None else embeddings
    if len(ref_records) != ref_embeddings.shape[0]:
        raise ValueError(f"reference_records ({len(ref_records)}) and reference_embeddings ({ref_embeddings.shape[0]}) must align.")

    rng = random.Random(int(seed))
    attrs = sorted({r.attr1 for r in ref_records} | {r.attr2 for r in ref_records})
    pair_to_idx: Dict[Tuple[int, int, int, int], int] = {}
    for i, r in enumerate(ref_records):
        pair_to_idx.setdefault((r.attr1, r.obj1, r.attr2, r.obj2), i)

    intervened: List[np.ndarray] = []
    metadata: List[Tuple[Tuple[int, int, int, int], int, bool, bool, Tuple[int, int, int, int], int, int]] = []
    skipped_missing_object_embedding = 0
    skipped_missing_control = 0
    skipped_same_object_pair = 0

    built_interventions = 0
    cap = None if max_interventions is None or int(max_interventions) <= 0 else int(max_interventions)

    for i, rec in enumerate(records):
        # For each base scene, sample target attributes different from both originals.
        if rec.obj1 == rec.obj2:
            skipped_same_object_pair += 1
            continue

        base_pair = (rec.attr1, rec.obj1, rec.attr2, rec.obj2)
        base_emb = embeddings[i]
        same_color = rec.attr1 == rec.attr2

        targets = [a for a in attrs if a not in {rec.attr1, rec.attr2}]
        if not targets:
            continue
        if len(targets) > max_targets_per_pair:
            targets = rng.sample(targets, max_targets_per_pair)

        for target_attr in targets:
            # Build steering from source attr->target attr for each slot separately.
            a1o1 = _lookup_object_embedding(object_embedding_bank, rec.attr1, rec.obj1, rec.pos1, object_embedding_mode)
            a2o2 = _lookup_object_embedding(object_embedding_bank, rec.attr2, rec.obj2, rec.pos2, object_embedding_mode)
            ato1 = _lookup_object_embedding(object_embedding_bank, target_attr, rec.obj1, rec.pos1, object_embedding_mode)
            ato2 = _lookup_object_embedding(object_embedding_bank, target_attr, rec.obj2, rec.pos2, object_embedding_mode)
            if any(x is None for x in [a1o1, a2o2, ato1, ato2]):
                skipped_missing_object_embedding += 1
                continue

            control_left = (target_attr, rec.obj1, rec.attr2, rec.obj2)
            control_right = (rec.attr1, rec.obj1, target_attr, rec.obj2)
            if control_left not in pair_to_idx or control_right not in pair_to_idx:
                # Need ground-truth control scenes in index for retrieval/probe evaluation.
                skipped_missing_control += 1
                continue

            steering_left = a1o1 - ato1
            steering_right = a2o2 - ato2

            intervened_left = base_emb - float(steering_strength) * steering_left
            intervened_right = base_emb - float(steering_strength) * steering_right

            intervened.append(intervened_left)
            metadata.append((base_pair, target_attr, True, same_color, control_left, pair_to_idx[control_left], i))
            built_interventions += 1
            if cap is not None and built_interventions >= cap:
                break

            intervened.append(intervened_right)
            metadata.append((base_pair, target_attr, False, same_color, control_right, pair_to_idx[control_right], i))
            built_interventions += 1
            if cap is not None and built_interventions >= cap:
                break

        if cap is not None and built_interventions >= cap:
            break

    if cap is not None and built_interventions >= cap:
        print({"note": "interventions_capped", "max_interventions": int(cap), "built": int(built_interventions)})

    if not intervened:
        raise RuntimeError("No interventions generated. Check dataset coverage and object embedding mode.")

    arr_intervened = np.stack(intervened, axis=0)
    # Retrieval baseline: top-1 nearest-neighbor in original scene bank, computed in batches for memory safety.
    retrieval_batch_size = max(1, int(retrieval_batch_size))

    bank_n = _normalize_embeddings(ref_embeddings)
    query_n = _normalize_embeddings(arr_intervened)

    pred_top1 = np.zeros((query_n.shape[0],), dtype=np.int64)

    bank_t = torch.from_numpy(bank_n)

    for start in range(0, query_n.shape[0], retrieval_batch_size):
        end = min(query_n.shape[0], start + retrieval_batch_size)
        q_t = torch.from_numpy(query_n[start:end])
        sims_batch = (q_t @ bank_t.T).cpu().numpy().astype(np.float32)

        pred_top1[start:end] = np.argmax(sims_batch, axis=1)

    total = len(metadata)
    total_same = 0
    total_diff = 0

    top1_hits = 0
    top1_hits_same = 0
    top1_hits_diff = 0

    probe_joint_total = 0
    probe_joint_hits = 0
    probe_joint_same_total = 0
    probe_joint_same_hits = 0
    probe_joint_diff_total = 0
    probe_joint_diff_hits = 0

    cases: List[InterventionCase] = []

    for j, info in enumerate(metadata):
        base_pair, target_attr, is_left, same_color, control_pair, control_idx, _base_i = info
        hit_top1 = int(pred_top1[j]) == int(control_idx)

        top1_hits += int(hit_top1)

        if split_same_diff:
            if same_color:
                total_same += 1
                top1_hits_same += int(hit_top1)
            else:
                total_diff += 1
                top1_hits_diff += int(hit_top1)

        probe_joint_hit: Optional[bool] = None

        if probe_artifacts is not None:
            # Probe evaluation is optional and uses train_embeddings-compatible probes.
            _probe_attr_hit, _probe_obj_hit, probe_joint_hit = _eval_probe_case(
                probe_artifacts=probe_artifacts,
                embedding=arr_intervened[j],
                control_pair=control_pair,
                attr_concept_idx=attr_concept_idx,
                obj_concept_idx=obj_concept_idx,
            )
            if probe_joint_hit is not None:
                probe_joint_total += 1
                probe_joint_hits += int(probe_joint_hit)
                if split_same_diff:
                    if same_color:
                        probe_joint_same_total += 1
                        probe_joint_same_hits += int(probe_joint_hit)
                    else:
                        probe_joint_diff_total += 1
                        probe_joint_diff_hits += int(probe_joint_hit)

        cases.append(
            InterventionCase(
                base_pair=base_pair,
                target_attr=target_attr,
                is_left=is_left,
                same_color=same_color,
                control_pair=control_pair,
                control_index=int(control_idx),
                predicted_index=int(pred_top1[j]),
                retrieval_top1_hit=bool(hit_top1),
                probe_joint_hit=probe_joint_hit,
            )
        )

    summary: Dict[str, Any] = {
        "num_records": len(records),
        "num_reference_records": len(ref_records),
        "num_interventions": total,
        "retrieval_top1_accuracy": _metric_ratio(top1_hits, total),
        "skipped_missing_object_embedding": skipped_missing_object_embedding,
        "skipped_missing_control": skipped_missing_control,
        "skipped_same_object_pair": skipped_same_object_pair,
        "num_unique_attributes": len(attrs),
        "retrieval_batch_size": retrieval_batch_size,
        "max_interventions": (None if cap is None else int(cap)),
        "object_embedding_mode": object_embedding_mode,
        "steering_strength": float(steering_strength),
        "probe_joint_accuracy": _metric_ratio(probe_joint_hits, probe_joint_total),
        "probe_cases_scored": probe_joint_total,
    }

    if split_same_diff:
        summary.update(
            {
                "num_interventions_same_color": total_same,
                "num_interventions_diff_color": total_diff,
                "retrieval_top1_same_color_accuracy": _metric_ratio(top1_hits_same, total_same),
                "retrieval_top1_diff_color_accuracy": _metric_ratio(top1_hits_diff, total_diff),
                "probe_joint_same_color_accuracy": _metric_ratio(probe_joint_same_hits, probe_joint_same_total),
                "probe_joint_diff_color_accuracy": _metric_ratio(probe_joint_diff_hits, probe_joint_diff_total),
            }
        )

    return summary, cases


def parse_args() -> argparse.Namespace:
    """CLI grouped by: data inputs, object-embedding mode, intervention, eval, outputs."""
    parser = argparse.ArgumentParser(description="Interventions: CLEVR/CLEVR2D, PUG:SPARE, text")

    parser.add_argument("--dataset", choices=["clevr", "clevr2d", "pug_spare", "text"], required=True)
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset dir/file with dataset.pkl")
    parser.add_argument("--embedding-path", type=str, required=True, help="Scene embedding file (.pkl/.pt/.npy)")

    parser.add_argument(
        "--object-embedding-mode",
        choices=["single_object", "avg_scene_position_independent", "avg_scene_position_dependent"],
        default="avg_scene_position_independent",
    )
    parser.add_argument(
        "--single-object-embeddings-path",
        type=str,
        default=None,
        help=(
            "Required for --object-embedding-mode single_object. "
            "Expected format: pickle dict with keys (attr,obj,pos) and values as embedding vectors."
        ),
    )

    parser.add_argument("--steering-strength", type=float, default=1.0, help="Scale k in intervened = base - k * steering")
    parser.add_argument("--max-targets-per-pair", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=0, help="If >0, randomly sample this many scenes before interventions")
    parser.add_argument("--max-interventions", type=int, default=0, help="If >0, cap total generated interventions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-batch-size", type=int, default=512, help="Batch size for retrieval similarity computation")

    parser.add_argument("--probe", action="store_true", help="Use train_embeddings-style probes for probe accuracy")
    parser.add_argument("--probe-path", type=str, default=None, help="Optional path to train_embeddings-style probe artifact")
    parser.add_argument("--probe-epochs", type=int, default=200)

    parser.add_argument("--world-name", type=str, default="Desert", help="PUG_SPARE world_name filter")
    parser.add_argument("--require-character-pos-null", action="store_true", help="PUG_SPARE null character_pos filter")

    parser.add_argument(
        "--output-json",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Optional output summary JSON path. "
            "If provided without a value, a descriptive filename is auto-generated "
            "under <dataset-path>/interventions/."
        ),
    )
    parser.add_argument("--output-cases-jsonl", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run full intervention experiment from CLI args."""
    args = parse_args()

    scene_embeddings = _load_embedding_array(Path(args.embedding_path))
    loaded = _load_dataset_records(
        dataset_path=Path(args.dataset_path),
        embeddings=scene_embeddings,
        adapter=args.dataset,
        world_name=args.world_name,
        require_character_pos_null=bool(args.require_character_pos_null),
    )

    object_embedding_bank = _build_object_embedding_bank(
        records=loaded.records,
        scene_embeddings=loaded.embeddings,
        mode=args.object_embedding_mode,
        adapter=args.dataset,
        single_object_embeddings_path=args.single_object_embeddings_path,
        attr_to_idx=loaded.attr_to_idx,
        obj_to_idx=loaded.obj_to_idx,
    )

    probe_artifacts = _resolve_probe_artifacts(
        use_probe=bool(args.probe),
        probe_path=args.probe_path,
        embedding_path=Path(args.embedding_path),
        dataset_path=Path(args.dataset_path),
        probe_epochs=int(args.probe_epochs),
    )

    split_same_diff = args.dataset in {"clevr", "clevr2d", "text"}

    base_records = loaded.records
    base_embeddings = loaded.embeddings

    max_scenes = int(args.max_scenes)
    if max_scenes > 0 and len(loaded.records) > max_scenes:
        rng = random.Random(int(args.seed))
        chosen = sorted(rng.sample(range(len(loaded.records)), max_scenes))
        base_records = [loaded.records[i] for i in chosen]
        base_embeddings = loaded.embeddings[np.array(chosen, dtype=np.int64)]
        print({"note": "scene_sampling_applied", "max_scenes": max_scenes, "kept": len(chosen)})

    summary, cases = run_interventions(
        records=base_records,
        embeddings=base_embeddings,
        reference_records=loaded.records,
        reference_embeddings=loaded.embeddings,
        object_embedding_mode=args.object_embedding_mode,
        object_embedding_bank=object_embedding_bank,
        max_targets_per_pair=int(args.max_targets_per_pair),
        steering_strength=float(args.steering_strength),
        seed=int(args.seed),
        retrieval_batch_size=int(args.retrieval_batch_size),
        max_interventions=(None if int(args.max_interventions) <= 0 else int(args.max_interventions)),
        probe_artifacts=probe_artifacts,
        attr_concept_idx=loaded.attr_concept_idx,
        obj_concept_idx=loaded.obj_concept_idx,
        split_same_diff=split_same_diff,
    )

    payload = {
        "dataset": args.dataset,
        "dataset_path": str(args.dataset_path),
        "embedding_path": str(args.embedding_path),
        "object_embedding_mode": args.object_embedding_mode,
        "steering_strength": float(args.steering_strength),
        "probe_enabled": bool(args.probe),
        "probe_source": (None if probe_artifacts is None else probe_artifacts.source),
        "summary": summary,
    }

    print(json.dumps(payload, indent=2))

    if args.output_json is not None:
        out = _auto_output_json_path(args) if str(args.output_json) == "auto" else Path(str(args.output_json))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print({"note": "output_json_saved", "path": str(out)})

    if args.output_cases_jsonl:
        out = Path(args.output_cases_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(asdict(case)) + "\n")


if __name__ == "__main__":
    main()
