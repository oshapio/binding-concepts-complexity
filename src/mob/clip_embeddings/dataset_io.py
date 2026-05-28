from pathlib import Path
import json
import pickle
from typing import Any, Optional, Tuple


__all__ = ["load_dataset"]


def load_dataset(path: str | Path) -> Tuple[Any, Optional[dict]]:
    """Load dataset.pkl and optional metadata.json.

    - If `path` is a directory, reads `<path>/dataset.pkl` and `<path>/metadata.json`.
    - If `path` is a file, treats it as the pickle and looks for a sibling metadata.json.
    Returns (dataset, metadata_dict_or_None).
    """
    p = Path(path)
    if p.is_dir():
        pkl_path = p / "dataset.pkl"
        meta_path = p / "metadata.json"
    else:
        pkl_path = p
        meta_path = p.with_name("metadata.json") if p.name == "dataset.pkl" else None

    if not pkl_path.exists():
        raise FileNotFoundError(f"dataset.pkl not found at {pkl_path}")

    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    meta = None
    if meta_path is not None and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    return data, meta
