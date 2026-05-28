#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DATASET_LABEL = {
    "text": "Text",
    "clevr": "CLEVR",
    "clevr2d": "CLEVR-2D",
    "pug_spare": "PUG:SPARE",
}

DATASET_ORDER = ["text", "clevr", "clevr2d", "pug_spare"]
CONCEPT_ORDER = ["Different", "Shared"]

MODE_MAP = {
    "avg_scene_position_independent": "obj-avg",
    "avg_scene_position_dependent": "pos-avg",
    "single_object": "single",
}

MODE_ORDER = ["obj-avg", "pos-avg", "single"]


def _find_json_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/interventions/*.json"))


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _infer_model_label(embedding_path: str) -> str:
    stem = Path(embedding_path).stem.lower()

    if "clip-vit-b_32" in stem:
        return "CLIP ViT-B/32"
    if "clip-vit-l_14" in stem:
        return "CLIP ViT-L/14"
    if "dinov2-vitb14" in stem or "dinov2-base" in stem:
        return "DINO ViT-B/14"
    if "dinov2-vitl14" in stem or "dinov2-large" in stem:
        return "DINO ViT-L/14"

    model_token = stem
    if "_embeddings" in model_token:
        model_token = model_token.split("_embeddings", 1)[0]
    model_token = model_token.replace("_", "-")
    return model_token


def _model_sort_key(model_label: str) -> Tuple[int, str]:
    order = {
        "CLIP ViT-B/32": 0,
        "CLIP ViT-L/14": 1,
        "DINO ViT-B/14": 2,
        "DINO ViT-L/14": 3,
    }
    return (order.get(model_label, 999), model_label)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"\\textit{{{value:.2f}}}"


def _extract_rows(payload: dict) -> List[Tuple[str, str, str, str, Optional[float], Optional[float]]]:
    dataset = str(payload.get("dataset", ""))
    embedding_path = str(payload.get("embedding_path", ""))
    mode_raw = str(payload.get("object_embedding_mode", ""))
    summary = payload.get("summary", {}) or {}

    if dataset not in DATASET_LABEL or mode_raw not in MODE_MAP:
        return []

    mode = MODE_MAP[mode_raw]
    model = _infer_model_label(embedding_path)

    out: List[Tuple[str, str, str, str, Optional[float], Optional[float]]] = []

    if dataset in {"text", "clevr", "clevr2d"}:
        out.append(
            (
                dataset,
                model,
                "Different",
                mode,
                summary.get("probe_joint_diff_color_accuracy"),
                summary.get("retrieval_top1_diff_color_accuracy"),
            )
        )
        out.append(
            (
                dataset,
                model,
                "Shared",
                mode,
                summary.get("probe_joint_same_color_accuracy"),
                summary.get("retrieval_top1_same_color_accuracy"),
            )
        )
    else:
        out.append(
            (
                dataset,
                model,
                "Different",
                mode,
                summary.get("probe_joint_accuracy"),
                summary.get("retrieval_top1_accuracy"),
            )
        )

    return out


def build_latex_table(rows: Dict[Tuple[str, str, str, str], Tuple[Optional[float], Optional[float]]], k_value: float) -> str:
    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append(
        "\\caption{Intervention results summary at $k="
        + f"{k_value:.1f}"
        + "$. Object embeddings are constructed by object averaging (obj-avg), position-wise averaging (pos-avg), or from single-object scenes (single).}"
    )
    lines.append("\\begin{tabular}{lllcccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & Concepts")
    lines.append("& \\multicolumn{3}{c}{Probing ($\\uparrow$)}")
    lines.append("& \\multicolumn{3}{c}{Retrieval ($\\uparrow$)} " + "\\\\")
    lines.append("\\cmidrule(lr){4-6} \\cmidrule(lr){7-9}")
    lines.append("& &")
    lines.append("& obj-avg & pos-avg & single")
    lines.append("& obj-avg & pos-avg & single " + "\\\\")
    lines.append("\\midrule")

    rendered_any = False

    for dataset in DATASET_ORDER:
        dataset_label = DATASET_LABEL[dataset]

        dataset_models = sorted(
            {k[1] for k in rows.keys() if k[0] == dataset},
            key=_model_sort_key,
        )
        if not dataset_models:
            continue

        if rendered_any:
            lines.append("\\midrule")

        for model in dataset_models:
            concepts = ["Different"] if dataset == "pug_spare" else CONCEPT_ORDER
            for concept in concepts:
                probe_vals = []
                retrieval_vals = []
                for mode in MODE_ORDER:
                    metrics = rows.get((dataset, model, concept, mode))
                    if metrics is None:
                        probe_vals.append(None)
                        retrieval_vals.append(None)
                    else:
                        probe_vals.append(metrics[0])
                        retrieval_vals.append(metrics[1])

                line = (
                    f"{dataset_label} & {model} & {concept}"
                    f" & {_fmt(probe_vals[0])} & {_fmt(probe_vals[1])} & {_fmt(probe_vals[2])}"
                    f" & {_fmt(retrieval_vals[0])} & {_fmt(retrieval_vals[1])} & {_fmt(retrieval_vals[2])} \\\\" 
                )
                lines.append(line)
                rendered_any = True

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\label{tab:intervention_summary_k1}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LaTeX intervention summary table from JSON outputs.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/clip_checks",
        help="Root directory containing dataset folders with interventions/*.json",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=1.0,
        help="Only include runs with this steering_strength.",
    )
    parser.add_argument(
        "--out-tex",
        type=str,
        default="data/clip_checks/intervention_summary_k1.tex",
        help="Output LaTeX table path.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    files = _find_json_files(root)

    dedup: Dict[Tuple[str, str, str, str], Tuple[Optional[float], Optional[float], float]] = {}

    for path in files:
        payload = _read_json(path)
        if payload is None:
            continue

        k_val = float(payload.get("steering_strength", -1.0))
        if abs(k_val - float(args.k)) > 1e-9:
            continue

        extracted = _extract_rows(payload)
        if not extracted:
            continue

        mtime = path.stat().st_mtime
        for dataset, model, concept, mode, probe, retrieval in extracted:
            key = (dataset, model, concept, mode)
            old = dedup.get(key)
            if old is None or mtime >= old[2]:
                dedup[key] = (probe, retrieval, mtime)

    rows = {k: (v[0], v[1]) for k, v in dedup.items()}

    latex = build_latex_table(rows=rows, k_value=float(args.k))
    out_path = Path(args.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex, encoding="utf-8")

    print({
        "json_files_scanned": len(files),
        "rows_filled": len(rows),
        "k": float(args.k),
        "out_tex": str(out_path),
    })


if __name__ == "__main__":
    main()
