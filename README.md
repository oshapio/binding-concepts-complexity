# How can embedding models bind concepts?

[![arXiv](https://img.shields.io/badge/arXiv-2605.31503-b31b1b.svg)](https://arxiv.org/abs/2605.31503)

Code accompanying the paper [*How can embedding models bind concepts?*](https://arxiv.org/abs/2605.31503)

---

## Table of contents

1. [Setup](#setup)
2. [Data](#2-data)
3. [Extracting CLIP embeddings](#3-extracting-clip-embeddings)
4. [Probe suite](#4-probe-suite)
5. [Interventions](#5-interventions)
6. [Scene complexity analysis](#6-scene-complexity-analysis)
7. [Amortized training](#7-amortized-training)

---

## Setup

**Requirements:** Python 3.10, [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

`uv` will create a virtualenv and install all dependencies (including CLIP from source).

### Paths

The scripts below assume the following variables are set:

```bash
CODE_REPO=/mnt/lustre/work/oh/owl661/mobpub/mob_project
OUTPUT_ROOT=$CODE_REPO/data/clip_checks_public

# Synthetic 20x20 text dataset
TEXT_DATASET_20X20=/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/objs2_concepts2_values20_nodedup_max1000000000_actual160400_mixed_20260104-111651_False/text_dataset.pkl

# Raw image sources (only needed for re-extraction)
RAW_DATA_ROOT=$CODE_REPO/src/mob/multi_obj_clip_analysis

mkdir -p $OUTPUT_ROOT/{clevr,clevr2d,pug_spare}
```

---

## 2. Data

Raw image inputs and the synthetic text dataset are available on Dropbox:

> https://www.dropbox.com/scl/fo/mzzc432bf3orozmp3e1pa/ABNZER3kVyE9ZnnrQP_GQHU?rlkey=le4ab3dh6g0s72x25n0mpli8s&dl=0

The pipeline expects you to run `get_clip_embeddings.py` (section 3).
This script generates all derived files under `$OUTPUT_ROOT`, including
`dataset.pkl`, `metadata.json`, `labels.csv`, and embedding `.pkl` files.

---

## 3. Extracting CLIP embeddings

### 3a. Image embeddings

```bash
for DS in clevr clevr2d pug_spare; do
  python "$CODE_REPO/src/mob/clip_embeddings/get_clip_embeddings.py" \
    --mode image \
    --dataset "$DS" \
    --model_name clip-ViT-B/32 \
    --raw_data_root "$RAW_DATA_ROOT" \
    --output_root "$OUTPUT_ROOT"
done
```

This produces a unified export layout:
- `$OUTPUT_ROOT/clevr/{dataset.pkl,metadata.json,labels.csv,clevr_clip-ViT-B_32_embeddings.pkl}`
- `$OUTPUT_ROOT/clevr2d/{dataset.pkl,metadata.json,labels.csv,clevr2d_clip-ViT-B_32_embeddings.pkl}`
- `$OUTPUT_ROOT/pug_spare/{dataset.pkl,metadata.json,labels.csv,Desert_clip-ViT-B_32_embeddings.pkl}`

Raw inputs expected at:
- `$RAW_DATA_ROOT/datasets/`: CLEVR/CLEVR2D (`*_labels.pkl`, `*_images.pkl`)
- `$RAW_DATA_ROOT/pug_spare_dataset/`: PUG_SPARE (`PUG_SPARE.csv` + world folders)

### 3b. Text embeddings

Operates on the synthetic 20×20 text dataset independently of the image datasets.

```bash
python "$CODE_REPO/src/mob/clip_embeddings/get_clip_embeddings.py" \
  --mode text \
  --dataset_path "$TEXT_DATASET_20X20" \
  --model_name clip-ViT-B/32
```

Output: `$(dirname "$TEXT_DATASET_20X20")/clip_clip-ViT-B_32_text_embeddings.pkl`

---

## 4. Probe suite

Linear probes on object identities and concepts, fitted on frozen CLIP features with a controlled scene-level train/test split. Also includes four **subtraction variants** with estimated concept/object contributions removed.

**Script:** `src/mob/clip_embeddings/run_probe_suite.py`

```bash
# Derived dataset paths (produced in section 3a)
DATASET_CLEVR=$OUTPUT_ROOT/clevr/dataset.pkl
DATASET_CLEVR2D=$OUTPUT_ROOT/clevr2d/dataset.pkl
DATASET_PUG_SPARE=$OUTPUT_ROOT/pug_spare/dataset.pkl

for DS_PATH in "$DATASET_CLEVR" "$DATASET_CLEVR2D" "$DATASET_PUG_SPARE"; do
  DS_NAME=$(basename "$(dirname "$DS_PATH")")
  if [[ "$DS_NAME" == "pug_spare" ]]; then
    EMB_PATH="$OUTPUT_ROOT/$DS_NAME/Desert_clip-ViT-B_32_embeddings.pkl"
  else
    EMB_PATH="$OUTPUT_ROOT/$DS_NAME/${DS_NAME}_clip-ViT-B_32_embeddings.pkl"
  fi

  python "$CODE_REPO/src/mob/clip_embeddings/run_probe_suite.py" \
    --dataset-path "$DS_PATH" \
    --embedding-path "$EMB_PATH"
done
```

`--embedding-path` accepts either:
- a base embedding `.pkl` file, or
- a directory containing exactly one `*_embeddings.pkl` file.

Output: `<embedding_dir>/probe_suite_<timestamp>/results.json`. Each entry includes `summary_metrics` (train/test accuracies per pack) and `dataset_metadata`.

The train/test split is scene-level with a fixed seed. Trained embeddings are saved as `trained_embeddings_dim512_simdot_fit_conceptsTrue_fit_objectsTrue_<train-ratio>__<embedding_name>.pt`.

---

## 5. Interventions

Interventions evaluate whether a steered scene embedding retrieves the intended control scene and preserves concept-object structure under probes.

**Script:** `src/mob/clip_embeddings/run_interventions.py`

### 5a. What to prepare beforehand

Before running interventions, make sure you have:

1. **Dataset folder with `dataset.pkl`** under `$OUTPUT_ROOT/<dataset>/`
2. **Scene embeddings** for the same dataset/model
3. **Probe artifact** (`trained_embeddings_*.pt`) for that embedding file *(optional)*
4. **Single-object embeddings** if using `--object-embedding-mode single_object`

If you pass `--probe` without `--probe-path`, `run_interventions.py` will try to resolve probes from the embedding artifact and, if missing, train probes internally via `train_embeddings.py`.

Single-object embeddings can be produced with `get_clip_embeddings.py` from a single-object image dictionary:

```bash
python "$CODE_REPO/src/mob/clip_embeddings/get_clip_embeddings.py" \
  --mode image \
  --dataset clevr \
  --model_name dinov2-vitb14 \
  --single_object_images_path "$OUTPUT_ROOT/clevr/CLEVR_posfix_images_single.pkl"
```


### 5b. Run interventions (single dataset/model)

Example for CLEVR + DINO ViT-B/14 + single-object bank:

```bash
python "$CODE_REPO/src/mob/clip_embeddings/run_interventions.py" \
  --dataset clevr \
  --dataset-path "$OUTPUT_ROOT/clevr" \
  --embedding-path "$OUTPUT_ROOT/clevr/clevr_dinov2-vitb14_embeddings.pkl" \
  --probe \
  --probe-path "$OUTPUT_ROOT/clevr/embeddings/trained_embeddings_dim768_simdot_fit_conceptsTrue_fit_objectsTrue_0.4__clevr_dinov2-vitb14_embeddings.pkl.pt" \
  --object-embedding-mode single_object \
  --single-object-embeddings-path "$OUTPUT_ROOT/clevr/CLEVR_posfix_images_single_dinov2-vitb14_embeddings.pkl" \
  --output-json
```

`--output-json` without a value auto-generates a descriptive file under:
`<dataset-path>/interventions/`.

Supported object embedding modes:
- `avg_scene_position_independent` (obj-avg)
- `avg_scene_position_dependent` (pos-avg)
- `single_object` (requires `--single-object-embeddings-path`; not supported for PUG:SPARE).
---

## 6. Scene complexity analysis

Probe accuracy as a function of training set size, under an **object-level** split: train on scenes where all objects are in the train set, test on scenes with entirely unseen objects. Stricter than the scene-level split in section 4.

**Script:** `src/mob/clip_embeddings/approximate_complexity_scenes.py`
**Launcher:** `src/mob/clip_embeddings/run_analyses.sh`

The launcher sweeps training fractions (0.1–0.9) across learning rates.

Available profiles:

| Profile | Description |
|---------|-------------|
| `regular_mlp` | Non-multiplicative concat MLP + hidden-width sweep |
| `mult_linear` | Multiplicative probe + linear head |
| `sum_mult_linear` | Multiplicative + sum probe + linear head |

```bash
EMB_CLEVR="$OUTPUT_ROOT/clevr/embeddings/trained_embeddings_dim512_simdot_fit_conceptsTrue_fit_objectsTrue_0.4__clevr_clip-ViT-B_32_embeddings.pkl.pt"
EMB_CLEVR2D="$OUTPUT_ROOT/clevr2d/embeddings/trained_embeddings_dim512_simdot_fit_conceptsTrue_fit_objectsTrue_0.4__clevr2d_clip-ViT-B_32_embeddings.pkl.pt"
EMB_PUG_SPARE="$OUTPUT_ROOT/pug_spare/embeddings/trained_embeddings_dim512_simdot_fit_conceptsTrue_fit_objectsTrue_0.4__Desert_clip-ViT-B_32_embeddings.pkl.pt"

# 1) regular MLP
for EMB in "$EMB_CLEVR" "$EMB_CLEVR2D" "$EMB_PUG_SPARE"; do
  "$CODE_REPO/src/mob/clip_embeddings/run_analyses.sh" \
    --lrs 1e-2 \
    --profiles regular_mlp \
    --regular-hidden-specs 1024 \
    --embeddings "$EMB"
done

# 2) multiplicative linear (no hidden MLP)
for EMB in "$EMB_CLEVR" "$EMB_CLEVR2D" "$EMB_PUG_SPARE"; do
  "$CODE_REPO/src/mob/clip_embeddings/run_analyses.sh" \
    --lrs 1e-2 \
    --profiles mult_linear \
    --embeddings "$EMB"
done

# 3) sum + multiplicative linear (no hidden MLP)
for EMB in "$EMB_CLEVR" "$EMB_CLEVR2D" "$EMB_PUG_SPARE"; do
  "$CODE_REPO/src/mob/clip_embeddings/run_analyses.sh" \
    --lrs 1e-2 \
    --profiles sum_mult_linear \
    --embeddings "$EMB"
done
```

---

## 7. Amortized training

Small transformer that decomposes scene embeddings into per-object embeddings. Uses an object-level generalization split via `--train-object-fraction`.

**Scripts:** `src/mob/clip_embeddings/amortization/{train_clean.py,models.py,scenes.py}`

```bash
mkdir -p "$OUTPUT_ROOT/amortized"

python "$CODE_REPO/src/mob/clip_embeddings/amortization/train_clean.py" \
  --epochs 50000 \
  --model-d-out 64 \
  --model-d-model 256 \
  --model-num-heads 4 \
  --model-num-layers 6 \
  --model-lr 3e-5 \
  --sim-type cos \
  --train-batch-size 512 \
  --probe-concepts true \
  --probe-objects true \
  --max-num-objects 2 \
  --num-concepts 2 \
  --num-vals-per-concept 20 \
  --test-every-steps 200 \
  --test-num-batches 10 \
  --train-object-fraction 0.8 \
  --use-wandb false \
  --save-model true \
  --working-dir "$OUTPUT_ROOT/amortized"
```

Convert amortization checkpoints (`model_best_test_objects.pt`) to complexity-analysis inputs:

```bash
OUT_ROOT_AMORT="$OUTPUT_ROOT/amortization"

AMORT_CKPT_1="/mnt/lustre/work/oh/owl661/mob_project/experiments/configs/2026-02-28_14:08:08.146297---f4734b959a3416e52933---2235050/models/model_best_test_objects.pt"

for CKPT in "$AMORT_CKPT_1"; do
  RUN_NAME=$(basename "$(dirname "$(dirname "$CKPT")")" | tr '-' '_')
  python "$CODE_REPO/src/mob/clip_embeddings/extract_dataset_and_embeddings_from_pretrained_clike.py" \
    --model-path "$CKPT" \
    --output-root "$OUT_ROOT_AMORT" \
    --output-name "$RUN_NAME" \
    --objects-source all \
    --max-objs 400 \
    --batch-size 1024
done
```

This writes, for each converted checkpoint:
- `$OUT_ROOT_AMORT/<run_name>/dataset.pkl`
- `$OUT_ROOT_AMORT/<run_name>/metadata.json`
- `$OUT_ROOT_AMORT/<run_name>/scene_embeddings.pkl`

Run complexity on a converted amortization embedding:

```bash
EMB_AMORT="/mnt/lustre/work/oh/owl661/mobpub/mob_project/data/clip_checks_public/amortization/2026_02_28_14:08:08.146297___f4734b959a3416e52933___2235050/scene_embeddings.pkl"

"$CODE_REPO/src/mob/clip_embeddings/run_analyses.sh" \
  --lrs 1e-2 \
  --profiles regular_mlp \
  --regular-hidden-specs 1024 \
  --embeddings "$EMB_AMORT"

"$CODE_REPO/src/mob/clip_embeddings/run_analyses.sh" \
  --lrs 1e-2 \
  --profiles mult_linear \
  --embeddings "$EMB_AMORT"

"$CODE_REPO/src/mob/clip_embeddings/run_analyses.sh" \
  --lrs 1e-2 \
  --profiles sum_mult_linear \
  --embeddings "$EMB_AMORT"
```
