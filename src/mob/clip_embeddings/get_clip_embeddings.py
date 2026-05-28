#!/usr/bin/env python3
"""Simple unified script for CLIP text/image embeddings.

This file intentionally mirrors the logic and output naming from:
- 001_prepare_text_embeddings_clip.ipynb
- 002_prepare_vision_embeddings.ipynb (CLIP branch)
"""

import argparse
import itertools
import json
import os
import pickle
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_clip_model(model_name: str, device: str):
    import clip

    use_random = "-rand" in model_name
    model_name_use = model_name.replace("-rand", "").replace("clip-", "")
    model, preprocess = clip.load(model_name_use, device=device)

    if use_random:
        print("Initializing parameters")
        before = str(list(model.parameters())[0])
        model.initialize_parameters()
        after = str(list(model.parameters())[0])
        if before == after:
            raise RuntimeError("Parameters were not initialized")

    return model, preprocess


def _parse_vision_model_name(model_name: str):
    family = "clip"
    name = model_name

    if model_name.startswith("clip-"):
        family = "clip"
        name = model_name[len("clip-") :]
    elif model_name.startswith("dinov2-"):
        family = "dinov2"
        name = model_name[len("dinov2-") :]

    use_random = False
    if family == "clip" and name.endswith("-rand"):
        use_random = True
        name = name[: -len("-rand")]

    return family, name, use_random


def load_vision_model(model_name: str, device: str):
    family, name, use_random = _parse_vision_model_name(model_name)

    if family == "clip":
        import clip

        model, preprocess = clip.load(name, device=device)
        if use_random:
            print("Initializing parameters")
            before = str(list(model.parameters())[0])
            model.initialize_parameters()
            after = str(list(model.parameters())[0])
            if before == after:
                raise RuntimeError("Parameters were not initialized")
        print(f"Loaded vision model: family={family}, name={name}, device={device}")
        return model, preprocess

    if family == "dinov2":
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:
            raise ImportError(
                "DINOv2 support requires `transformers`. Install with: pip install transformers"
            ) from e

        aliases = {
            "vits14": "facebook/dinov2-small",
            "vitb14": "facebook/dinov2-base",
            "vitl14": "facebook/dinov2-large",
            "vitg14": "facebook/dinov2-giant",
        }
        hf_name = aliases.get(name, name)
        if "/" not in hf_name and not hf_name.startswith("facebook/dinov2-"):
            hf_name = f"facebook/{hf_name}"

        image_processor = AutoImageProcessor.from_pretrained(hf_name)
        backbone = AutoModel.from_pretrained(hf_name).to(device)
        backbone.eval()

        class _DinoV2Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
                out = self.model(pixel_values=pixel_values)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    return out.pooler_output
                return out.last_hidden_state[:, 0]

        def preprocess(pil_image):
            pv = image_processor(images=pil_image, return_tensors="pt")["pixel_values"]
            return pv[0]

        print(f"Loaded vision model: family={family}, name={hf_name}, device={device}")
        return _DinoV2Wrapper(backbone), preprocess

    raise ValueError(
        "Unknown vision model. Supported examples: "
        "clip-ViT-B/32, ViT-B/32, clip-ViT-B/32-rand, dinov2-vitb14, dinov2-vitl14, dinov2-facebook/dinov2-base"
    )


def get_text_concept_space():
    return [
        [
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "pink",
            "brown",
            "gray",
            "black",
            "small",
            "large",
            "tiny",
            "round",
            "sharp",
            "rough",
            "bumpy",
            "smooth",
            "matte",
            "glossy",
        ],
        [
            "circle",
            "square",
            "triangle",
            "star",
            "heart",
            "spiral",
            "sphere",
            "cylinder",
            "cone",
            "pyramid",
            "car",
            "cat",
            "dog",
            "man",
            "woman",
            "tree",
            "flower",
            "rock",
            "bird",
            "fish",
        ],
    ]


def generate_all_text_scenes(concept_space):
    obj_assignments = list(itertools.product(*[range(len(values)) for values in concept_space]))
    scenes = []
    for obj1 in obj_assignments:
        for obj2 in obj_assignments:
            scenes.append(
                [
                    list(obj1),
                    list(obj2),
                ]
            )
    return scenes


def normalize_scene_dataset_to_indices(scene_dataset, concept_space):
    value_to_index = [{value: idx for idx, value in enumerate(values)} for values in concept_space]
    normalized = []

    for scene in scene_dataset:
        normalized_scene = []
        for obj in scene:
            normalized_obj = []
            for concept_i, concept in enumerate(obj):
                if isinstance(concept, (int, np.integer)):
                    normalized_obj.append(int(concept))
                else:
                    concept_str = str(concept)
                    if concept_str not in value_to_index[concept_i]:
                        raise ValueError(
                            f"Unknown concept token '{concept_str}' for concept index {concept_i}."
                        )
                    normalized_obj.append(value_to_index[concept_i][concept_str])
            normalized_scene.append(normalized_obj)
        normalized.append(normalized_scene)

    return normalized


def build_text_captions(scene_dataset=None):
    concept_space = get_text_concept_space()

    if scene_dataset is None:
        scene_dataset = generate_all_text_scenes(concept_space)

    captions = []
    for scene in scene_dataset:
        scene_caption = "a"
        for obj_i, obj in enumerate(scene):
            for concept_i, concept in enumerate(obj):
                if isinstance(concept, (int, np.integer)):
                    concept_token = concept_space[concept_i][int(concept)]
                else:
                    concept_token = str(concept)
                scene_caption += f" {concept_token}"
            if obj_i != len(scene) - 1:
                scene_caption += " and a"
        captions.append(scene_caption)
    return captions


def generate_single_object_text_rows(concept_space):
    rows = []
    for attr_idx, attr in enumerate(concept_space[0]):
        for obj_idx, obj in enumerate(concept_space[1]):
            key_tuple = (attr_idx, obj_idx, "*")
            key_str = f"({attr_idx}, {obj_idx}, *)"
            caption = f"{attr} {obj}"
            rows.append(
                {
                    "key_tuple": key_tuple,
                    "key_str": key_str,
                    "attr_idx": attr_idx,
                    "obj_idx": obj_idx,
                    "attr": attr,
                    "obj": obj,
                    "pos": "*",
                    "caption": caption,
                }
            )
    return rows


def encode_text_captions(model, captions, batch_size: int, device: str, desc: str):
    import clip

    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size), desc=desc):
            batch_captions = captions[i : i + batch_size]
            text_tokens = clip.tokenize(batch_captions).to(device)
            text_features = model.encode_text(text_tokens)
            all_emb.append(text_features.cpu())

    return torch.cat(all_emb, dim=0) if all_emb else torch.empty(0)


def run_text_embeddings(dataset_path: Path | None, output_dir: Path, model_name: str, batch_size: int, device: str):
    vision_family, _, _ = _parse_vision_model_name(model_name)
    if vision_family != "clip":
        raise ValueError(
            f"Text mode currently supports CLIP only. Got model_name={model_name}. "
            "Use --mode image for DINOv2 backbones."
        )

    concept_space = get_text_concept_space()

    if dataset_path and dataset_path.exists():
        raw_scene_dataset = pickle.load(open(dataset_path, "rb"))
    else:
        raw_scene_dataset = generate_all_text_scenes(concept_space)

    scene_dataset = normalize_scene_dataset_to_indices(raw_scene_dataset, concept_space)

    captions_dataset = build_text_captions(scene_dataset)

    model, _ = load_clip_model(model_name=model_name, device=device)
    model.eval()

    caption_embeddings = encode_text_captions(
        model=model,
        captions=captions_dataset,
        batch_size=batch_size,
        device=device,
        desc="Computing text embeddings",
    )

    # Also export single-object text embeddings in run_interventions-compatible format:
    # dict[(attr_idx, obj_idx, "*")] -> embedding vector
    single_rows = generate_single_object_text_rows(concept_space)
    single_captions = [row["caption"] for row in single_rows]
    single_embeddings = encode_text_captions(
        model=model,
        captions=single_captions,
        batch_size=batch_size,
        device=device,
        desc="Computing single-object text embeddings",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    text_dataset_path = output_dir / "dataset.pkl"
    pickle.dump(scene_dataset, open(text_dataset_path, "wb"))

    out_path = output_dir / f"clip_{model_name.replace('/', '_')}_text_embeddings.pkl"
    pickle.dump(caption_embeddings, open(out_path, "wb"))

    single_object_out_path = output_dir / f"clip_{model_name.replace('/', '_')}_single_object_embeddings.pkl"
    key_to_embedding = {
        row["key_tuple"]: single_embeddings[i].numpy().astype(np.float32)
        for i, row in enumerate(single_rows)
    }
    with open(single_object_out_path, "wb") as f:
        pickle.dump(key_to_embedding, f)

    print(f"Saved text dataset: {text_dataset_path}")
    print(f"Saved text embeddings: {out_path}")
    print(f"Saved text single-object embeddings: {single_object_out_path}")
    print(f"Shape: {tuple(caption_embeddings.shape)}")


def prepare_image_dataset(dataset: str, repo_root: Path, world_name: str, output_root: Path):
    datasets = ["pug_spare", "clevr", "clevr2d"]
    if dataset not in datasets:
        raise ValueError(f"dataset {dataset} not in {datasets}")

    path_save = output_root / dataset
    path_save.mkdir(parents=True, exist_ok=True)
    dataset_scenes_save_path = path_save / "dataset.pkl"

    if dataset == "pug_spare":
        csv_path = repo_root / "src" / "mob" / "multi_obj_clip_analysis" / "pug_spare_dataset" / "PUG_SPARE.csv"
        df = pd.read_csv(csv_path)
        rows_to_process = df[(df["character_pos"].isna()) & (df["world_name"] == world_name)]
    else:
        labels_name = "CLEVR_posfix_labels.pkl" if dataset == "clevr" else "CLEVR2d_posfix_labels.pkl"
        labels_path = repo_root / "src" / "mob" / "multi_obj_clip_analysis" / "datasets" / labels_name
        raw_labels = pd.read_pickle(labels_path)

        clevr_colors = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
        clevr_shapes = ["cube", "cylinder", "sphere"]

        rows = []
        for i, scene in enumerate(raw_labels):
            obj1, obj2 = scene
            rows.append(
                {
                    "filename": f"{i:07d}.png",
                    "object1_color": clevr_colors[obj1[0]],
                    "object1_shape": clevr_shapes[obj1[1]],
                    "object2_color": clevr_colors[obj2[0]],
                    "object2_shape": clevr_shapes[obj2[1]],
                    "object1_color_idx": obj1[0],
                    "object1_shape_idx": obj1[1],
                    "object2_color_idx": obj2[0],
                    "object2_shape_idx": obj2[1],
                }
            )
        rows_to_process = pd.DataFrame(rows)

    rows_to_process.to_csv(path_save / "labels.csv", index=False)

    scenes = []
    if dataset == "pug_spare":
        unique_character_names = rows_to_process["character_name"].unique().tolist()
        unique_character_textures = rows_to_process["character_texture"].unique().tolist()

        for i in range(len(rows_to_process)):
            row = rows_to_process.iloc[i]
            scene = [
                [
                    unique_character_names.index(row["character_name"]),
                    unique_character_textures.index(row["character_texture"]),
                ],
                [
                    unique_character_names.index(row["character2_name"]),
                    unique_character_textures.index(row["character2_texture"]),
                ],
            ]
            scenes.append(scene)

        metadata = {
            "num_objects": 2,
            "num_concepts": 2,
            "values_per_concept": [len(unique_character_names), len(unique_character_textures)],
        }
    else:
        for i in range(len(rows_to_process)):
            row = rows_to_process.iloc[i]
            scenes.append(
                [
                    [row["object1_color_idx"], row["object1_shape_idx"]],
                    [row["object2_color_idx"], row["object2_shape_idx"]],
                ]
            )

        metadata = {
            "num_objects": 2,
            "num_concepts": 2,
            "values_per_concept": [8, 3],
        }

    with open(dataset_scenes_save_path, "wb") as f:
        pickle.dump(scenes, f)

    with open(path_save / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return path_save, rows_to_process


class ImagePathDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.preprocess = preprocess
        self.valid_paths = []
        self.failed_paths = []
        for p in paths:
            if os.path.isfile(p):
                self.valid_paths.append(p)
            else:
                self.failed_paths.append(p)
        if self.failed_paths:
            print(f"WARNING: {len(self.failed_paths)} images not found and will be skipped.")

    def __len__(self):
        return len(self.valid_paths)

    def __getitem__(self, idx):
        path = self.valid_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.preprocess(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None


class ImagePickleDataset(Dataset):
    def __init__(self, images, preprocess):
        self.images = images
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        return self.preprocess(img.convert("RGB"))


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.empty(0)
    return torch.stack(batch)


def _build_caption(row: pd.Series, dataset: str) -> str:
    if dataset == "pug_spare":
        return (
            f"a {row['character_texture']} {row['character_name']} and a "
            f"{row['character2_texture']} {row['character2_name']}"
        )
    if dataset == "clevr2d":
        shape_map = {
            "cube": "square",
            "cylinder": "triangle",
            "sphere": "circle",
        }
        obj1_shape = shape_map.get(str(row["object1_shape"]), str(row["object1_shape"]))
        obj2_shape = shape_map.get(str(row["object2_shape"]), str(row["object2_shape"]))
        return (
            f"a {row['object1_color']} {obj1_shape} and a "
            f"{row['object2_color']} {obj2_shape}"
        )
    return (
        f"a {row['object1_color']} {row['object1_shape']} and a "
        f"{row['object2_color']} {row['object2_shape']}"
    )


def _save_paper_grid(dataset: str, rows_to_process: pd.DataFrame, out_dir: Path, n_rows: int = 2, n_cols: int = 4):
    n_grid = min(n_rows * n_cols, len(rows_to_process))
    if n_grid == 0:
        return

    # Pick saved sanity images from out_dir to keep this lightweight.
    entries = []
    for i in range(n_grid):
        row = rows_to_process.iloc[i]
        file_name = f"{i:05d}_{row['filename']}"
        img_path = out_dir / file_name
        if not img_path.exists():
            continue
        entries.append((Image.open(img_path).convert("RGB"), _build_caption(row=row, dataset=dataset)))

    if not entries:
        return

    tile_w, tile_h = entries[0][0].size
    img_aspect = float(tile_h) / float(tile_w)

    fig_width = 6.5
    cell_width = fig_width / n_cols
    # Add extra vertical space per row for wrapped caption text.
    cell_height = (img_aspect * cell_width) + 0.45
    fig_height = max(3.2, n_rows * cell_height)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), constrained_layout=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx in range(n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        ax.axis("off")

        if idx >= len(entries):
            continue

        img, caption = entries[idx]
        ax.imshow(np.array(img))
        ax.set_title("\n".join(textwrap.wrap(caption, width=22)), fontsize=7, pad=3)

    out_path = out_dir / "paper_grid_2x4.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def dump_sanity_images(dataset: str, rows_to_process: pd.DataFrame, out_dir: Path, repo_root: Path, world_name: str, n: int = 12):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n, len(rows_to_process))

    if dataset == "pug_spare":
        base_img_path = repo_root / "src" / "mob" / "multi_obj_clip_analysis" / "pug_spare_dataset" / world_name
        for i in range(n):
            row = rows_to_process.iloc[i]
            src = base_img_path / row["filename"]
            if src.exists():
                img = Image.open(src).convert("RGB")
                img.save(out_dir / f"{i:05d}_{row['filename']}")
    else:
        images_name = "CLEVR_posfix_images.pkl" if dataset == "clevr" else "CLEVR2d_posfix_images.pkl"
        images_pkl_path = repo_root / "src" / "mob" / "multi_obj_clip_analysis" / "datasets" / images_name
        with open(images_pkl_path, "rb") as f:
            all_images = pickle.load(f)
        for i in range(n):
            row = rows_to_process.iloc[i]
            img = all_images[i]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img.convert("RGB").save(out_dir / f"{i:05d}_{row['filename']}")

    rows_to_process.head(n).to_csv(out_dir / "sample_labels.csv", index=False)

    paper_rows = []
    n_paper = min(2, n)
    for i in range(n_paper):
        row = rows_to_process.iloc[i]
        paper_rows.append(
            {
                "dataset": dataset,
                "sample_index": i,
                "filename": row["filename"],
                "caption": _build_caption(row=row, dataset=dataset),
            }
        )
    pd.DataFrame(paper_rows).to_csv(out_dir / "paper_samples.csv", index=False)
    _save_paper_grid(dataset=dataset, rows_to_process=rows_to_process, out_dir=out_dir, n_rows=2, n_cols=4)
    print(f"Saved sanity check files to: {out_dir}")


def run_image_embeddings(
    dataset: str,
    model_name: str,
    batch_size: int,
    device: str,
    world_name: str,
    repo_root: Path,
    output_root: Path,
):
    path_save, rows_to_process = prepare_image_dataset(
        dataset=dataset, repo_root=repo_root, world_name=world_name, output_root=output_root
    )

    model, preprocess = load_vision_model(model_name=model_name, device=device)
    model.eval()
    vision_family, _, _ = _parse_vision_model_name(model_name)

    if dataset == "pug_spare":
        base_img_path = repo_root / "src" / "mob" / "multi_obj_clip_analysis" / "pug_spare_dataset" / world_name
        all_image_paths = [str(base_img_path / fname) for fname in rows_to_process["filename"]]
        image_dataset = ImagePathDataset(all_image_paths, preprocess)
        num_workers = 8 if vision_family == "clip" else 0
    else:
        images_name = "CLEVR_posfix_images.pkl" if dataset == "clevr" else "CLEVR2d_posfix_images.pkl"
        images_pkl_path = repo_root / "src" / "mob" / "multi_obj_clip_analysis" / "datasets" / images_name
        with open(images_pkl_path, "rb") as f:
            all_images = pickle.load(f)
        image_dataset = ImagePickleDataset(all_images, preprocess)
        num_workers = 0

    loader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    all_embeddings = []
    with torch.no_grad():
        for images_tensor in tqdm(loader, desc="Computing image embeddings"):
            if images_tensor.shape[0] == 0:
                continue
            images_tensor = images_tensor.to(device)
            features = model.encode_image(images_tensor)
            all_embeddings.append(features.cpu())

    embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)

    if dataset == "pug_spare":
        save_path = path_save / f"{world_name}_{model_name.replace('/', '_')}_embeddings.pkl"
    else:
        save_path = path_save / f"{dataset}_{model_name.replace('/', '_')}_embeddings.pkl"

    pickle.dump(embeddings, open(save_path, "wb"))
    print(f"Saved image embeddings: {save_path}")
    print(f"Shape: {tuple(embeddings.shape)}")

    sanity_dir = path_save / "sanity_checks" / f"{dataset}_{model_name.replace('/', '_')}"
    dump_sanity_images(dataset=dataset, rows_to_process=rows_to_process, out_dir=sanity_dir, repo_root=repo_root, world_name=world_name)


def run_single_object_image_embeddings(
    single_object_images_path: Path,
    output_path: Path,
    dataset: str,
    model_name: str,
    batch_size: int,
    device: str,
):
    """Compute CLIP image embeddings for a pickle dict {(attr,obj,pos): image}.

    Output is a pickle dict with identical keys and values replaced by embedding vectors.
    """
    with open(single_object_images_path, "rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, dict):
        raise ValueError("single_object_images_path must point to a pickle dict of key -> image")

    def _is_int_like(x) -> bool:
        if isinstance(x, (int, np.integer)):
            return True
        if isinstance(x, str):
            return x.strip().lstrip("-").isdigit()
        return False

    def _to_int(x) -> int:
        if isinstance(x, (int, np.integer)):
            return int(x)
        return int(str(x).strip())

    def _normalize_single_object_key(key):
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError(f"single_object key must be a 3-tuple (attr,obj,pos), got: {key}")

        raw_attr, raw_obj, raw_pos = key
        pos = str(raw_pos).strip().lower()

        if _is_int_like(raw_attr) and _is_int_like(raw_obj):
            return (_to_int(raw_attr), _to_int(raw_obj), pos)

        if dataset == "clevr":
            color_to_idx = {name: i for i, name in enumerate(["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"])}
            shape_to_idx = {name: i for i, name in enumerate(["cube", "cylinder", "sphere"])}
            attr = str(raw_attr).strip().lower()
            obj = str(raw_obj).strip().lower()
            if attr not in color_to_idx or obj not in shape_to_idx:
                raise ValueError(f"Unknown CLEVR single_object key token: {(raw_attr, raw_obj, raw_pos)}")
            return (color_to_idx[attr], shape_to_idx[obj], pos)

        if dataset == "clevr2d":
            color_to_idx = {name: i for i, name in enumerate(["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"])}
            shape_to_idx = {name: i for i, name in enumerate(["square", "triangle", "circle"])}
            attr = str(raw_attr).strip().lower()
            obj = str(raw_obj).strip().lower()
            if attr not in color_to_idx or obj not in shape_to_idx:
                raise ValueError(f"Unknown CLEVR2D single_object key token: {(raw_attr, raw_obj, raw_pos)}")
            return (color_to_idx[attr], shape_to_idx[obj], pos)

        # For datasets without a fixed global concept dictionary in this script (e.g., pug_spare),
        # require numeric keys to avoid ambiguous mapping.
        raise ValueError(
            f"single_object key {(raw_attr, raw_obj, raw_pos)} is not index-based and cannot be auto-mapped for dataset={dataset}"
        )

    source_keys = list(raw.keys())
    keys = [_normalize_single_object_key(k) for k in source_keys]
    images = [raw[k] for k in source_keys]
    if len(images) == 0:
        raise ValueError("single-object image dictionary is empty")

    model, preprocess = load_vision_model(model_name=model_name, device=device)
    model.eval()

    image_dataset = ImagePickleDataset(images, preprocess)
    loader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    all_embeddings = []
    with torch.no_grad():
        for images_tensor in tqdm(loader, desc="Computing single-object image embeddings"):
            if images_tensor.shape[0] == 0:
                continue
            images_tensor = images_tensor.to(device)
            features = model.encode_image(images_tensor)
            all_embeddings.append(features.cpu())

    embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)
    if embeddings.shape[0] != len(keys):
        raise RuntimeError(
            f"Mismatch between keys ({len(keys)}) and embeddings ({embeddings.shape[0]})."
        )

    key_to_embedding = {
        keys[i]: embeddings[i].numpy().astype(np.float32)
        for i in range(len(keys))
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(key_to_embedding, f)

    print(f"Saved single-object image embeddings (index-keyed): {output_path}")
    print(f"Num entries: {len(key_to_embedding)}")
    print(f"Embedding dim: {int(embeddings.shape[1]) if embeddings.ndim == 2 else 0}")


def main():
    parser = argparse.ArgumentParser(description="Unified CLIP text or image embeddings.")
    parser.add_argument("--mode", choices=["text", "image"], required=True)

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Optional path to dataset.pkl for text mode. If omitted, captions are generated from the handcrafted concept space.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32-rand",
        help=(
            "Vision backbone name. Examples: clip-ViT-B/32, clip-ViT-B/32-rand, "
            "ViT-B/32, dinov2-vitb14, dinov2-vitl14, dinov2-facebook/dinov2-base"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--dataset", choices=["pug_spare", "clevr", "clevr2d"], default="clevr")
    parser.add_argument("--world_name", type=str, default="Desert")
    parser.add_argument(
        "--repo_root",
        type=str,
        default="",
        help="Optional repo root for src/ and data/ paths. Defaults to this checkout root.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Image mode only: output root for <dataset>/dataset.pkl + embeddings. Defaults to <repo_root>/data/clip_checks.",
    )
    parser.add_argument(
        "--single_object_images_path",
        type=str,
        default="",
        help=(
            "Optional image-mode input: pickle dict with format {(attr,obj,pos): image}. "
            "If provided, CLIP embeddings are computed and saved as a dict with the same keys."
        ),
    )
    parser.add_argument(
        "--single_object_output_path",
        type=str,
        default="",
        help=(
            "Optional output path for --single_object_images_path embeddings. "
            "Defaults to <input_stem>_<model>_embeddings.pkl next to the input file."
        ),
    )

    args = parser.parse_args()

    if args.mode == "text":
        dataset_path = Path(args.dataset_path) if args.dataset_path else None
        if dataset_path:
            text_output_dir = dataset_path.parent
        else:
            repo_root = Path(args.repo_root) if args.repo_root else default_repo_root()
            text_output_dir = repo_root / "data" / "clip_checks" / "text"
        run_text_embeddings(
            dataset_path=dataset_path,
            output_dir=text_output_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        if args.single_object_images_path:
            input_path = Path(args.single_object_images_path)
            if args.single_object_output_path:
                out_path = Path(args.single_object_output_path)
            else:
                model_tag = args.model_name.replace("/", "_")
                out_path = input_path.with_name(f"{input_path.stem}_{model_tag}_embeddings.pkl")

            run_single_object_image_embeddings(
                single_object_images_path=input_path,
                output_path=out_path,
                dataset=args.dataset,
                model_name=args.model_name,
                batch_size=args.batch_size,
                device=args.device,
            )
        else:
            repo_root = Path(args.repo_root) if args.repo_root else default_repo_root()
            output_root = Path(args.output_root) if args.output_root else (repo_root / "data" / "clip_checks")
            output_root.mkdir(parents=True, exist_ok=True)
            run_image_embeddings(
                dataset=args.dataset,
                model_name=args.model_name,
                batch_size=args.batch_size,
                device=args.device,
                world_name=args.world_name,
                repo_root=repo_root,
                output_root=output_root,
            )


if __name__ == "__main__":
    main()
