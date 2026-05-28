from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dataset_io import load_dataset
from itertools import combinations
import numpy as np
import torch.nn.functional as F
import wandb
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
import os
# wandb_enabled = False
# wandb.mode = "offline"


# --------------------
# Precompute utilities
# --------------------

def group_scene_indices_by_count(data):
    """Return a dict k -> list of scene indices that have exactly k objects.

    This lets us build one dataset per distinct object count k.

    Example:
        data = [
            [[1, 2], [3, 4]],             # scene 0 has k=2 objects
            [[5, 6]],                      # scene 1 has k=1 object
            [[7, 8], [9, 0], [1, 1]],      # scene 2 has k=3 objects
        ]
        result = group_scene_indices_by_count(data)
        # result == {1: [1], 2: [0], 3: [2]}
    """
    by_k = {}
    for scene_idx, scene in enumerate(data):
        k = len(scene)
        by_k.setdefault(k, []).append(scene_idx)
    return by_k


def build_object_id_map(data):
    """Build a stable global mapping from object attribute tuples -> integer ids.

    - object_to_id: maps tuple(attrs) to an integer id
    - id_to_object: reverse list where id_to_object[id] = tuple(attrs)
    """
    object_to_id, id_to_object = {}, []
    for scene in data:
        for obj in scene:
            key = tuple(int(v) for v in obj)
            if key not in object_to_id:
                object_to_id[key] = len(id_to_object)
                id_to_object.append(key)
    return object_to_id, id_to_object


def combination_positions_by_order(k, max_order):
    """Precompute index combinations for each order r = 1..min(k, max_order).

    - For a scene with k objects, this returns combinations of object indices
      without replacement, so each list size equals C(k, r).
    - Combinations are order-invariant: (0, 2) is the same as (2, 0), and only
      the canonical (0, 2) appears.

    Examples:
        - k=3, max_order=1:
            r=1 -> [(0,), (1,), (2,)]            # C(3,1)=3
        - k=3, max_order=2:
            r=1 -> [(0,), (1,), (2,)]            # C(3,1)=3
            r=2 -> [(0, 1), (0, 2), (1, 2)]      # C(3,2)=3
        - k=4, max_order=3 (showing r=3 only):
            r=3 -> [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]  # C(4,3)=4

    Returns:
        Dict[int, List[Tuple[int, ...]]]: mapping order r to list of index
        tuples, each tuple selecting r positions from range(k) without
        replacement.
    """
    return {r: list(combinations(range(k), r)) for r in range(1, min(k, max_order) + 1)}


def encode_combination_class_id(object_ids_sorted, base_U):
    """Encode an order-invariant combination of object ids into a single integer.

    Useful when you need a compact single-target class id per combination
    instead of a list of object ids. Input must be sorted to ensure order
    invariance.

    Encoding scheme (base-U, left-fold):
        cid_0 = 0
        cid_{i+1} = cid_i * U + oid_i

    Examples:
        - U=10, ids=[3]       -> 3
        - U=10, ids=[3, 7]    -> 3*10 + 7 = 37
        - U=100, ids=[3, 7]   -> 3*100 + 7 = 307
        - U=50, ids=[1, 0, 2] -> ((1*50)+0)*50 + 2 = 2502
    """
    cid = 0
    for oid in object_ids_sorted:
        cid = cid * base_U + int(oid)
    return cid


def precompute_caches_per_k(data, metadata, indices_by_k, object_to_id, max_order, encode_as="ids"):
    """Precompute per-scene labels for each k to make __getitem__ O(1).

    For each scene with k objects we materialize:
      - concepts: Dict[int, List[int]]  # per-concept labels, length k
      - object_ids: List[int]           # global stable ids for objects in scene
      - combinations: Dict[int, ...]    # for each r, either list of object-id lists
                                        # or list of encoded class ids
    encode_as: "ids" | "class_ids" | "both"

    Example (shape only):
        Suppose k=3, max_order=2. For one scene:
          concepts = {0: [c00, c01, c02], 1: [c10, c11, c12], ...}
          object_ids = [o0, o1, o2]
          combinations = {
              1: [[o0], [o1], [o2]]                 # C(3,1)=3
              2: [[o0, o1], [o0, o2], [o1, o2]]     # C(3,2)=3
          }`
        If encode_as == "class_ids", the lists for 1 and 2 become ints per entry.
    """
    U = len(object_to_id)
    C = metadata["num_concepts"]
    caches = {}
    for k, scene_indices in sorted(indices_by_k.items()):
        comb_pos = combination_positions_by_order(k, max_order)
        per_scene = []
        for scene_idx in scene_indices:
            objects = data[scene_idx]  # length-k list of object-attr lists

            # Concept labels per concept c: length-k lists
            concepts = {c: [int(obj[c]) for obj in objects] for c in range(C)}

            # Map objects in this scene to global stable ids
            object_ids = [object_to_id[tuple(int(v) for v in obj)] for obj in objects]

            # Build combination labels per order r
            combinations_dict = {}
            combinations_class_ids_dict = {}
            for r, pos_list in comb_pos.items():
                # Always compute raw ids when requested (or when returning both)
                if encode_as in ("ids", "both"):
                    combos_ids = [[object_ids[p] for p in pos] for pos in pos_list]
                    combinations_dict[r] = combos_ids
                # Optionally compute class ids for compact single-target encoding
                if encode_as in ("class_ids", "both"):
                    combos_cids = [
                        encode_combination_class_id(sorted(object_ids[p] for p in pos), U)
                        for pos in pos_list
                    ]
                    combinations_class_ids_dict[r] = combos_cids

            item = {
            "scene_idx": scene_idx,
                "k": k,
                "concepts": concepts,
                "object_ids": object_ids,
            }
            if encode_as == "ids":
                item["combinations"] = combinations_dict
            elif encode_as == "class_ids":
                item["combinations"] = combinations_class_ids_dict
            else:  # both
                item["combinations"] = combinations_dict
                item["combinations_class_ids"] = combinations_class_ids_dict

            per_scene.append(item)
        caches[k] = per_scene
    return caches


class CountDatasetPrecomputed:
    """Simple wrapper over a precomputed list of per-scene dicts for a fixed k.

    __getitem__ returns the precomputed dict directly (O(1)).

    Example item structure:
        item = {
            "scene_idx": 42,
            "k": 3,
            "concepts": {0: [v00, v01, v02], 1: [v10, v11, v12], ...},
            "object_ids": [o0, o1, o2],
            "combinations": {
                1: [[o0], [o1], [o2]],
                2: [[o0, o1], [o0, o2], [o1, o2]],
            },
        }
    """

    def __init__(self, precomputed_scenes):
        self._items = precomputed_scenes

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def tensorize_full_per_k(datasets_per_k, device="cpu"):
    """Stack all precomputed items per k into Torch tensors (no batching).
    
    Returns dict k -> {
        "B": int,                         # number of scenes for this k
        "k": int,                         # object count
        "scene_idx": LongTensor[B],       # original scene indices
        "y_concepts": Dict[int, LongTensor[B, k]],
        "y_object_ids": LongTensor[B, k],
        "y_combinations": Dict[int, LongTensor],           # ids: [B, C(k,r), r], class_ids: [B, C(k,r)]
        "y_combinations_class": Dict[int, LongTensor],     # present if precomputed with encode_as="both"
        "positions": Dict[int, LongTensor[T, r]],          # shared combination index positions per r
    }

    Examples (shapes only): for k=3, max_order=2
        y_concepts[c] -> [B, 3]
        y_object_ids -> [B, 3]
        y_combinations[1] -> [B, 3, 1] (ids mode) or [B, 3] (class_ids)
        y_combinations[2] -> [B, 3, 2] (ids mode) or [B, 3] (class_ids)
    """

    out = {}
    for k, ds in sorted(datasets_per_k.items()):
        if len(ds) == 0:
            continue

        B = len(ds)
        k_fixed = ds[0]["k"]
        concept_keys = sorted(ds[0]["concepts"].keys())

        # Concepts: Dict[int, LongTensor[B, k]]
        y_concepts = {
            c: torch.tensor([ds[i]["concepts"][c] for i in range(B)], dtype=torch.long, device=device)
            for c in concept_keys
        }

        # Per-object ids: LongTensor[B, k]
        y_object_ids = torch.tensor([ds[i]["object_ids"] for i in range(B)], dtype=torch.long, device=device)

        # Combinations per order r:
        # - ids mode -> LongTensor[B, C(k,r), r]
        # - class_ids mode -> LongTensor[B, C(k,r)]
        first_item = ds[0]
        y_combinations = {
            r: torch.tensor([ds[i]["combinations"][r] for i in range(B)], dtype=torch.long, device=device)
            for r in first_item["combinations"].keys()
        }
        # Shared positions per r (same for all scenes of this k)
        positions = {}
        if len(first_item["combinations"]) > 0:
            max_r = max(first_item["combinations"].keys())
            pos_all = combination_positions_by_order(k_fixed, max_r)
            for r in first_item["combinations"].keys():
                positions[r] = torch.tensor(pos_all[r], dtype=torch.long, device=device)
        # Optionally include class-id-encoded combinations if present
        y_combinations_class = None
        if "combinations_class_ids" in first_item:
            y_combinations_class = {
                r: torch.tensor([ds[i]["combinations_class_ids"][r] for i in range(B)], dtype=torch.long, device=device)
                for r in first_item["combinations_class_ids"].keys()
            }

        scene_idx = torch.tensor([ds[i]["scene_idx"] for i in range(B)], dtype=torch.long, device=device)

        pack = {
            "B": B,
            "k": k_fixed,
            "scene_idx": scene_idx,
            "y_concepts": y_concepts,
            "y_object_ids": y_object_ids,
            "y_combinations": y_combinations,
            "positions": positions,
        }
        if y_combinations_class is not None:
            pack["y_combinations_class"] = y_combinations_class
        out[k] = pack
    return out

def train_embeddings_simple(
    full_targets,
    num_concepts,
    values_per_concept,
    num_unique_objects,
    embedding_dim=64,
    epochs=1,
    lr=1e-3,
    weights=None,
    device="cpu",
    use_lr_cos: bool = False,
    lr_eta_min: float = 0.0,
    lr_T_max: Optional[int] = None,
    similarity: str = "dot",
    sim_scale: float = 10.0,
    make_box: bool = True,
    init_mode: str = "rand",
    cube_min: float = -1.0,
    cube_max: float = 1.0,
    fit_concepts: bool = False,
    fit_objects: bool = False,
    caption_embeddings: Optional[torch.Tensor] = None,
    train_ratio: float = 1.0,
):
    """Train per-scene embeddings with scene-level soft-CE targets.

    Losses:
      - Concepts: soft CE over [B,V] (values present anywhere in scene)
      - Objects: soft CE over [B,U] (objects present anywhere in scene)

    weights: dict keys {"concept", "object"}
    """

    if weights is None:
        weights = {"concept": 1.0, "object": 1.0}
    else:
        weights = weights.copy()
        weights.setdefault("concept", 1.0)
        weights.setdefault("object", 1.0)

    if isinstance(values_per_concept, int):
        concept_value_sizes = [int(values_per_concept)] * num_concepts
    else:
        concept_value_sizes = [int(v) for v in values_per_concept]
        if len(concept_value_sizes) != num_concepts:
            raise ValueError(
                f"Expected {num_concepts} entries in values_per_concept, got {len(concept_value_sizes)}"
            )
    if any(v <= 0 for v in concept_value_sizes):
        raise ValueError(f"Concept cardinalities must be >= 1, got {concept_value_sizes}")

    # Train/test split is only supported when using frozen caption embeddings.
    # (In this mode, "embeddings" are fixed features; probes are trained on top.)
    if caption_embeddings is None and float(train_ratio) != 1.0:
        raise ValueError("train_ratio is only supported when caption_embeddings are provided (i.e. --embedding_path).")
    if not (0.0 < float(train_ratio) <= 1.0):
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")

    # Build learnable embeddings and probes

    # init all embeddings: as many as there are scenes
    num_scenes = np.sum([pack["B"] for pack in full_targets.values()])
    if caption_embeddings is not None:
        print(f"Loading caption embeddings")
        embeddings = caption_embeddings[:num_scenes].clone().to(device).requires_grad_(False)
        embeddings.requires_grad_(False)
        print(f"Embeddings loaded: {embeddings.shape}")
    else:
        embeddings = nn.Parameter(torch.randn(num_scenes, embedding_dim).to(device), requires_grad=True)

    # Deterministic train/test split over global scene indices
    split_active = (caption_embeddings is not None) and (float(train_ratio) < 1.0)
    is_train = None
    train_ids = None
    test_ids = None
    if split_active:
        if num_scenes < 2:
            raise ValueError(f"Need at least 2 scenes to split, got num_scenes={num_scenes}")
        # Use a fixed seed to make splits reproducible unless/until we expose it as an arg.
        split_seed = 0
        g = torch.Generator(device="cpu")
        g.manual_seed(split_seed)
        perm = torch.randperm(int(num_scenes), generator=g, device="cpu")
        n_train = int(float(train_ratio) * int(num_scenes))
        n_train = max(1, min(int(num_scenes) - 1, n_train))
        train_ids = perm[:n_train].to(device)
        test_ids = perm[n_train:].to(device)
        is_train = torch.zeros(int(num_scenes), dtype=torch.bool, device=device)
        is_train[train_ids] = True
        print(
            {
                "split_active": True,
                "train_ratio": float(train_ratio),
                "split_seed": split_seed,
                "num_scenes": int(num_scenes),
                "n_train": int(train_ids.numel()),
                "n_test": int(test_ids.numel()),
            }
        )

    max_obj_pred_counts = {}
    for pack in full_targets.values():
        if "y_combinations_class" not in pack:
            continue
        for r, v in pack["y_combinations_class"].items():
            curr = int(v.max().item()) + 1
            if r not in max_obj_pred_counts or curr > max_obj_pred_counts[r]:
                max_obj_pred_counts[r] = curr

    # we can now make probes for each object too 
    probes_single_concept = nn.ModuleDict(
        {str(i): nn.Linear(embedding_dim, concept_value_sizes[i], bias=False) for i in range(num_concepts) }
    )
    probes_single_concept = probes_single_concept.to(device)
    probes_object = nn.ModuleDict({str(t): nn.Linear(embedding_dim, v, bias=False) for t, v in max_obj_pred_counts.items()})
    probes_object = probes_object.to(device)

    # Make temperature learnable when using cosine similarity
    temperature_raw = None
    extra_params = list(probes_single_concept.parameters()) + list(probes_object.parameters())

    if similarity == "cos":
        # Initialize raw parameter such that exp(raw) = sim_scale
        # So raw = log(sim_scale)
        init_val = torch.log(torch.tensor(sim_scale))
        temperature_raw = nn.Parameter(init_val.to(device), requires_grad=True)
        params = [embeddings, temperature_raw] + extra_params
    else:
        params = [embeddings] + extra_params
    
    optimizer = Adam(params, lr=lr)
    # optional cosine annealing scheduler to decay LR to eta_min by final epoch
    scheduler = None
    if use_lr_cos:
        T_max_value = lr_T_max if lr_T_max is not None else epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max_value, eta_min=lr_eta_min)

    def compute_logits(probe: nn.Linear, x, mode: str) :
        # x: [B, D], probe.weight: [C, D]
        if mode == "cos":
            x_norm = F.normalize(x, p=2, dim=1)
            w = probe.weight
            w_norm = F.normalize(w, p=2, dim=1)
            # Apply exp to ensure temperature is positive
            scale = torch.exp(temperature_raw) if temperature_raw is not None else sim_scale
            return scale * (x_norm @ w_norm.t())
        # default dot-product linear layer
        return probe(x)
    # Precompute target matrices once (they don't change across epochs)
    print("Precomputing target matrices...")
    precomputed_targets = {}
    with torch.no_grad():
        for pack in full_targets.values():
            B = pack["B"]
            k = pack["k"]
            y_concepts = pack["y_concepts"]
            y_combinations_class = pack["y_combinations_class"]

            # Precompute concept target matrices
            concept_targets = {}
            for c in y_concepts.keys():
                y_c = y_concepts[c]
                vocab_size = concept_value_sizes[c]
                y_c_matrix = torch.zeros(B, vocab_size, device=device, dtype=torch.long)
                y_c_matrix.scatter_add_(dim=-1, index=y_c, src=torch.ones_like(y_c))
                y_c_matrix = y_c_matrix.float() / y_c_matrix.sum(dim=-1, keepdim=True).float()
                concept_targets[c] = y_c_matrix

            # Precompute object combination target matrices
            object_targets = {}
            for r in y_combinations_class.keys():
                y_r = y_combinations_class[r]
                # Guard against out-of-range class ids
                vocab_size = max_obj_pred_counts[r]
                assert int(y_r.max().item()) < vocab_size, f"y_r has class id {int(y_r.max().item())} >= vocab_size {vocab_size} for order r={r}"
                y_r_matrix = torch.zeros(B, vocab_size).long().to(device)
                y_r_matrix.scatter_add_(dim=-1, index=y_r, src=torch.ones_like(y_r))
                y_r_matrix = y_r_matrix.float() / y_r_matrix.sum(dim=-1, keepdim=True).float()
                object_targets[r] = y_r_matrix

            precomputed_targets[k] = {
                "concept_targets": concept_targets,
                "object_targets": object_targets,
            }
    print(f"Precomputation done for {len(precomputed_targets)} packs.")

    def _avg_map(values_by_pack):
        out = {}
        for pack_key, vals in values_by_pack.items():
            if len(vals) == 0:
                continue
            out[str(pack_key)] = float(np.mean(vals))
        return out

    final_summary_metrics: Dict[str, Any] = {}

    for epoch in range(epochs):
        total_loss = 0.0
        concept_accs_per_pack = {}
        object_accs_per_pack = {}
        test_concept_accs_per_pack = {}
        test_object_accs_per_pack = {}

        concept_loss_per_pack = {}
        object_loss_per_pack = {}
        test_concept_loss_per_pack = {}
        test_object_loss_per_pack = {}

        optimizer.zero_grad(set_to_none=True)
        log_step = (epoch % 500 == 0) or (epoch == epochs - 1)

        for pack in full_targets.values():
            B = pack["B"]
            k = pack["k"]
            scene_idx = pack["scene_idx"]
            y_concepts = pack["y_concepts"]
            y_combinations_class = pack["y_combinations_class"]

            # Get precomputed targets for this pack
            pack_targets = precomputed_targets[k]

            # optimize the basic concept embeddings first
            if split_active:
                mask_train = is_train[scene_idx]
                mask_test = ~mask_train
            else:
                mask_train = None
                mask_test = None

            embeddings_scene = embeddings[scene_idx] if not split_active else embeddings[scene_idx[mask_train]]
            for c in y_concepts.keys():
                if not fit_concepts:
                    continue
                y_c = y_concepts[c] if not split_active else y_concepts[c][mask_train]
                # Use precomputed target matrix
                y_c_matrix = pack_targets["concept_targets"][c] if not split_active else pack_targets["concept_targets"][c][mask_train]

                if split_active and embeddings_scene.numel() == 0:
                    continue
                logits_c = compute_logits(probes_single_concept[str(c)], embeddings_scene, similarity)
                loss_c = F.cross_entropy(logits_c, y_c_matrix)

                if log_step:
                    # track accuracy: min correct should be higher than max incorrect.
                    pos_logits = torch.gather(logits_c, 1, y_c)
                    min_pos = pos_logits.min(dim=-1).values

                    mask_positives = torch.zeros_like(logits_c, dtype=torch.bool).to(device)
                    mask_positives.scatter_(1, y_c, True)
                    neg_logits = logits_c.masked_fill(mask_positives, -float('inf'))
                    max_neg = neg_logits.max(dim=-1).values

                    accuracy = min_pos > max_neg

                    if k not in concept_accs_per_pack:
                        concept_accs_per_pack[k] = []
                    denom = int(y_c.shape[0]) if split_active else int(B)
                    concept_accs_per_pack[k] += [accuracy.sum().item() / max(1, denom)]

                if k not in concept_loss_per_pack:
                    concept_loss_per_pack[k] = []
                concept_loss_per_pack[k] += [loss_c.item()]

                total_loss += loss_c

                # Test metrics (no grad) on held-out scenes
                if split_active and log_step and mask_test.any():
                    with torch.no_grad():
                        emb_test = embeddings[scene_idx[mask_test]]
                        y_c_test = y_concepts[c][mask_test]
                        y_c_matrix_test = pack_targets["concept_targets"][c][mask_test]
                        test_logits_c = compute_logits(probes_single_concept[str(c)], emb_test, similarity)
                        test_loss_c = F.cross_entropy(test_logits_c, y_c_matrix_test)
                        pos_logits_t = torch.gather(test_logits_c, 1, y_c_test)
                        min_pos_t = pos_logits_t.min(dim=-1).values
                        mask_pos_t = torch.zeros_like(test_logits_c, dtype=torch.bool).to(device)
                        mask_pos_t.scatter_(1, y_c_test, True)
                        neg_logits_t = test_logits_c.masked_fill(mask_pos_t, -float("inf"))
                        max_neg_t = neg_logits_t.max(dim=-1).values
                        acc_t = (min_pos_t > max_neg_t).float().mean().item()
                        if k not in test_concept_accs_per_pack:
                            test_concept_accs_per_pack[k] = []
                        test_concept_accs_per_pack[k] += [acc_t]
                        if k not in test_concept_loss_per_pack:
                            test_concept_loss_per_pack[k] = []
                        test_concept_loss_per_pack[k] += [test_loss_c.item()]

            # # optimize per-object loss too 
            for r in y_combinations_class.keys():
                if not fit_objects:
                    continue
                y_r = y_combinations_class[r] if not split_active else y_combinations_class[r][mask_train]
                # Use precomputed target matrix
                y_r_matrix = pack_targets["object_targets"][r] if not split_active else pack_targets["object_targets"][r][mask_train]
                if split_active and embeddings_scene.numel() == 0:
                    continue
                logits_r = compute_logits(probes_object[str(r)], embeddings_scene, similarity)
                loss_r = F.cross_entropy(logits_r, y_r_matrix)
                total_loss += loss_r

                if log_step:
                    pos_logits = torch.gather(logits_r, 1, y_r)
                    min_pos = pos_logits.min(dim=-1).values
                    mask_positives = torch.zeros_like(logits_r, dtype=torch.bool).to(device)
                    mask_positives.scatter_(1, y_r, True)
                    neg_logits = logits_r.masked_fill(mask_positives, -float('inf'))
                    max_neg = neg_logits.max(dim=-1).values
                    accuracy = min_pos > max_neg
                    if k not in object_accs_per_pack:
                        object_accs_per_pack[k] = []
                    denom = int(y_r.shape[0]) if split_active else int(B)
                    object_accs_per_pack[k] += [accuracy.sum().item() / max(1, denom)]

                if k not in object_loss_per_pack:
                    object_loss_per_pack[k] = []
                object_loss_per_pack[k] += [loss_r.item()]

                # Test metrics (no grad) on held-out scenes
                if split_active and log_step and mask_test.any():
                    with torch.no_grad():
                        emb_test = embeddings[scene_idx[mask_test]]
                        y_r_test = y_combinations_class[r][mask_test]
                        y_r_matrix_test = pack_targets["object_targets"][r][mask_test]
                        test_logits_r = compute_logits(probes_object[str(r)], emb_test, similarity)
                        test_loss_r = F.cross_entropy(test_logits_r, y_r_matrix_test)
                        pos_logits_t = torch.gather(test_logits_r, 1, y_r_test)
                        min_pos_t = pos_logits_t.min(dim=-1).values
                        mask_pos_t = torch.zeros_like(test_logits_r, dtype=torch.bool).to(device)
                        mask_pos_t.scatter_(1, y_r_test, True)
                        neg_logits_t = test_logits_r.masked_fill(mask_pos_t, -float("inf"))
                        max_neg_t = neg_logits_t.max(dim=-1).values
                        acc_t = (min_pos_t > max_neg_t).float().mean().item()
                        if k not in test_object_accs_per_pack:
                            test_object_accs_per_pack[k] = []
                        test_object_accs_per_pack[k] += [acc_t]
                        if k not in test_object_loss_per_pack:
                            test_object_loss_per_pack[k] = []
                        test_object_loss_per_pack[k] += [test_loss_r.item()]

            # now backprop jointly
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Print and log concept accuracies and losses
        if log_step:
            wandb_log = {"epoch": epoch, "total_loss": total_loss.item()}

            # Log learnable temperature if using cosine similarity
            if temperature_raw is not None:
                temp_value = torch.exp(temperature_raw).item()
                wandb_log["temperature"] = temp_value
                print(f"Epoch {epoch} - Temperature: {temp_value:.4f}")

            if concept_accs_per_pack:
                print(f"Epoch {epoch} - Train Concept Accuracies:")
                for pack_key, accuracies in concept_accs_per_pack.items():
                    avg_acc = np.mean(accuracies)
                    print(f"  Pack {pack_key}: {avg_acc:.4f}")
                    wandb_log[f"train/concept_acc/pack_{pack_key}"] = avg_acc
            if split_active and test_concept_accs_per_pack:
                print(f"Epoch {epoch} - Test Concept Accuracies:")
                for pack_key, accuracies in test_concept_accs_per_pack.items():
                    avg_acc = np.mean(accuracies)
                    print(f"  Pack {pack_key}: {avg_acc:.4f}")
                    wandb_log[f"test/concept_acc/pack_{pack_key}"] = avg_acc
            if object_accs_per_pack:
                print(f"Epoch {epoch} - Train Object Accuracies:")
                for pack_key, accuracies in object_accs_per_pack.items():
                    avg_acc = np.mean(accuracies)
                    print(f"  Pack {pack_key}: {avg_acc:.4f}")
                    wandb_log[f"train/object_acc/pack_{pack_key}"] = avg_acc
            if split_active and test_object_accs_per_pack:
                print(f"Epoch {epoch} - Test Object Accuracies:")
                for pack_key, accuracies in test_object_accs_per_pack.items():
                    avg_acc = np.mean(accuracies)
                    print(f"  Pack {pack_key}: {avg_acc:.4f}")
                    wandb_log[f"test/object_acc/pack_{pack_key}"] = avg_acc
            if concept_loss_per_pack:
                print(f"Epoch {epoch} - Train Concept Losses:")
                for pack_key, losses in concept_loss_per_pack.items():
                    avg_loss = np.mean(losses)
                    print(f"  Pack {pack_key}: {avg_loss:.4f}")
                    wandb_log[f"train/concept_loss/pack_{pack_key}"] = avg_loss
            if split_active and test_concept_loss_per_pack:
                print(f"Epoch {epoch} - Test Concept Losses:")
                for pack_key, losses in test_concept_loss_per_pack.items():
                    avg_loss = np.mean(losses)
                    print(f"  Pack {pack_key}: {avg_loss:.4f}")
                    wandb_log[f"test/concept_loss/pack_{pack_key}"] = avg_loss
            if object_loss_per_pack:
                print(f"Epoch {epoch} - Train Object Losses:")
                for pack_key, losses in object_loss_per_pack.items():
                    avg_loss = np.mean(losses)
                    print(f"  Pack {pack_key}: {avg_loss:.4f}")
                    wandb_log[f"train/object_loss/pack_{pack_key}"] = avg_loss
            if split_active and test_object_loss_per_pack:
                print(f"Epoch {epoch} - Test Object Losses:")
                for pack_key, losses in test_object_loss_per_pack.items():
                    avg_loss = np.mean(losses)
                    print(f"  Pack {pack_key}: {avg_loss:.4f}")
                    wandb_log[f"test/object_loss/pack_{pack_key}"] = avg_loss
            
            wandb.log(wandb_log)

            # Snapshot machine-readable metrics for export consumers.
            final_summary_metrics = {
                "epoch": int(epoch),
                "train": {
                    "concept_acc": _avg_map(concept_accs_per_pack),
                    "object_acc": _avg_map(object_accs_per_pack),
                    "concept_loss": _avg_map(concept_loss_per_pack),
                    "object_loss": _avg_map(object_loss_per_pack),
                },
                "test": {
                    "concept_acc": _avg_map(test_concept_accs_per_pack),
                    "object_acc": _avg_map(test_object_accs_per_pack),
                    "concept_loss": _avg_map(test_concept_loss_per_pack),
                    "object_loss": _avg_map(test_object_loss_per_pack),
                },
            }
        # print(f"Epoch {epoch}, Concept Accuracy: {np.mean(concept_accuracy_total)}")

    
    result = {
        "embeddings": embeddings,
        "probes_single_concept": probes_single_concept,
        "probes_object": probes_object,
        "similarity": similarity,  # Save which similarity mode was used
        "sim_scale": sim_scale,    # Save the similarity scale parameter
    }
    if temperature_raw is not None:
        result["temperature_raw"] = temperature_raw
        result["temperature"] = torch.exp(temperature_raw)  # Also save the actual temperature value
    result["summary_metrics"] = final_summary_metrics
    
    return result


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Load dataset and metadata")
    parser.add_argument(
        "--dataset-path",
        type=str,
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/checks/objs2_concepts2_values3_dedup_max1000000_actual54_mixed_20251001-011105_True/dataset.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/objs2_concepts2_values10_nodedup_max1000000000_actual10100_mixed_20260104-093043_False/dataset.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/objs2_concepts2_values20_nodedup_max1000000000_actual160400_mixed_20260104-111651_False/dataset.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/pug_spare/dataset.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/objs2_concepts2_values20_nodedup_max1000000000_actual160400_mixed_20260104-111651_False/dataset.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/pug_spare/dataset.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/pug_spare/dataset.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/checks/objs2_concepts2_values10_dedup_max1000000_actual5150_mixed_20251001-011106_True/dataset.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/checks/objs2_concepts2_values4_dedup_max1000000_actual152_mixed_20251001-011105_True/dataset.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/2026_01_17_14:43:14.337155___15763b6f93b9695e1193___502224/dataset.pkl",
        default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/clevr/dataset.pkl",
        help="Path to dataset folder (containing dataset.pkl) or to dataset.pkl file.",
    ) 

    parser.add_argument(
        "--num-objects-predict",
        type=int,
        default=1,  #2 -> we also fit "red square and blue circle"
        help="Number of objects to predict.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
        help="Embedding dimensionality.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["none", "cos"],
        default="cos",
        help="LR scheduler to use (none or cos).",
    )
    parser.add_argument(
        "--lr-eta-min",
        type=float,
        default=0.0,
        help="Minimum LR for cosine scheduler (eta_min).",
    )
    parser.add_argument(
        "--lr-T-max",
        type=int,
        default=None,
        help="T_max for cosine scheduler; defaults to epochs if not set.",
    )
    parser.add_argument(
        "--similarity",
        type=str,
        choices=["dot", "cos"],
        default="dot",
        help="Similarity for logits: dot (linear) or cos (unit-norm).",
    )
    parser.add_argument(
        "--sim-scale",
        type=float,
        default=1.0,
        help="Scaling factor applied to cosine similarity logits.",
    )
    parser.add_argument(
        "--make_box",
        type=bool,
        default=False,
        help="Make a box of the objects (For single objects and don't backprop).",
    )

    # Initialization options
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=["rand", "cube"],
        default="rand",
        help="Embedding initialization: random normal or cube-aligned by concepts.",
    )
    parser.add_argument(
        "--cube-min",
        type=float,
        default=-1.0,
        help="Minimum value for cube initialization mapping.",
    )
    parser.add_argument(
        "--cube-max",
        type=float,
        default=1.0,
        help="Maximum value for cube initialization mapping.",
    )
    
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/clevr/clevr_dinov2-vitb14_embeddings.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/pug_spare/Desert_ViT-B_32-rand_embeddings.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/objs2_concepts2_values20_nodedup_max1000000000_actual160400_mixed_20260104-111651_False/no_concepts_clip_ViT-B_32_text_embeddings.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/pug_spare/no_obj_Desert_ViT-B_32_embeddings.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/pug_spare/no_concepts_Desert_ViT-B_32_embeddings.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/pug_spare/Desert_ViT-B_32_embeddings.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/pug_spare/no_obj_Desert_ViT-B_32_embeddings.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/pug_spare/no_concepts_Desert_ViT-B_32_embeddings.pkl",
        # default="/mnt/lustre/work/oh/owl661/mob_project/data/clip_checks/pug_spare/True_no_concepts_Desert_ViT-B_32_embeddings.pkl",
        help="Path to embeddings file.",
    )

    parser.add_argument(
        "--train_ratio",
        "--train-ratio",
        type=float,
        default=0.4,
        # default=1.0,
        help="Train split ratio in (0,1]. Only supported when --embedding_path is provided.",
    )
    
    parser.add_argument(
        "--fit-concepts",
        type=bool,
        default=True,
    )
    
    parser.add_argument(
        "--fit-objects",
        type=bool,
        default=True,
    )

    args = parser.parse_args(argv)

    data, metadata = load_dataset(Path(args.dataset_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Treat common sentinel strings as "no embeddings"
    if isinstance(args.embedding_path, str) and args.embedding_path.strip().lower() in {"none", "null", ""}:
        args.embedding_path = None

    if args.embedding_path is not None:
        print(f"Loading embeddings from {args.embedding_path}")
        caption_embeddings = pickle.load(open(args.embedding_path, "rb"))
        caption_embeddings = torch.tensor(caption_embeddings).float()
        caption_embeddings = caption_embeddings.to(device)
        print(f"Embeddings loaded: {caption_embeddings.shape}")
        print(f"Overwriting dimension of embeddings from {args.embedding_dim} to {caption_embeddings.shape[1]}")
        args.embedding_dim = caption_embeddings.shape[1]
        if not (0.0 < float(args.train_ratio) <= 1.0):
            raise ValueError(f"--train-ratio must be in (0, 1], got {args.train_ratio}")
    else:
        caption_embeddings = None
        if float(args.train_ratio) != 1.0:
            raise ValueError("--train-ratio is only supported when --embedding_path is provided.")

    num_objs = metadata["num_objects"]
    num_concepts = metadata["num_concepts"]

    raw_values = metadata.get("values_per_concept")
    if raw_values is None:
        raw_values = metadata.get("num_values_per_concept")
    if raw_values is None:
        raise KeyError("Metadata must include 'values_per_concept' or 'num_values_per_concept'")

    if isinstance(raw_values, int):
        values_per_concept = tuple([int(raw_values)] * num_concepts)
    else:
        values_per_concept = tuple(int(v) for v in raw_values)
        if len(values_per_concept) != num_concepts:
            raise ValueError(
                f"Expected {num_concepts} concept value counts, got {len(values_per_concept)}: {values_per_concept}"
            )

    def _format_values_label(values: tuple[int, ...]) -> str:
        if not values:
            return "[]"
        if len(set(values)) == 1:
            return str(values[0])
        return "[" + ",".join(str(v) for v in values) + "]"

    values_label = _format_values_label(values_per_concept)

    num_objects_predict = min(num_objs, args.num_objects_predict)
    
    # Initialize wandb with informative run name
    run_name = (
        f"emb_objs{num_objs}_concepts{num_concepts}_vals{values_label}_"
        f"pred{num_objects_predict}_dim{args.embedding_dim}_ep{args.epochs}_lr{args.lr}_sim{args.similarity}"
    )
    if args.similarity == "cos":
        run_name += f"_scale{args.sim_scale}"
    if args.lr_scheduler == "cos":
        run_name += f"_cosann"
    if args.fit_concepts:
        run_name += "_fit_concepts"
    if args.fit_objects:
        run_name += "_fit_objects"
    if caption_embeddings is not None and float(args.train_ratio) < 1.0:
        run_name += f"_trainratio{args.train_ratio}"
    wandb.init(
        project="mob-embeddings",
        name=run_name,
        config={
            "num_objects": num_objs,
            "num_concepts": num_concepts,
            "values_per_concept": list(values_per_concept),
            "num_objects_predict": num_objects_predict,
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "lr_eta_min": args.lr_eta_min,
            "lr_T_max": args.lr_T_max,
            "similarity": args.similarity,
            "sim_scale": args.sim_scale,
            "dataset_path": str(args.dataset_path),
            "make_box": args.make_box,
            "init_mode": args.init_mode,
            "cube_min": args.cube_min,
            "cube_max": args.cube_max,
            "fit_concepts": args.fit_concepts,
            "fit_objects": args.fit_objects,
            "embedding_path": args.embedding_path,
            "train_ratio": float(args.train_ratio),
        },
        mode="offline",
    )
    # make converters from tuples of concept values to object indices
    
    # Keep this trainer focused on 2-object scenes only.
    kept_indices = [i for i, scene in enumerate(data) if len(scene) == 2]
    dropped = len(data) - len(kept_indices)
    if dropped > 0:
        print({"dropped_non_two_object_scenes": int(dropped)})
    if not kept_indices:
        raise ValueError("No scenes with exactly 2 objects were found in dataset.")

    data = [data[i] for i in kept_indices]

    if caption_embeddings is not None:
        # Align embeddings to filtered 2-object scenes.
        if int(caption_embeddings.shape[0]) == int(len(kept_indices) + dropped):
            keep_tensor = torch.as_tensor(kept_indices, device=caption_embeddings.device, dtype=torch.long)
            caption_embeddings = caption_embeddings[keep_tensor]
        elif int(caption_embeddings.shape[0]) != int(len(data)):
            print("Trimming scenes/embeddings to shared prefix after 2-object filtering.")
            shared_n = min(int(caption_embeddings.shape[0]), int(len(data)))
            caption_embeddings = caption_embeddings[:shared_n]
            data = data[:shared_n]
        print(f"Embeddings aligned to 2-object scenes: {tuple(caption_embeddings.shape)}")

    print({"data_loaded": data is not None, "metadata": metadata, "num_scenes_used": len(data)})

    # ------------------------------
    # Build precomputed cache (k=2 only)
    # ------------------------------
    indices_by_k = {2: list(range(len(data)))}
    object_to_id, _ = build_object_id_map(data)
    caches = precompute_caches_per_k(
        data=data,
        metadata=metadata,
        indices_by_k=indices_by_k,
        object_to_id=object_to_id,
        max_order=num_objects_predict,
        encode_as="both",  # provide both raw ids and compact class ids
    )

    datasets_per_k = {2: CountDatasetPrecomputed(caches[2])}

    # Small summary: number of scenes per k, and counts of combinations per r for the first scene
    summary = {}
    for k, ds in datasets_per_k.items():
        if len(ds) == 0:
            summary[k] = {"num_scenes": 0}
            continue
        sample = ds[0]
        comb_sizes = {r: len(v) for r, v in sample["combinations"].items()}
        summary[k] = {
            "num_scenes": len(ds),
            "combination_counts": comb_sizes,  # should match C(k, r)
        }
    print({"datasets_per_k": summary})
    
    # Log dataset summary to wandb
    for k, info in summary.items():
        wandb.log({f"dataset/num_scenes_k{k}": info.get("num_scenes", 0)})

    # Choose device: prefer CUDA if available
    print({"device": device})
    wandb.config.update({"device": device})

    full_targets = tensorize_full_per_k(datasets_per_k, device=device)
    shapes = {}
    for k, pack in sorted(full_targets.items()):
        comb_shapes = {r: tuple(t.shape) for r, t in pack["y_combinations"].items()}
        comb_class_shapes = (
            {r: tuple(t.shape) for r, t in pack["y_combinations_class"].items()}
            if "y_combinations_class" in pack else None
        ) 
        concept_shapes = {c: tuple(t.shape) for c, t in pack["y_concepts"].items()}
        shapes[k] = {
            "B": pack["B"],
            "k": pack["k"],
            "scene_idx": tuple(pack["scene_idx"].shape),
            "y_object_ids": tuple(pack["y_object_ids"].shape),
            "y_concepts": concept_shapes,
            "y_combinations": comb_shapes,
            "y_combinations_class": comb_class_shapes,
        }
    print({"full_targets_shapes": shapes})

    # -----------------
    # Optional training
    # -----------------
    result: Dict[str, Any] = {"training_skipped": False, "embeddings_saved": None}

    try:
        num_unique_objects = len(object_to_id)
        trained_obj = train_embeddings_simple(
            full_targets=full_targets,
            num_concepts=num_concepts,
            values_per_concept=values_per_concept,
            num_unique_objects=num_unique_objects,
            embedding_dim=args.embedding_dim,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            use_lr_cos=(args.lr_scheduler == "cos"),
            lr_eta_min=args.lr_eta_min,
            lr_T_max=args.lr_T_max,
            similarity=args.similarity,
            sim_scale=args.sim_scale,
            make_box=args.make_box,
            init_mode=args.init_mode,
            cube_min=args.cube_min,
            cube_max=args.cube_max,
            fit_concepts=args.fit_concepts,
            fit_objects=args.fit_objects,
            caption_embeddings=caption_embeddings,
            train_ratio=float(args.train_ratio),
        )
        
        # Save trained embeddings in dataset folder (handle file or directory input)
        dataset_path = Path(args.dataset_path)
        dataset_dir = dataset_path if dataset_path.is_dir() else dataset_path.parent
        embeddings_dir = dataset_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the trained object embeddings with dimension and similarity in filename
        postfix = ""
        if args.embedding_path is not None:
            # get the filename of the path
            filename = os.path.basename(args.embedding_path)
            postfix = f"_{filename}"
        embeddings_path = embeddings_dir / f"trained_embeddings_dim{args.embedding_dim}_sim{args.similarity}_fit_concepts{args.fit_concepts}_fit_objects{args.fit_objects}_{args.train_ratio}_{postfix}.pt"
        torch.save(trained_obj, embeddings_path)
        print({"embeddings_saved": str(embeddings_path)})
        result["embeddings_saved"] = str(embeddings_path)
        if isinstance(trained_obj, dict) and "summary_metrics" in trained_obj:
            result["summary_metrics"] = trained_obj["summary_metrics"]
        
        # Log embeddings path and final stats to wandb
        final_log = {
            "embeddings_path": str(embeddings_path),
            "num_unique_objects": num_unique_objects,
        }

        # Log final learned temperature if using cosine similarity
        if "temperature" in trained_obj:
            final_temp = trained_obj["temperature"].item()  # Already exp-transformed
            final_log["final_temperature"] = final_temp
            print(f"Final learned temperature: {final_temp:.4f}")
        
        wandb.log(final_log)
        wandb.save(str(embeddings_path))
        
    except Exception as e:
        # If torch isn't installed, or GPU unavailable, skip training
        print({"training_skipped": True, "reason": str(e)})
        result["training_skipped"] = True
        result["reason"] = str(e)
    
    wandb.finish()
    return result



if __name__ == "__main__":
    main()
