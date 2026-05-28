import numpy as np
import torch
from torch.utils.data import IterableDataset


class SceneConstructor:
    def __init__(
        self,
        num_concepts: int,
        num_vals_per_concept: int,
        max_num_objects: int,
        allowed_objects=None,
    ):
        self.num_concepts = num_concepts
        self.num_vals_per_concept = num_vals_per_concept
        self.max_num_objects = max_num_objects
        self.allowed_objects = None
        if allowed_objects is not None:
            allowed_objects = np.asarray(allowed_objects, dtype=np.int64)
            if allowed_objects.ndim != 2 or allowed_objects.shape[1] != num_concepts:
                raise ValueError("allowed_objects must be shape [num_objects, num_concepts]")
            if allowed_objects.shape[0] == 0:
                raise ValueError("allowed_objects must contain at least one object")
            self.allowed_objects = allowed_objects

    def get_random_scenes(self, num_scenes, sampling_mode="random"):
        assert sampling_mode in ["random"]
        num_objects = np.random.randint(1, self.max_num_objects + 1, (num_scenes,))
        if self.allowed_objects is None:
            batch_grid_concept_values = np.random.randint(
                0,
                self.num_vals_per_concept,
                (num_scenes, self.max_num_objects, self.num_concepts),
            )
        else:
            obj_indices = np.random.randint(
                0, self.allowed_objects.shape[0], (num_scenes, self.max_num_objects)
            )
            batch_grid_concept_values = self.allowed_objects[obj_indices]

        scenes = []
        for scene_idx in range(num_scenes):
            scene_objects = []
            for obj_idx in range(num_objects[scene_idx]):
                scene_objects.append(batch_grid_concept_values[scene_idx, obj_idx, :])
            scenes.append(scene_objects)
        return scenes


def tokenize_scenes(
    scenes,
    soo_id,
    eoo_id,
    eos_id,
    pad_id,
    num_concepts,
    num_vals_per_concept,
    max_scene_size,
):
    batch = len(scenes)
    tokenized_scenes = np.zeros((batch, max_scene_size), dtype=np.int64)
    tokenized_scenes.fill(pad_id)
    value_add_shift = np.arange(num_concepts) * num_vals_per_concept
    for scene_idx, scene in enumerate(scenes):
        current_idx = 0
        for obj in scene:
            obj_shifted = obj + value_add_shift
            tokenized_scenes[scene_idx][current_idx] = soo_id
            tokenized_scenes[scene_idx][current_idx + 1 : current_idx + 1 + num_concepts] = (
                obj_shifted
            )
            tokenized_scenes[scene_idx][current_idx + 1 + num_concepts] = eoo_id
            current_idx = current_idx + 1 + num_concepts + 1
        tokenized_scenes[scene_idx][current_idx] = eos_id
    return torch.tensor(tokenized_scenes, dtype=torch.long)


def get_labels_concepts(
    tokenized_scenes,
    max_scene_size,
    num_concept_values_take_max,
    pad_token,
    soo_id,
    eoo_id,
    eos_id,
):
    exclude_values = torch.tensor([pad_token, soo_id, eoo_id, eos_id])
    unique_concepts = tokenized_scenes.unique()
    unique_concepts = unique_concepts[~torch.isin(unique_concepts, exclude_values)]
    concept_values_id = torch.randperm(unique_concepts.shape[0])[:num_concept_values_take_max]
    concept_values = unique_concepts[concept_values_id]
    batch = tokenized_scenes.shape[0]
    labels = torch.zeros((batch, concept_values.shape[0]))
    tokenized_labels = torch.zeros((concept_values_id.shape[0], max_scene_size), dtype=torch.long)
    tokenized_labels.fill_(pad_token)
    tokenized_labels[:, 0] = concept_values
    tokenized_labels[:, 1] = eos_id
    for scene_idx, scene in enumerate(tokenized_scenes):
        for concept_idx, concept_value in enumerate(concept_values):
            labels[scene_idx, concept_idx] = 1.0 if concept_value in scene else 0.0
    return tokenized_labels, labels


def get_labels_object(
    tokenized_scenes,
    max_scene_size,
    soo_id,
    eoo_id,
    eos_id,
    pad_token,
    num_concepts,
    num_vals_per_concept,
    num_object_values_take_max=32,
    num_build_obj_negatives_per_pos=1,
    num_swap_obj_negatives_per_scene=0,
    allowed_query_objects=None,
):
    batch = tokenized_scenes.shape[0]
    device = tokenized_scenes.device
    if allowed_query_objects is not None:
        if not torch.is_tensor(allowed_query_objects):
            allowed_query_objects = torch.tensor(
                allowed_query_objects, dtype=torch.long, device=device
            )
        else:
            allowed_query_objects = allowed_query_objects.to(device=device, dtype=torch.long)

    block_size = 1 + num_concepts + 1
    max_objects_in_seq = (max_scene_size - 1) // block_size
    concept_offsets = torch.arange(1, 1 + num_concepts, device=device)
    object_offsets = torch.arange(0, max_objects_in_seq, device=device) * block_size
    gather_indices = object_offsets.unsqueeze(1) + concept_offsets.unsqueeze(0)
    extracted_objects = tokenized_scenes[:, gather_indices.flatten()].reshape(
        batch, max_objects_in_seq, num_concepts
    )
    soo_indices = object_offsets
    soo_tokens = tokenized_scenes[:, soo_indices]
    valid_object_mask = soo_tokens == soo_id
    flat_valid_objects = extracted_objects[valid_object_mask]
    unique_positives = torch.unique(flat_valid_objects, dim=0)

    num_pos = unique_positives.shape[0]
    if num_pos > num_object_values_take_max:
        perm = torch.randperm(num_pos, device=device)[:num_object_values_take_max]
        unique_positives = unique_positives[perm]

    num_pos = unique_positives.shape[0]
    if num_pos == 0:
        return (
            torch.zeros((0, max_scene_size), dtype=torch.long, device=device),
            torch.zeros((batch, 0), device=device),
        )

    swap_negatives = None
    if num_swap_obj_negatives_per_scene > 0:
        swap_list = []
        for scene_idx in range(batch):
            obj_indices = torch.nonzero(valid_object_mask[scene_idx], as_tuple=False).squeeze(1)
            if obj_indices.numel() < 2:
                continue
            objs = extracted_objects[scene_idx, obj_indices]
            num_objs = objs.shape[0]
            for _ in range(num_swap_obj_negatives_per_scene):
                pair = torch.randperm(num_objs, device=device)[:2]
                i1 = int(pair[0].item())
                i2 = int(pair[1].item())
                c_idx = int(torch.randint(0, num_concepts, (1,), device=device).item())
                swapped = objs[i1].clone()
                swapped[c_idx] = objs[i2][c_idx]
                swap_list.append(swapped)
        if swap_list:
            swap_negatives = torch.stack(swap_list, dim=0)
            if allowed_query_objects is not None:
                matches = (swap_negatives[:, None, :] == allowed_query_objects[None, :, :]).all(
                    dim=-1
                )
                keep = matches.any(dim=1)
                swap_negatives = swap_negatives[keep]
                if swap_negatives.numel() == 0:
                    swap_negatives = None

    if allowed_query_objects is None:
        candidates_to_perturb = unique_positives.repeat(num_build_obj_negatives_per_pos, 1)
        c_indices = torch.randint(0, num_concepts, (candidates_to_perturb.shape[0],), device=device)
        row_indices = torch.arange(candidates_to_perturb.shape[0], device=device)
        current_vals = candidates_to_perturb[row_indices, c_indices]
        bases = (current_vals // num_vals_per_concept) * num_vals_per_concept
        local_vals = current_vals % num_vals_per_concept
        shifts = torch.randint(1, num_vals_per_concept, (candidates_to_perturb.shape[0],), device=device)
        new_local_vals = (local_vals + shifts) % num_vals_per_concept
        new_vals = bases + new_local_vals
        candidates_to_perturb[row_indices, c_indices] = new_vals
        unique_negatives = torch.unique(candidates_to_perturb, dim=0)
        all_queries = torch.cat([unique_positives, unique_negatives], dim=0)
    else:
        if allowed_query_objects.shape[0] == 0:
            return (
                torch.zeros((0, max_scene_size), dtype=torch.long, device=device),
                torch.zeros((batch, 0), device=device),
            )
        num_neg = num_pos * num_build_obj_negatives_per_pos
        if num_neg > 0:
            neg_indices = torch.randint(0, allowed_query_objects.shape[0], (num_neg,), device=device)
            negatives = allowed_query_objects[neg_indices]
        else:
            negatives = torch.zeros((0, num_concepts), dtype=torch.long, device=device)
        all_queries = torch.cat([unique_positives, negatives], dim=0)

    if swap_negatives is not None:
        all_queries = torch.cat([all_queries, swap_negatives], dim=0)
    all_queries = torch.unique(all_queries, dim=0)

    scene_objs_exp = extracted_objects.unsqueeze(1)
    queries_exp = all_queries.unsqueeze(0).unsqueeze(2)
    concept_matches = scene_objs_exp == queries_exp
    object_matches = concept_matches.all(dim=-1)
    mask_exp = valid_object_mask.unsqueeze(1)
    object_matches = object_matches & mask_exp
    labels = object_matches.any(dim=-1).float()

    num_queries = all_queries.shape[0]
    tokenized_queries = torch.full((num_queries, max_scene_size), pad_token, dtype=torch.long, device=device)
    tokenized_queries[:, 0] = soo_id
    tokenized_queries[:, 1 : 1 + num_concepts] = all_queries
    tokenized_queries[:, 1 + num_concepts] = eoo_id
    tokenized_queries[:, 1 + num_concepts + 1] = eos_id
    return tokenized_queries, labels


class InfiniteSceneDataset(IterableDataset):
    def __init__(
        self,
        batch_size,
        num_concepts,
        num_vals_per_concept,
        max_num_objects,
        soo_id,
        eoo_id,
        eos_id,
        pad_token,
        max_scene_size,
        probe_concepts=True,
        probe_objects=True,
        num_concept_values_take_max=64,
        num_object_values_take_max=64,
        num_build_obj_negatives_per_pos=1,
        num_swap_obj_negatives_per_scene=0,
        allowed_objects=None,
        allowed_query_objects=None,
    ):
        self.batch_size = batch_size
        self.num_concepts = num_concepts
        self.num_vals_per_concept = num_vals_per_concept
        self.max_num_objects = max_num_objects
        self.soo_id = soo_id
        self.eoo_id = eoo_id
        self.eos_id = eos_id
        self.pad_token = pad_token
        self.max_scene_size = max_scene_size
        self.probe_concepts = probe_concepts
        self.probe_objects = probe_objects
        self.allowed_objects = None
        self.allowed_query_objects_global = None
        if allowed_objects is not None:
            self.allowed_objects = np.asarray(allowed_objects, dtype=np.int64)
        if allowed_query_objects is None and self.allowed_objects is not None:
            allowed_query_objects = self.allowed_objects
        if allowed_query_objects is not None:
            allowed_query_objects = np.asarray(allowed_query_objects, dtype=np.int64)
            value_add_shift = np.arange(num_concepts, dtype=np.int64) * num_vals_per_concept
            allowed_query_objects_global = allowed_query_objects + value_add_shift[None, :]
            self.allowed_query_objects_global = torch.tensor(
                allowed_query_objects_global, dtype=torch.long
            )

        self.scene_constructor = SceneConstructor(
            num_concepts,
            num_vals_per_concept,
            max_num_objects,
            allowed_objects=self.allowed_objects,
        )
        self.num_concept_values_take_max = num_concept_values_take_max
        self.num_object_values_take_max = num_object_values_take_max
        self.num_build_obj_negatives_per_pos = num_build_obj_negatives_per_pos
        self.num_swap_obj_negatives_per_scene = num_swap_obj_negatives_per_scene

    def __iter__(self):
        while True:
            scene_objects = self.scene_constructor.get_random_scenes(
                self.batch_size, sampling_mode="random"
            )
            tokenized_scenes = tokenize_scenes(
                scene_objects,
                self.soo_id,
                self.eoo_id,
                self.eos_id,
                self.pad_token,
                self.num_concepts,
                self.num_vals_per_concept,
                self.max_scene_size,
            )
            num_objects_per_scene = torch.tensor(
                [len(scene) for scene in scene_objects], dtype=torch.long
            )

            target_tokenized, target_labels = None, None
            if self.probe_concepts:
                target_tokenized, target_labels = get_labels_concepts(
                    tokenized_scenes,
                    self.max_scene_size,
                    num_concept_values_take_max=self.num_concept_values_take_max,
                    pad_token=self.pad_token,
                    soo_id=self.soo_id,
                    eoo_id=self.eoo_id,
                    eos_id=self.eos_id,
                )

            target_tokenized_object, target_labels_object = None, None
            if self.probe_objects:
                target_tokenized_object, target_labels_object = get_labels_object(
                    tokenized_scenes,
                    self.max_scene_size,
                    soo_id=self.soo_id,
                    eoo_id=self.eoo_id,
                    eos_id=self.eos_id,
                    pad_token=self.pad_token,
                    num_concepts=self.num_concepts,
                    num_vals_per_concept=self.num_vals_per_concept,
                    num_object_values_take_max=self.num_object_values_take_max,
                    num_build_obj_negatives_per_pos=self.num_build_obj_negatives_per_pos,
                    num_swap_obj_negatives_per_scene=self.num_swap_obj_negatives_per_scene,
                    allowed_query_objects=self.allowed_query_objects_global,
                )

            yield (
                tokenized_scenes,
                target_tokenized,
                target_labels,
                target_tokenized_object,
                target_labels_object,
                num_objects_per_scene,
            )
