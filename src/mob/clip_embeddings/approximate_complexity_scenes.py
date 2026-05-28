"""Converted from 015_check_concepts_multiply.ipynb (scene-only approximation)."""

import argparse
import datetime
import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import wandb

def set_seed(seed: int) -> None:
    """Best-effort reproducibility for splits + model init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def _resolve_autosave_dir(save_dir, embeddings_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if save_dir is None:
        base_dir = Path(embeddings_path).resolve().parent / "approximate_complexity_scenes_runs"
    else:
        base_dir = Path(save_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / f"approximate_complexity_scenes_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze approximate complexity from trained embeddings."
    )
    parser.add_argument(
        "--embeddings-path",
        required=True,
        help="Path to trained embeddings .pt file (from clip_embeddings/train_embeddings.py).",
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--train-fraction",
        type=float,
        default=0.4,
        help="Fraction of combinations used for training.",
    )
    parser.add_argument(
        "--lr_use",
        type=float,
        default=1e-2,
        help="Learning rate for training.",
    )
    
    parser.add_argument(
        "--hidden-layers",
        nargs="*",
        type=int,
        default=[],
        help="Sizes of hidden layers for MLP, e.g. --hidden-layers 128 64"
    )
    parser.add_argument(
        "--mult_probes",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Use multiplicative probe (product over slot embeddings) instead of concatenation.",
    )
    parser.add_argument(
        "--use_W_mult",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, use a learnable matrix W in the multiplicative probe: (e0 @ W) * prod(e1..). If false, uses identity.",
    )
    parser.add_argument(
        "--sum_and_mult",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="When using --mult_probes, also learn separate sum embeddings and feed concat([mult, sum]) to the MLP.",
    )
    parser.add_argument(
        "--mult_within_obj",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Use multiplicative probe within objects instead of concatenating.",
    )
    
    split_group.add_argument(
        "--train-percent",
        type=float,
        help="Percent of combinations used for training (0-100).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Base directory for autosaved outputs (timestamped subdir).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible splits and initialization.",
    )
    return parser.parse_args()


_ARGS = parse_args()

set_seed(_ARGS.seed)

wandb.init(
    project="2mob-scenes-compelxity-analysis",
    config=_ARGS,
)

trained_embeddings_path = _ARGS.embeddings_path

_AUTOSAVE_DIR = _resolve_autosave_dir(_ARGS.save_dir, trained_embeddings_path)
print(f"autosave dir: {_AUTOSAVE_DIR}")
wandb.summary["autosave_dir"] = str(_AUTOSAVE_DIR)

train_split_ratio = _ARGS.train_fraction
if _ARGS.train_percent is not None:
    train_split_ratio = _ARGS.train_percent / 100.0
if not (0.0 < train_split_ratio <= 1.0):
    raise ValueError("train split ratio must be in (0, 1].")

print(f"args: {vars(_ARGS)}")

dataset_path = os.path.join(os.path.dirname(os.path.dirname(trained_embeddings_path)), "dataset.pkl")
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(os.path.dirname(trained_embeddings_path), "dataset.pkl")
dataset = pickle.load(open(dataset_path, "rb"))
if trained_embeddings_path.endswith(".pkl"):
    embeds = pickle.load(open(trained_embeddings_path, "rb"))
else:   
    embeds = torch.load(trained_embeddings_path, weights_only=False, map_location="cpu")

if not isinstance(embeds, dict) or "embeddings" not in embeds or "probes_single_concept" not in embeds or "probes_object" not in embeds:
    raise ValueError(
        "Expected a trained embeddings artifact from clip_embeddings/train_embeddings.py "
        "(dict with keys: embeddings, probes_single_concept, probes_object)."
    )

num_concepts = len(embeds["probes_single_concept"])
num_values_per_concept = [
    embeds["probes_single_concept"][f"{i}"].out_features for i in range(num_concepts)
]
num_objects = 2

print(f"num_concepts: {num_concepts}, num_values_per_concept: {num_values_per_concept}")

# Collect all scenes with two objects.
two_object_scene_ids = [
    idx for idx, scene in enumerate(dataset) if len(scene) == num_objects
]
two_object_scenes = [dataset[idx] for idx in two_object_scene_ids]
two_object_scenes_embeddings = [embeds["embeddings"][idx] for idx in two_object_scene_ids]
two_object_scenes_embeddings = torch.stack(two_object_scenes_embeddings)

single_object_scene_ids = [
    idx for idx, scene in enumerate(dataset) if len(scene) == 1
]
single_object_scenes = np.array([dataset[idx] for idx in single_object_scene_ids])
if len(single_object_scenes) != 0:
    single_object_scenes = single_object_scenes.squeeze(1)

print(f"there are {len(two_object_scenes)} scenes with two objects")

scene_embeddings_dim = two_object_scenes_embeddings[0].shape[0]

# Quickly pre-compute targets for all 2-object scenes. We will use them for calculating accuracy.
object_target_ids = []
single_tuples = [tuple(x) for x in single_object_scenes]
for scene in two_object_scenes:
    scene_objects = []
    for obj in scene:
        key = tuple([int(v) for v in obj])
        if key not in single_tuples:
            single_tuples.append(key)
        idx_j = single_tuples.index(key)
        scene_objects.append(idx_j)
    object_target_ids.append(scene_objects)
object_target_ids = np.array(object_target_ids)
print(f"object_target_ids.shape: {object_target_ids.shape}")

print(embeds["probes_single_concept"])
print(embeds["probes_object"])

two_object_scene_embeddings_mean = torch.mean(two_object_scenes_embeddings, dim=0, keepdim=True)


def get_accuracy_details(x, embeds_pred, concept_probes, object_probe, object_ids):
    """Return per-scene correctness for concepts and objects."""
    concept_correct = []
    with torch.no_grad():
        for i in range(num_concepts):
            concept_probe = concept_probes[str(i)]
            logits = concept_probe(embeds_pred)
            true_concept_idxs = x.view(x.shape[0], num_objects, num_concepts)[:, :, i]
            pos_logits = torch.gather(logits, 1, true_concept_idxs)
            min_pos = pos_logits.min(dim=-1).values

            mask_positives = torch.zeros_like(logits, dtype=torch.bool).to(device)
            mask_positives.scatter_(1, true_concept_idxs, True)
            neg_logits = logits.masked_fill(mask_positives, -float("inf"))
            max_neg = neg_logits.max(dim=-1).values

            concept_correct.append(min_pos > max_neg)

        concept_correct = torch.stack(concept_correct, dim=1)

        object_probes = object_probe["1"]
        object_logits = object_probes(embeds_pred)
        object_pos_logits = torch.gather(object_logits, 1, object_ids)

        mask_positives = torch.zeros_like(object_logits, dtype=torch.bool).to(device)
        mask_positives.scatter_(1, object_ids, True)
        object_neg_logits = object_logits.masked_fill(mask_positives, -float("inf"))
        object_max_neg = object_neg_logits.max(dim=-1).values
        object_min_pos = object_pos_logits.min(dim=-1).values
        object_correct = object_min_pos > object_max_neg

    return concept_correct, object_correct


# Device for training/evaluation tensors.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Scene-only mode.
X_all = torch.tensor(two_object_scenes).to(device)
X_all = X_all.view(X_all.shape[0], -1)
Y_all = two_object_scenes_embeddings.to(device)
num_objects_for_split = len(single_tuples)

num_train = int(num_objects_for_split * train_split_ratio)
rand_perm = torch.randperm(num_objects_for_split)
train_perm = rand_perm[:num_train]
test_perm = rand_perm[num_train:]

mask_all_train_scenes = torch.all(
    torch.stack(
        [
            torch.isin(torch.tensor(object_target_ids)[:, i], train_perm)
            for i in range(object_target_ids.shape[1])
        ]
    ),
    dim=0,
)

mask_all_test_scenes = torch.all(
    torch.stack(
        [
            torch.isin(torch.tensor(object_target_ids)[:, i], test_perm)
            for i in range(object_target_ids.shape[1])
        ]
    ),
    dim=0,
)

mask_train_test_scenes = ~(mask_all_train_scenes | mask_all_test_scenes)

assert not torch.isin(train_perm, test_perm).any(), "train/test object sets intersect"
assert not (mask_all_train_scenes & mask_all_test_scenes).any(), "train/test scene masks intersect"

object_target_ids_tensor = torch.tensor(object_target_ids)
train_scene_obj_ids = object_target_ids_tensor[mask_all_train_scenes]
test_scene_obj_ids = object_target_ids_tensor[mask_all_test_scenes]
if train_scene_obj_ids.numel() > 0:
    assert torch.isin(train_scene_obj_ids, train_perm).all(), "train scenes use test objects"
if test_scene_obj_ids.numel() > 0:
    assert torch.isin(test_scene_obj_ids, test_perm).all(), "test scenes use train objects"

X_train = X_all[mask_all_train_scenes]
Y_train = Y_all[mask_all_train_scenes]

if train_scene_obj_ids.numel() > 0:
    assert (
        torch.unique(train_scene_obj_ids).numel() <= train_perm.numel()
    ), "X_train uses more unique objects than train split"

train_scene_indices = torch.where(mask_all_train_scenes)[0]
test_scene_indices = torch.where(mask_all_test_scenes)[0]
splits_path = _AUTOSAVE_DIR / "splits.pt"
torch.save(
    {
        "train_object_ids": train_perm.cpu(),
        "test_object_ids": test_perm.cpu(),
        "train_scene_indices": train_scene_indices.cpu(),
        "test_scene_indices": test_scene_indices.cpu(),
        "mask_all_train_scenes": mask_all_train_scenes.cpu(),
        "mask_all_test_scenes": mask_all_test_scenes.cpu(),
        "mask_train_test_scenes": mask_train_test_scenes.cpu(),
    },
    splits_path,
)
wandb.summary["splits_path"] = str(splits_path)

class MLP(nn.Module):
    def __init__(
        self,
        num_concepts_expanded,
        num_vals_per_concept_expanded,
        embed_dim,
        hidden_layers=None,
        multi_probes=False,
        use_W_mult=False,
        sum_and_mult=True,
        mult_within_obj=False,
    ):
        super(MLP, self).__init__()
        if hidden_layers is None:
            hidden_layers = []
        self.num_concepts_expanded = num_concepts_expanded
        self.num_vals_per_concept_expanded = num_vals_per_concept_expanded
        self.hidden_layers = hidden_layers
        self.multi_probes = multi_probes
        self.use_W_mult = use_W_mult
        self.sum_and_mult = sum_and_mult

        if len(num_vals_per_concept_expanded) != self.num_concepts_expanded:
            raise ValueError(
                "num_vals_per_concept_expanded must match num_concepts_expanded"
            )

        net = []
        
        if not self.multi_probes:
            in_dim = embed_dim * self.num_concepts_expanded
        else:
            in_dim = embed_dim * 2 if self.sum_and_mult else embed_dim
        for hidden_layer in self.hidden_layers:
            net.append(nn.Linear(in_dim, hidden_layer))
            net.append(nn.ReLU())
            in_dim = hidden_layer
        net.append(nn.Linear(in_dim, embed_dim))
        self.net = nn.Sequential(*net)
        self.mult_within_obj = mult_within_obj
        # Optional bilinear-ish multiplicative probe:
        # mult(e0, e1, ..., eS) = (e0 @ W) * e1 * ... * eS
        # which reduces to plain elementwise product when W = I.
        if self.use_W_mult:
            self.W_multiplicative = nn.Parameter(torch.eye(embed_dim))
        else:
            self.register_buffer("W_multiplicative", torch.eye(embed_dim), persistent=False)

        # Create a learnable embedding for each concept slot.
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_vals_per_concept_expanded[i], embed_dim)
                for i in range(self.num_concepts_expanded)
            ]
        )

        # Separate embeddings for the additive (sum) probe.
        if self.multi_probes and self.sum_and_mult:
            self.sum_embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_vals_per_concept_expanded[i], embed_dim)
                    for i in range(self.num_concepts_expanded)
                ]
            )

    def forward(self, x):
        # x is (batch_size, self.num_concepts_expanded) with scene concepts flattened.
        if not self.multi_probes:
            embeddings = torch.cat(
                [self.embeddings[i](x[:, i]) for i in range(self.num_concepts_expanded)],
                dim=1,
            )
        else:
            stacked = torch.stack(
                [self.embeddings[i](x[:, i]) for i in range(self.num_concepts_expanded)],
                dim=1,
            )  # [B, S, D]
            if self.mult_within_obj:
                obj1_embeddings = stacked[:, :self.num_concepts_expanded // 2, :].prod(dim=1)
                obj2_embeddings = stacked[:, self.num_concepts_expanded // 2:, :].prod(dim=1)
                mult = obj1_embeddings + obj2_embeddings
            else:
                e0 = stacked[:, 0, :] @ self.W_multiplicative
                if stacked.shape[1] == 1:
                    mult = e0
                else:
                    mult = e0 * stacked[:, 1:, :].prod(dim=1)

            if self.sum_and_mult:
                sum_emb = torch.stack(
                    [self.sum_embeddings[i](x[:, i]) for i in range(self.num_concepts_expanded)],
                    dim=1,
                ).sum(dim=1)
                embeddings = torch.cat([mult, sum_emb], dim=1)
            else:
                embeddings = mult
        return self.net(embeddings)

num_concepts_for_model = num_objects * num_concepts
num_vals_per_concept_for_model = num_values_per_concept * num_objects

model = MLP(
    num_concepts_for_model,
    num_vals_per_concept_for_model,
    scene_embeddings_dim,
    hidden_layers=_ARGS.hidden_layers,
    multi_probes=_ARGS.mult_probes,
    use_W_mult=_ARGS.use_W_mult,
    sum_and_mult=_ARGS.sum_and_mult,
    mult_within_obj=_ARGS.mult_within_obj,
).to(device)

object_target_ids = torch.tensor(object_target_ids).to(device)
two_object_scenes_tensor = torch.tensor(two_object_scenes).to(device)
two_object_scene_embeddings_tensor = two_object_scenes_embeddings.to(device)
two_object_scene_embeddings_mean_tensor = two_object_scene_embeddings_mean.to(device)

optim = torch.optim.Adam(model.parameters(), lr=_ARGS.lr_use)

num_epochs = 4000

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)

# Report the number of train/test/intersection scenes and log to wandb
num_all_scenes = len(two_object_scenes)
num_train_only_scenes = mask_all_train_scenes.sum().item()
num_test_only_scenes = mask_all_test_scenes.sum().item()
num_train_test_scenes = mask_train_test_scenes.sum().item()

print(
    f"num_all_scenes: {num_all_scenes}, "
    f"num_train_only_scenes: {num_train_only_scenes}, "
    f"num_test_only_scenes: {num_test_only_scenes}, "
    f"num_train_test_scenes: {num_train_test_scenes}"
)

wandb.summary["num_all_scenes"] = num_all_scenes
wandb.summary["num_train_only_scenes"] = num_train_only_scenes
wandb.summary["num_test_only_scenes"] = num_test_only_scenes
wandb.summary["num_train_test_scenes"] = num_train_test_scenes

best_test_object_acc = float("-inf")
best_epoch = None
best_model_path = _AUTOSAVE_DIR / "mlp_best.pt"

for i in (pbar := tqdm(range(num_epochs))):
    model.train()
    optim.zero_grad()
    pred = model(X_train)
    loss = nn.MSELoss()(pred, Y_train)
    loss.backward()
    optim.step()

    if i % 100 == 0:
        with torch.no_grad():
            pred = model(X_all)
            mse_all_data = torch.mean((Y_all - pred) ** 2).item()
            embeds_pred = pred

            concept_correct, object_correct = get_accuracy_details(
                two_object_scenes_tensor,
                embeds_pred,
                embeds["probes_single_concept"].to(device),
                embeds["probes_object"].to(device),
                object_target_ids,
            )

            acc_all_data = object_correct.float().mean().item()

            def summarize_mask(mask):
                if mask.sum().item() == 0:
                    return float("nan"), float("nan")
                mask = mask.to(concept_correct.device)
                concept_acc = concept_correct[mask].float().mean().item()
                object_acc = object_correct[mask].float().mean().item()
                return concept_acc, object_acc

            def summarize_mse_r2(mask):
                if mask.sum().item() == 0:
                    return float("nan"), float("nan")
                mask = mask.to(two_object_scene_embeddings_tensor.device)
                y_true = two_object_scene_embeddings_tensor[mask]
                y_pred = embeds_pred[mask]
                mse = torch.mean((y_true - y_pred) ** 2).item()
                sse = torch.sum((y_true - y_pred) ** 2)
                y_true_centered = y_true - two_object_scene_embeddings_mean_tensor
                sst = torch.sum(y_true_centered ** 2)
                r2 = (1 - sse / sst).item() if sst > 0 else float("nan")
                return mse, r2

            acc_org_concept, acc_org_object = summarize_mask(
                torch.ones_like(mask_all_train_scenes, dtype=torch.bool)
            )
            mse_all, r2_all = summarize_mse_r2(
                torch.ones_like(mask_all_train_scenes, dtype=torch.bool)
            )
            
            stuff_report = {
                "train_loss": loss.item(),
                "epoch": i,
                "lr": scheduler.get_last_lr()[0],
                "train_all_scenes_concept_acc": acc_org_concept,
                "train_all_scenes_object_acc": acc_org_object,
                "train_all_scenes_mse": mse_all,
                "train_all_scenes_r2": r2_all,
                "mse_all_data": mse_all_data,
                "acc_all_data": acc_all_data,
            }

            print(
                "all scenes: "
                f"acc_org_concept: {acc_org_concept}, "
                f"acc_org_object: {acc_org_object}, "
                f"mse: {mse_all}, "
                f"r2: {r2_all}"
            )
            
            for mask, name in zip([mask_all_train_scenes, mask_all_test_scenes, mask_train_test_scenes], ["train_only_scenes", "test_only_scenes", "train_test_scenes"]):
                    
                acc_org_concept, acc_org_object = summarize_mask(mask)
                mse_mask, r2_mask = summarize_mse_r2(mask)

                stuff_report[f"{name}_concept_acc"] = acc_org_concept
                stuff_report[f"{name}_object_acc"] = acc_org_object
                stuff_report[f"{name}_mse"] = mse_mask
                stuff_report[f"{name}_r2"] = r2_mask

                # Track best test-only object accuracy and checkpoint.
                if name == "test_only_scenes":
                    if acc_org_object > best_test_object_acc:
                        best_test_object_acc = acc_org_object
                        best_epoch = i
                        print(
                            f"new best test object acc: {acc_org_object} at epoch {i}, saving."
                        )

                        model_state = {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        }
                        torch.save(
                            {
                                "state_dict": model_state,
                                "args": vars(_ARGS),
                                "epoch": i,
                                "best_test_only_object_acc": best_test_object_acc,
                                "num_concepts_for_model": num_concepts_for_model,
                                "num_vals_per_concept_for_model": num_vals_per_concept_for_model,
                                "seed": _ARGS.seed,
                            },
                            best_model_path,
                        )
                        wandb.summary["best_model_path"] = str(best_model_path)
                        wandb.summary["best_epoch"] = best_epoch

                    stuff_report["Best so far"] = best_test_object_acc

                print(
                    f"\t name: {name}, "
                    f"acc_org_concept: {acc_org_concept}, "
                    f"acc_org_object: {acc_org_object}, "
                    f"mse: {mse_mask}, "
                    f"r2: {r2_mask}"
                )
        wandb.log(stuff_report, step=i)

    pbar.set_postfix(loss=loss.item())

model_path = _AUTOSAVE_DIR / "mlp.pt"
model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
torch.save(
    {
        "state_dict": model_state,
        "args": vars(_ARGS),
        "num_concepts_for_model": num_concepts_for_model,
        "num_vals_per_concept_for_model": num_vals_per_concept_for_model,
    },
    model_path,
)
wandb.summary["model_path"] = str(model_path)
print(f"saved model: {model_path}")
