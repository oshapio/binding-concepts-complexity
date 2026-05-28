import argparse
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models import SceneEncoder
from scenes import InfiniteSceneDataset


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def make_linear_warmup_decay(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)


def object_ids_to_values(object_ids, num_concepts, num_vals_per_concept):
    object_ids = np.asarray(object_ids, dtype=np.int64)
    base = num_vals_per_concept ** np.arange(num_concepts, dtype=np.int64)
    values = (object_ids[:, None] // base[None, :]) % num_vals_per_concept
    return values.astype(np.int64)


def compute_correct_by_scene(dot_prod_embeddings, labels, num_objects_per_scene):
    is_active = labels.sum(dim=1) > 0
    if not is_active.any():
        return None, None
    active_embeds = dot_prod_embeddings[is_active]
    active_labels = labels[is_active].long()
    mask = active_labels == 1
    masked_embeds = active_embeds.masked_fill(~mask, float("inf"))
    min_correct_logits = masked_embeds.min(dim=1).values
    masked_embeds_incorrect = active_embeds.masked_fill(mask, float("-inf"))
    max_incorrect_logits = masked_embeds_incorrect.max(dim=1).values
    correct = (min_correct_logits > max_incorrect_logits).float()
    return correct, num_objects_per_scene[is_active]


def summarize_accuracy_by_num_objects(correct, active_num_objects):
    by_count = {}
    if correct is None:
        return by_count
    for count in torch.unique(active_num_objects):
        count_mask = active_num_objects == count
        if count_mask.any():
            by_count[int(count.item())] = correct[count_mask].mean().item()
    return by_count


def mean_active_soft_ce(ce_loss, logits, labels):
    labels_sum = labels.sum(dim=1, keepdim=True)
    labels_forward = labels / (labels_sum + 1e-10)
    is_active = labels_sum.squeeze(1) > 0
    per_row_loss = ce_loss(logits, labels_forward)
    if is_active.any():
        return per_row_loss[is_active].mean(), is_active
    return per_row_loss.new_zeros(()), is_active


def make_loader(dataset, num_workers, prefetch_factor):
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(dataset, batch_size=None, num_workers=0)


def amortizer(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim_type = params["sim_type"]
    epochs = params["epochs"]
    d_out_size = params["model_d_out"]
    d_model = params["model_d_model"]
    num_heads = params["model_num_heads"]
    num_layers = params["model_num_layers"]
    d_out_size_probe = d_out_size + (1 if sim_type == "dot" else 0)
    lr = params["model_lr"]
    batch_size = params["train_batch_size"]
    num_grad_steps = params["train_num_grad_steps"]
    test_every_steps = params["test_every_steps"]
    test_num_batches = params["test_num_batches"]
    test_batch_size = params["test_batch_size"]
    use_scheduler = params["use_scheduler"]
    weight_decay = params["weight_decay"]
    num_concepts = params["num_concepts"]
    num_vals_per_concept = params["num_vals_per_concept"]
    probe_concepts = params["probe_concepts"]
    probe_objects = params["probe_objects"]
    max_num_objects = params["max_num_objects"]
    train_object_fraction = float(params["train_object_fraction"])
    object_split_seed = params["object_split_seed"]
    save_model = params["save_model"]
    save_best_object_model = params["save_best_object_model"]
    num_concept_values_take_max = params["num_concept_values_take_max"]
    num_object_values_take_max = params["num_object_values_take_max"]
    num_build_obj_negatives_per_pos = params["num_negative_objs_per_pos"]
    num_swap_obj_negatives_per_scene = params["num_swap_obj_negatives_per_scene"]
    train_probe_all_objects = params["train_probe_all_objects"]
    use_cliplike_text_encoder = params["use_cliplike_text_encoder"]
    use_wandb = params["use_wandb"]
    working_dir = os.path.expanduser(params["working_dir"])
    train_num_workers = params["train_num_workers"]
    test_num_workers = params["test_num_workers"]
    prefetch_factor = params["prefetch_factor"]

    if train_object_fraction <= 0.0 or train_object_fraction > 1.0:
        raise ValueError("train_object_fraction must be in (0.0, 1.0].")

    if use_wandb:
        wandb.init(
            project="mob-amortization-tests",
            name=f"{num_concepts}c_{num_vals_per_concept}v_{max_num_objects}o_{d_out_size}_{sim_type}",
            config=params,
        )
    else:
        wandb.init(mode="disabled")

    num_concept_tokens = num_concepts * num_vals_per_concept
    soo_id = num_concept_tokens
    eoo_id = soo_id + 1
    eos_id = eoo_id + 1
    pad_token = eos_id + 1
    vocab_size = pad_token + 1
    max_scene_size = (num_concepts + 2) * max_num_objects + 1
    save_dir = os.path.join(working_dir, "models")
    if save_model:
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model.pt")
    best_object_save_path = os.path.join(save_dir, "model_best_test_objects.pt")

    total_objects = num_vals_per_concept ** num_concepts
    train_object_count = total_objects
    test_object_count = 0
    train_objects = None
    test_objects = None
    train_object_ids = None
    test_object_ids = None
    object_split_path = None
    train_query_objects = None

    if train_object_fraction < 1.0:
        rng = np.random.RandomState(object_split_seed)
        object_ids = rng.permutation(total_objects)
        train_count = max(1, int(total_objects * train_object_fraction))
        if total_objects > 1:
            train_count = min(train_count, total_objects - 1)
        train_object_ids = object_ids[:train_count]
        test_object_ids = object_ids[train_count:]
        train_objects = object_ids_to_values(
            train_object_ids, num_concepts, num_vals_per_concept
        )
        test_objects = object_ids_to_values(
            test_object_ids, num_concepts, num_vals_per_concept
        )
        train_object_count = train_count
        test_object_count = total_objects - train_count
        object_split_dir = os.path.join(working_dir, "splits")
        os.makedirs(object_split_dir, exist_ok=True)
        object_split_path = os.path.join(
            object_split_dir,
            f"object_split_seed{object_split_seed}_frac{train_object_fraction:.4f}.npz",
        )
        np.savez_compressed(
            object_split_path,
            train_object_ids=train_object_ids,
            test_object_ids=test_object_ids,
            train_objects=train_objects,
            test_objects=test_objects,
            num_concepts=num_concepts,
            num_vals_per_concept=num_vals_per_concept,
            total_objects=total_objects,
            train_object_fraction=train_object_fraction,
            object_split_seed=object_split_seed,
        )
        print(
            f"Object split: train {train_object_count}/{total_objects}, "
            f"test {test_object_count}"
        )

    if train_probe_all_objects and train_object_fraction < 1.0:
        all_object_ids = np.arange(total_objects, dtype=np.int64)
        train_query_objects = object_ids_to_values(
            all_object_ids, num_concepts, num_vals_per_concept
        )
        print(f"Training probes use all objects: {train_query_objects.shape[0]}")

    scene_encoder = SceneEncoder(
        d_model=d_model,
        d_out_size=d_out_size,
        num_heads=num_heads,
        num_layers=num_layers,
        pad_token=pad_token,
        vocab_size=vocab_size,
        max_scene_size=max_scene_size,
        use_cliplike_text_encoder=use_cliplike_text_encoder,
    ).to(device)
    probe_encoder = SceneEncoder(
        d_model=d_model,
        d_out_size=d_out_size_probe,
        num_heads=num_heads,
        num_layers=num_layers,
        pad_token=pad_token,
        vocab_size=vocab_size,
        max_scene_size=max_scene_size,
        use_cliplike_text_encoder=use_cliplike_text_encoder,
    ).to(device)

    print(f"Scene Encoder Params: {sum(p.numel() for p in scene_encoder.parameters())/1e6:.2f}M")
    print(f"Probe Encoder Params: {sum(p.numel() for p in probe_encoder.parameters())/1e6:.2f}M")

    wandb.log(
        {
            "scene_params_millions": sum(p.numel() for p in scene_encoder.parameters()) / 1e6,
            "probe_params_millions": sum(p.numel() for p in probe_encoder.parameters()) / 1e6,
        }
    )

    scene_encoder.train()
    probe_encoder.train()

    tau = torch.tensor(1.15, requires_grad=True, device=device)
    optimizer = torch.optim.AdamW(
        list(scene_encoder.parameters()) + list(probe_encoder.parameters()) + [tau],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = make_linear_warmup_decay(optimizer, int(0.1 * epochs), epochs)
    ce_loss = torch.nn.CrossEntropyLoss(reduction="none")

    dataset = InfiniteSceneDataset(
        batch_size=batch_size,
        num_concepts=num_concepts,
        num_vals_per_concept=num_vals_per_concept,
        max_num_objects=max_num_objects,
        soo_id=soo_id,
        eoo_id=eoo_id,
        eos_id=eos_id,
        pad_token=pad_token,
        max_scene_size=max_scene_size,
        probe_concepts=probe_concepts,
        probe_objects=probe_objects,
        num_concept_values_take_max=num_concept_values_take_max,
        num_object_values_take_max=num_object_values_take_max,
        num_build_obj_negatives_per_pos=num_build_obj_negatives_per_pos,
        num_swap_obj_negatives_per_scene=num_swap_obj_negatives_per_scene,
        allowed_objects=train_objects,
        allowed_query_objects=train_query_objects,
    )
    dataloader = make_loader(dataset, train_num_workers, prefetch_factor)
    data_iter = iter(dataloader)

    test_data_iter = None
    if train_object_fraction < 1.0 and test_object_count > 0 and test_every_steps > 0:
        test_query_objects = None
        if train_objects is not None and test_objects is not None:
            test_query_objects = np.concatenate([train_objects, test_objects], axis=0)
        test_dataset = InfiniteSceneDataset(
            batch_size=test_batch_size,
            num_concepts=num_concepts,
            num_vals_per_concept=num_vals_per_concept,
            max_num_objects=max_num_objects,
            soo_id=soo_id,
            eoo_id=eoo_id,
            eos_id=eos_id,
            pad_token=pad_token,
            max_scene_size=max_scene_size,
            probe_concepts=probe_concepts,
            probe_objects=probe_objects,
            num_concept_values_take_max=num_concept_values_take_max,
            num_object_values_take_max=num_object_values_take_max,
            num_build_obj_negatives_per_pos=num_build_obj_negatives_per_pos,
            num_swap_obj_negatives_per_scene=num_swap_obj_negatives_per_scene,
            allowed_objects=test_objects,
            allowed_query_objects=test_query_objects,
        )
        test_dataloader = make_loader(test_dataset, test_num_workers, prefetch_factor)
        test_data_iter = iter(test_dataloader)

    model_config = {
        "d_model": d_model,
        "d_out_size": d_out_size,
        "d_out_size_probe": d_out_size_probe,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "max_scene_size": max_scene_size,
        "pad_token": pad_token,
        "soo_id": soo_id,
        "eoo_id": eoo_id,
        "eos_id": eos_id,
        "num_concepts": num_concepts,
        "num_vals_per_concept": num_vals_per_concept,
        "max_num_objects": max_num_objects,
        "total_objects": total_objects,
        "train_object_fraction": train_object_fraction,
        "object_split_seed": object_split_seed,
        "train_object_count": train_object_count,
        "test_object_count": test_object_count,
        "object_split_path": object_split_path,
        "train_object_ids": train_object_ids,
        "test_object_ids": test_object_ids,
        "sim_type": sim_type,
        "params": params,
    }

    best_test_objects_accuracy = float("-inf")
    best_test_objects_step = -1

    def save_checkpoint(step, is_final=False, checkpoint_path=None, tag=None, metrics=None):
        out_path = checkpoint_path or save_path
        torch.save(
            {
                "scene_encoder": scene_encoder.state_dict(),
                "probe_encoder": probe_encoder.state_dict(),
                "tau": tau.detach().cpu(),
                "global_step": step,
                "config": model_config,
                "metrics": metrics,
            },
            out_path,
        )
        if is_final:
            print({"checkpoint_saved": out_path})
        elif tag is not None:
            print({"checkpoint_saved": out_path, "tag": tag, "step": step})

    save_every_steps = max(1, int(0.05 * max(1, epochs)))

    pbar = tqdm(range(epochs))
    for i in pbar:
        try:
            (
                tokenized_scenes,
                target_tokenized,
                target_labels,
                target_tokenized_object,
                target_labels_object,
                num_objects_per_scene,
            ) = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            (
                tokenized_scenes,
                target_tokenized,
                target_labels,
                target_tokenized_object,
                target_labels_object,
                num_objects_per_scene,
            ) = next(data_iter)

        tokenized_scenes = tokenized_scenes.to(device)
        num_objects_per_scene = num_objects_per_scene.to(device)

        tasks_data = []
        if target_tokenized is not None:
            tasks_data.append((target_tokenized.to(device), target_labels.to(device), "concepts"))
        if target_tokenized_object is not None:
            tasks_data.append(
                (target_tokenized_object.to(device), target_labels_object.to(device), "objects")
            )

        for grad_step in range(num_grad_steps):
            loss = 0.0
            embedded_scenes = scene_encoder(tokenized_scenes)
            embedding_scenes_normed = embedded_scenes / (
                1e-6 + torch.norm(embedded_scenes, dim=1, keepdim=True)
            )

            for targets, labels, task_type in tasks_data:
                target_scenes = probe_encoder(targets)
                if sim_type == "dot":
                    target_scenes, target_bias = target_scenes[:, :-1], target_scenes[:, -1:]
                target_scenes_normed = target_scenes / (
                    1e-6 + torch.norm(target_scenes, dim=1, keepdim=True)
                )
                temp_exp = torch.exp(tau).clamp(max=10.0)

                if sim_type == "dot":
                    dot_prod_embeddings = embedded_scenes @ target_scenes.T
                    dot_prod_embeddings = dot_prod_embeddings + target_bias.T
                else:
                    dot_prod_embeddings = temp_exp * (
                        embedding_scenes_normed @ target_scenes_normed.T
                    )

                loss_forward, _ = mean_active_soft_ce(ce_loss, dot_prod_embeddings, labels)
                loss = loss + loss_forward

                if i % 100 == 0 and (grad_step == 0 or grad_step == num_grad_steps - 1):
                    with torch.no_grad():
                        correct, active_num_objects = compute_correct_by_scene(
                            dot_prod_embeddings, labels, num_objects_per_scene
                        )
                        if correct is None:
                            retrieval_accuracy = torch.tensor(0.0, device=device)
                            acc_by_nobj = {}
                        else:
                            retrieval_accuracy = correct.mean()
                            acc_by_nobj = summarize_accuracy_by_num_objects(
                                correct, active_num_objects
                            )
                    print(
                        f"\t{task_type} accuracy: {retrieval_accuracy:.3f}, "
                        f"loss: {loss_forward:.3f}"
                    )
                    log_payload = {
                        f"{task_type}_accuracy_{grad_step}": retrieval_accuracy.item(),
                        f"{task_type}_loss_{grad_step}": loss_forward.item(),
                    }
                    for count, acc in acc_by_nobj.items():
                        log_payload[f"train_{task_type}_acc_nobj_{count}"] = acc
                    if grad_step == 0:
                        log_payload["loss"] = float(loss.item())
                        log_payload["tau"] = float(tau.item())
                        log_payload["lr_record"] = scheduler.get_last_lr()[0]
                        log_payload["epoch"] = i
                        print(
                            f"epoch: {i}, loss: {loss.item():.3f}, "
                            f"tau: {tau.item():.3f}, lr: {scheduler.get_last_lr()[0]:.3e}"
                        )
                    wandb.log(log_payload, step=i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if use_scheduler:
            scheduler.step()

        if test_data_iter is not None and test_every_steps > 0 and i % test_every_steps == 0:
            scene_encoder.eval()
            probe_encoder.eval()
            test_loss_sum = {}
            test_acc_sum = {}
            test_count = {}
            test_acc_sum_by_nobj = {}
            test_count_by_nobj = {}
            with torch.no_grad():
                for _ in range(test_num_batches):
                    (
                        tokenized_scenes,
                        target_tokenized,
                        target_labels,
                        target_tokenized_object,
                        target_labels_object,
                        num_objects_per_scene,
                    ) = next(test_data_iter)
                    tokenized_scenes = tokenized_scenes.to(device)
                    num_objects_per_scene = num_objects_per_scene.to(device)

                    tasks_data = []
                    if target_tokenized is not None:
                        tasks_data.append(
                            (target_tokenized.to(device), target_labels.to(device), "concepts")
                        )
                    if target_tokenized_object is not None:
                        tasks_data.append(
                            (
                                target_tokenized_object.to(device),
                                target_labels_object.to(device),
                                "objects",
                            )
                        )

                    embedded_scenes = scene_encoder(tokenized_scenes)
                    embedding_scenes_normed = embedded_scenes / (
                        1e-6 + torch.norm(embedded_scenes, dim=1, keepdim=True)
                    )

                    for targets, labels, task_type in tasks_data:
                        target_scenes = probe_encoder(targets)
                        if sim_type == "dot":
                            target_scenes, target_bias = target_scenes[:, :-1], target_scenes[:, -1:]
                        target_scenes_normed = target_scenes / (
                            1e-6 + torch.norm(target_scenes, dim=1, keepdim=True)
                        )
                        temp_exp = torch.exp(tau).clamp(max=100.0)

                        if sim_type == "dot":
                            dot_prod_embeddings = embedded_scenes @ target_scenes.T
                            dot_prod_embeddings = dot_prod_embeddings + target_bias.T
                        else:
                            dot_prod_embeddings = temp_exp * (
                                embedding_scenes_normed @ target_scenes_normed.T
                            )

                        loss_forward, _ = mean_active_soft_ce(ce_loss, dot_prod_embeddings, labels)
                        correct, active_num_objects = compute_correct_by_scene(
                            dot_prod_embeddings, labels, num_objects_per_scene
                        )
                        retrieval_accuracy = (
                            torch.tensor(0.0, device=device)
                            if correct is None
                            else correct.mean()
                        )

                        test_loss_sum.setdefault(task_type, 0.0)
                        test_acc_sum.setdefault(task_type, 0.0)
                        test_count.setdefault(task_type, 0)
                        test_acc_sum_by_nobj.setdefault(task_type, {})
                        test_count_by_nobj.setdefault(task_type, {})

                        test_loss_sum[task_type] += loss_forward.item()
                        test_acc_sum[task_type] += retrieval_accuracy.item()
                        test_count[task_type] += 1

                        if correct is not None:
                            for count in torch.unique(active_num_objects):
                                count_mask = active_num_objects == count
                                if count_mask.any():
                                    count_int = int(count.item())
                                    test_acc_sum_by_nobj[task_type].setdefault(count_int, 0.0)
                                    test_count_by_nobj[task_type].setdefault(count_int, 0)
                                    test_acc_sum_by_nobj[task_type][count_int] += correct[
                                        count_mask
                                    ].sum().item()
                                    test_count_by_nobj[task_type][count_int] += int(
                                        count_mask.sum().item()
                                    )

            for task_type in test_loss_sum:
                avg_loss = test_loss_sum[task_type] / max(1, test_count[task_type])
                avg_acc = test_acc_sum[task_type] / max(1, test_count[task_type])
                print(f"\ttest {task_type} accuracy: {avg_acc:.3f}, loss: {avg_loss:.3f}")
                wandb.log(
                    {
                        f"test_{task_type}_accuracy": avg_acc,
                        f"test_{task_type}_loss": avg_loss,
                    },
                    step=i,
                )
                if task_type == "objects" and avg_acc > best_test_objects_accuracy:
                    best_test_objects_accuracy = avg_acc
                    best_test_objects_step = i
                    print(
                        f"\tnew best test objects accuracy: "
                        f"{best_test_objects_accuracy:.4f} at step {best_test_objects_step}"
                    )
                    wandb.log(
                        {
                            "test_objects_accuracy_best_so_far": best_test_objects_accuracy,
                            "test_objects_accuracy_best_step": best_test_objects_step,
                        },
                        step=i,
                    )
                    if save_model and save_best_object_model:
                        save_checkpoint(
                            i,
                            checkpoint_path=best_object_save_path,
                            tag="best-test-objects",
                            metrics={
                                "test_objects_accuracy_best": best_test_objects_accuracy,
                                "test_objects_accuracy_best_step": best_test_objects_step,
                            },
                        )

                if test_acc_sum_by_nobj.get(task_type):
                    acc_by_nobj = {
                        count: test_acc_sum_by_nobj[task_type][count]
                        / max(1, test_count_by_nobj[task_type][count])
                        for count in sorted(test_acc_sum_by_nobj[task_type].keys())
                    }
                    wandb.log(
                        {
                            f"test_{task_type}_acc_nobj_{count}": acc
                            for count, acc in acc_by_nobj.items()
                        },
                        step=i,
                    )

            scene_encoder.train()
            probe_encoder.train()

        if save_model and i > 0 and i % save_every_steps == 0:
            save_checkpoint(i)

    if best_test_objects_step >= 0:
        print(
            f"Best test objects accuracy: {best_test_objects_accuracy:.4f} "
            f"at step {best_test_objects_step}"
        )

    if save_model:
        save_checkpoint(epochs, is_final=True)
    else:
        print("Not saving model")

    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Amortized clean trainer (scene/object probes).")
    parser.add_argument("--sim-type", choices=["dot", "cos"], default="cos")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--model-d-out", type=int, default=3)
    parser.add_argument("--model-d-model", type=int, default=256)
    parser.add_argument("--model-num-heads", type=int, default=4)
    parser.add_argument("--model-num-layers", type=int, default=6)
    parser.add_argument("--model-lr", type=float, default=3e-5)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--train-num-grad-steps", type=int, default=1)
    parser.add_argument("--test-every-steps", type=int, default=0)
    parser.add_argument("--test-num-batches", type=int, default=10)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--use-scheduler", type=_parse_bool, default=True)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-concepts", type=int, default=2)
    parser.add_argument("--num-vals-per-concept", type=int, default=10)
    parser.add_argument("--probe-concepts", type=_parse_bool, default=True)
    parser.add_argument("--probe-objects", type=_parse_bool, default=False)
    parser.add_argument("--max-num-objects", type=int, default=1)
    parser.add_argument("--train-object-fraction", type=float, default=1.0)
    parser.add_argument("--object-split-seed", type=int, default=0)
    parser.add_argument("--save-model", type=_parse_bool, default=False)
    parser.add_argument("--save-best-object-model", type=_parse_bool, default=True)
    parser.add_argument("--num-concept-values-take-max", type=int, default=64)
    parser.add_argument("--num-object-values-take-max", type=int, default=64)
    parser.add_argument("--num-negative-objs-per-pos", type=int, default=0)
    parser.add_argument("--num-swap-obj-negatives-per-scene", type=int, default=1)
    parser.add_argument("--train-probe-all-objects", type=_parse_bool, default=True)
    parser.add_argument("--use-cliplike-text-encoder", type=_parse_bool, default=False)
    parser.add_argument("--use-wandb", type=_parse_bool, default=False)
    parser.add_argument("--working-dir", type=str, default="~/tmp/amortization_tests")
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--test-num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    params = {
        "sim_type": args.sim_type,
        "epochs": args.epochs,
        "model_d_out": args.model_d_out,
        "model_d_model": args.model_d_model,
        "model_num_heads": args.model_num_heads,
        "model_num_layers": args.model_num_layers,
        "model_lr": args.model_lr,
        "train_batch_size": args.train_batch_size,
        "train_num_grad_steps": args.train_num_grad_steps,
        "test_every_steps": args.test_every_steps,
        "test_num_batches": args.test_num_batches,
        "test_batch_size": args.test_batch_size,
        "use_scheduler": args.use_scheduler,
        "weight_decay": args.weight_decay,
        "num_concepts": args.num_concepts,
        "num_vals_per_concept": args.num_vals_per_concept,
        "probe_concepts": args.probe_concepts,
        "probe_objects": args.probe_objects,
        "max_num_objects": args.max_num_objects,
        "train_object_fraction": args.train_object_fraction,
        "object_split_seed": args.object_split_seed,
        "save_model": args.save_model,
        "save_best_object_model": args.save_best_object_model,
        "num_concept_values_take_max": args.num_concept_values_take_max,
        "num_object_values_take_max": args.num_object_values_take_max,
        "num_negative_objs_per_pos": args.num_negative_objs_per_pos,
        "num_swap_obj_negatives_per_scene": args.num_swap_obj_negatives_per_scene,
        "train_probe_all_objects": args.train_probe_all_objects,
        "use_cliplike_text_encoder": args.use_cliplike_text_encoder,
        "use_wandb": args.use_wandb,
        "working_dir": args.working_dir,
        "train_num_workers": args.train_num_workers,
        "test_num_workers": args.test_num_workers,
        "prefetch_factor": args.prefetch_factor,
    }
    amortizer(params)


if __name__ == "__main__":
    main()
