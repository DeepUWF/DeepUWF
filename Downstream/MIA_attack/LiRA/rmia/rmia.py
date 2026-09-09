import gc
import math
import os
import sys

import numpy as np
import torch

from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
)

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import trunc_normal_


# ============================================================
# Project imports
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# rmia.py:
# DeepUWF/MIA_attack/rmia/rmia.py
#
# We need:
# DeepUWF
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        CURRENT_DIR
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )

from Model import models_vit
from Model.util.pos_embed import interpolate_pos_embed
import Model.util.lr_decay as lrd


# ============================================================
# Dataset wrapper
# ============================================================

class XYDataset(Dataset):
    """
    Expose only (image, label).

    Supports datasets returning either:
        image, label
    or:
        image, label, path
    """

    def __init__(
        self,
        dataset,
    ):
        self.dataset = dataset

    def __len__(
        self,
    ):
        return len(
            self.dataset
        )

    def __getitem__(
        self,
        index,
    ):
        sample = self.dataset[
            index
        ]

        if (
            not isinstance(
                sample,
                (tuple, list),
            )
            or len(sample) < 2
        ):
            raise ValueError(
                "Dataset sample must contain "
                "at least (image, label)."
            )

        return (
            sample[0],
            int(sample[1]),
        )


# ============================================================
# DataLoader helper
# ============================================================

def _make_loader(
    dataset,
    batch_size,
    shuffle,
    device,
    num_workers=2,
):
    if len(dataset) == 0:
        raise ValueError(
            "Cannot build a DataLoader "
            "for an empty dataset."
        )

    return DataLoader(
        dataset,
        batch_size=min(
            int(batch_size),
            len(dataset),
        ),
        shuffle=shuffle,
        num_workers=int(
            num_workers
        ),
        pin_memory=(
            torch.device(device).type
            == "cuda"
        ),
        drop_last=False,
        persistent_workers=(
            int(num_workers) > 0
        ),
    )


# ============================================================
# Build architecture-matched shadow ViT
# ============================================================

def build_shadow_model(
    pretrained_path,
    device,
    model_name="vit_large_patch16",
    img_size=448,
    num_classes=2,
    drop_path_rate=0.1,
    global_pool=True,
):
    """
    Build a shadow model matching the victim architecture.

    Initialization:
        same pre-finetuning pretrained checkpoint

    Important:
        pretrained_path should NOT be the final victim
        fine-tuned checkpoint.
    """

    model = models_vit.__dict__[
        model_name
    ](
        img_size=img_size,
        num_classes=num_classes,
        drop_path_rate=(
            drop_path_rate
        ),
        global_pool=global_pool,
    )

    checkpoint = torch.load(
        pretrained_path,
        map_location="cpu",
    )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        checkpoint_model = checkpoint[
            "model"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict"
        in checkpoint
    ):
        checkpoint_model = checkpoint[
            "model_state_dict"
        ]

    elif isinstance(
        checkpoint,
        dict,
    ):
        checkpoint_model = checkpoint

    else:
        raise ValueError(
            "Unsupported pretrained "
            "checkpoint format."
        )

    checkpoint_model = (
        checkpoint_model.copy()
    )

    state_dict = model.state_dict()

    # --------------------------------------------------------
    # Match victim fine-tuning behavior:
    # remove incompatible classification head only.
    # --------------------------------------------------------

    for key in (
        "head.weight",
        "head.bias",
    ):
        if (
            key in checkpoint_model
            and (
                key not in state_dict
                or checkpoint_model[
                    key
                ].shape
                != state_dict[
                    key
                ].shape
            )
        ):
            print(
                "Removing incompatible "
                f"pretrained parameter: {key}",
                flush=True,
            )

            del checkpoint_model[
                key
            ]

    # Required when pretrained and current
    # image resolutions differ.
    interpolate_pos_embed(
        model,
        checkpoint_model,
    )

    # Match your victim fine-tuning initialization.
    trunc_normal_(
        model.head.weight,
        std=2e-5,
    )

    if (
        model.head.bias
        is not None
    ):
        torch.nn.init.zeros_(
            model.head.bias
        )

    msg = model.load_state_dict(
        checkpoint_model,
        strict=False,
    )

    print(
        "Shadow pretrained load:",
        msg,
        flush=True,
    )

    return model.to(
        device
    )


# ============================================================
# Warmup + cosine LR
# ============================================================

def get_warmup_cosine_lr(
    epoch,
    total_epochs,
    base_lr,
    min_lr,
    warmup_epochs,
):
    """
    Epoch-level warmup + cosine schedule.
    """

    if (
        warmup_epochs > 0
        and epoch < warmup_epochs
    ):
        return (
            base_lr
            * float(epoch + 1)
            / float(warmup_epochs)
        )

    if total_epochs <= warmup_epochs:
        return base_lr

    progress = (
        float(
            epoch - warmup_epochs
        )
        / float(
            max(
                1,
                total_epochs
                - warmup_epochs
                - 1,
            )
        )
    )

    progress = min(
        max(
            progress,
            0.0,
        ),
        1.0,
    )

    return (
        min_lr
        + 0.5
        * (
            base_lr
            - min_lr
        )
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )


# ============================================================
# Train one matched ViT shadow
# ============================================================

def train_shadow_vit(
    model,
    train_dataset,
    device,
    epochs=10,
    batch_size=24,
    blr=0.001,
    min_lr=1e-6,
    warmup_epochs=2,
    weight_decay=0.05,
    layer_decay=0.75,
    label_smoothing=0.1,
    betas=(0.9, 0.999),
    num_workers=2,
):
    """
    Fine-tune one architecture-matched ViT shadow.

    Matches victim as closely as practical:
        - AdamW
        - layer-wise LR decay
        - base LR scaling
        - weight decay
        - label smoothing
        - warmup + cosine
    """

    loader = _make_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        device=device,
        num_workers=num_workers,
    )

    # --------------------------------------------------------
    # Victim-style LR scaling
    #
    # lr = blr * effective_batch_size / 256
    #
    # Here:
    # accum_iter = 1
    # world_size = 1
    # --------------------------------------------------------

    actual_batch_size = int(
        loader.batch_size
    )

    base_lr = (
        float(blr)
        * actual_batch_size
        / 256.0
    )

    print(
        f"  Shadow batch size: "
        f"{actual_batch_size}",
        flush=True,
    )

    print(
        f"  Shadow base LR: "
        f"{base_lr:.8e}",
        flush=True,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    if label_smoothing > 0:
        criterion = (
            LabelSmoothingCrossEntropy(
                smoothing=float(
                    label_smoothing
                )
            )
        )
    else:
        criterion = (
            torch.nn.CrossEntropyLoss()
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    try:
        param_groups = (
            lrd.param_groups_lrd(
                model,
                float(
                    weight_decay
                ),
                no_weight_decay_list=(
                    model.no_weight_decay()
                ),
                layer_decay=float(
                    layer_decay
                ),
            )
        )

        optimizer = (
            torch.optim.AdamW(
                param_groups,
                lr=base_lr,
                betas=tuple(
                    betas
                ),
            )
        )

        print(
            "  Using layer-wise "
            "learning-rate decay "
            f"(layer_decay="
            f"{layer_decay}).",
            flush=True,
        )

    except Exception as exc:
        print(
            "  Layer-wise LR decay "
            "failed; using standard "
            "AdamW.",
            flush=True,
        )

        print(
            f"  Reason: {exc}",
            flush=True,
        )

        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=base_lr,
                betas=tuple(
                    betas
                ),
                weight_decay=float(
                    weight_decay
                ),
            )
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        int(epochs)
    ):
        current_lr = (
            get_warmup_cosine_lr(
                epoch=epoch,
                total_epochs=int(
                    epochs
                ),
                base_lr=base_lr,
                min_lr=float(
                    min_lr
                ),
                warmup_epochs=int(
                    warmup_epochs
                ),
            )
        )

        for param_group in (
            optimizer.param_groups
        ):
            lr_scale = float(
                param_group.get(
                    "lr_scale",
                    1.0,
                )
            )

            param_group["lr"] = (
                current_lr
                * lr_scale
            )

        model.train()

        epoch_loss = 0.0
        num_batches = 0
        num_correct = 0
        num_samples = 0

        for (
            images,
            labels,
        ) in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            ).long()

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            epoch_loss += float(
                loss.item()
            )

            num_batches += 1

            with torch.no_grad():
                preds = torch.argmax(
                    logits,
                    dim=1,
                )

                num_correct += int(
                    preds.eq(
                        labels
                    ).sum().item()
                )

                num_samples += int(
                    labels.numel()
                )

        avg_loss = (
            epoch_loss
            / max(
                num_batches,
                1,
            )
        )

        train_acc = (
            num_correct
            / max(
                num_samples,
                1,
            )
        )

        print(
            f"    shadow epoch "
            f"{epoch + 1}/{epochs}, "
            f"lr={current_lr:.8e}, "
            f"loss={avg_loss:.6f}, "
            f"acc={train_acc:.4f}",
            flush=True,
        )

    return model


# ============================================================
# True-class probability
# ============================================================

@torch.no_grad()
def get_true_class_probs(
    model,
    dataset,
    device="cuda",
    batch_size=16,
    num_workers=2,
):
    """
    Return p_theta(y|x) in dataset order.
    """

    if len(dataset) == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    loader = _make_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        num_workers=num_workers,
    )

    model.eval()

    outputs_all = []

    for (
        images,
        labels,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).long()

        logits = model(
            images
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )

        true_probs = probs.gather(
            1,
            labels[:, None],
        ).squeeze(
            1
        )

        outputs_all.append(
            true_probs.cpu()
        )

    return (
        torch.cat(
            outputs_all
        )
        .numpy()
        .astype(
            np.float64
        )
    )


# ============================================================
# Balanced IN / OUT assignment
# ============================================================

def _balanced_in_indices(
    num_targets,
    shadow_index,
    previous_mask,
    generator,
):
    """
    Use paired complementary IN/OUT assignments.

    Shadow 0:
        random 50% IN

    Shadow 1:
        complement of shadow 0

    Shadow 2:
        new random 50% IN

    Shadow 3:
        complement of shadow 2
    """

    if shadow_index % 2 == 0:

        permutation = torch.randperm(
            num_targets,
            generator=generator,
        )

        num_in = (
            num_targets
            // 2
        )

        indices = permutation[
            :num_in
        ]

    else:
        if previous_mask is None:
            raise RuntimeError(
                "previous_mask is required "
                "for complementary assignment."
            )

        indices = torch.from_numpy(
            np.flatnonzero(
                ~previous_mask
            )
        ).long()

    return indices


# ============================================================
# Shadow/reference estimation
# ============================================================

def shadow_zone(
    shadow_dataset,
    target_dataset,
    pretrained_path,
    num_classes=2,
    model_name="vit_large_patch16",
    img_size=448,
    num_shadow_models=8,
    shadow_epochs=10,
    shadow_blr=0.001,
    shadow_min_lr=1e-6,
    shadow_warmup_epochs=2,
    shadow_weight_decay=0.05,
    shadow_layer_decay=0.75,
    shadow_label_smoothing=0.1,
    shadow_betas=(0.9, 0.999),
    random_sample_number=200,
    shadow_train_fraction=0.5,
    shadow_batch_size=24,
    eval_batch_size=16,
    num_workers=2,
    seed=42,
    device="cuda",
):
    """
    Estimate reference probabilities using K
    architecture-matched ViT shadow models.

    Full image tensors are never materialized.
    """

    num_targets = len(
        target_dataset
    )

    num_shadow = len(
        shadow_dataset
    )

    if num_targets == 0:
        raise ValueError(
            "target_dataset is empty."
        )

    if num_shadow == 0:
        raise ValueError(
            "shadow_dataset is empty."
        )

    if int(
        num_shadow_models
    ) < 2:
        raise ValueError(
            "num_shadow_models "
            "must be >= 2."
        )

    if int(
        num_shadow_models
    ) % 2 != 0:
        raise ValueError(
            "Use an even number of "
            "shadow models so each "
            "random IN assignment has "
            "a complementary OUT model."
        )

    if not (
        0.0
        < float(
            shadow_train_fraction
        )
        <= 1.0
    ):
        raise ValueError(
            "shadow_train_fraction "
            "must be in (0, 1]."
        )

    random_sample_number = min(
        int(
            random_sample_number
        ),
        num_shadow,
    )

    generator = (
        torch.Generator()
        .manual_seed(
            int(seed)
        )
    )

    # --------------------------------------------------------
    # Fixed random/reference set
    # --------------------------------------------------------

    random_indices = (
        torch.randperm(
            num_shadow,
            generator=generator,
        )[
            :random_sample_number
        ]
        .tolist()
    )

    random_dataset = Subset(
        shadow_dataset,
        random_indices,
    )

    # Small numeric arrays only.
    target_probs_all = np.zeros(
        (
            int(
                num_shadow_models
            ),
            num_targets,
        ),
        dtype=np.float64,
    )

    target_in_mask = np.zeros(
        (
            int(
                num_shadow_models
            ),
            num_targets,
        ),
        dtype=bool,
    )

    random_probs_all = np.zeros(
        (
            int(
                num_shadow_models
            ),
            random_sample_number,
        ),
        dtype=np.float64,
    )

    print(
        f"RMIA: target={num_targets}, "
        f"shadow_pool={num_shadow}, "
        f"shadow_models="
        f"{num_shadow_models}, "
        f"random_refs="
        f"{random_sample_number}",
        flush=True,
    )

    previous_mask = None

    # ========================================================
    # Train K matched ViT shadow models
    # ========================================================

    for shadow_index in range(
        int(
            num_shadow_models
        )
    ):
        print(
            "\n"
            + "=" * 70,
            flush=True,
        )

        print(
            f"Training shadow model "
            f"{shadow_index + 1}/"
            f"{num_shadow_models}",
            flush=True,
        )

        # ----------------------------------------------------
        # Measurement IN subset
        # ----------------------------------------------------

        in_indices = (
            _balanced_in_indices(
                num_targets=(
                    num_targets
                ),
                shadow_index=(
                    shadow_index
                ),
                previous_mask=(
                    previous_mask
                ),
                generator=generator,
            )
        )

        current_mask = np.zeros(
            num_targets,
            dtype=bool,
        )

        current_mask[
            in_indices.numpy()
        ] = True

        target_in_mask[
            shadow_index
        ] = current_mask

        previous_mask = (
            current_mask
        )

        # ----------------------------------------------------
        # Auxiliary subset
        # ----------------------------------------------------

        num_aux_train = max(
            1,
            int(
                round(
                    num_shadow
                    * float(
                        shadow_train_fraction
                    )
                )
            ),
        )

        aux_indices = (
            torch.randperm(
                num_shadow,
                generator=generator,
            )[
                :num_aux_train
            ]
            .tolist()
        )

        auxiliary_subset = Subset(
            shadow_dataset,
            aux_indices,
        )

        in_target_subset = Subset(
            target_dataset,
            in_indices.tolist(),
        )

        train_dataset = (
            ConcatDataset(
                [
                    auxiliary_subset,
                    in_target_subset,
                ]
            )
        )

        print(
            f"  Auxiliary samples: "
            f"{len(auxiliary_subset)}",
            flush=True,
        )

        print(
            f"  Measurement IN samples: "
            f"{len(in_target_subset)}",
            flush=True,
        )

        print(
            f"  Shadow train samples: "
            f"{len(train_dataset)}",
            flush=True,
        )

        # ----------------------------------------------------
        # Same architecture + same public initialization
        # ----------------------------------------------------

        model = build_shadow_model(
            pretrained_path=(
                pretrained_path
            ),
            device=device,
            model_name=model_name,
            img_size=img_size,
            num_classes=num_classes,
            drop_path_rate=0.1,
            global_pool=True,
        )

        # ----------------------------------------------------
        # Victim-matched fine-tuning recipe
        # ----------------------------------------------------

        model = train_shadow_vit(
            model=model,
            train_dataset=train_dataset,
            device=device,
            epochs=shadow_epochs,
            batch_size=(
                shadow_batch_size
            ),
            blr=shadow_blr,
            min_lr=shadow_min_lr,
            warmup_epochs=(
                shadow_warmup_epochs
            ),
            weight_decay=(
                shadow_weight_decay
            ),
            layer_decay=(
                shadow_layer_decay
            ),
            label_smoothing=(
                shadow_label_smoothing
            ),
            betas=shadow_betas,
            num_workers=num_workers,
        )

        # ----------------------------------------------------
        # Evaluate all measurement samples
        # ----------------------------------------------------

        target_probs_all[
            shadow_index
        ] = get_true_class_probs(
            model=model,
            dataset=target_dataset,
            device=device,
            batch_size=(
                eval_batch_size
            ),
            num_workers=(
                num_workers
            ),
        )

        # ----------------------------------------------------
        # Evaluate fixed random references
        # ----------------------------------------------------

        random_probs_all[
            shadow_index
        ] = get_true_class_probs(
            model=model,
            dataset=random_dataset,
            device=device,
            batch_size=(
                eval_batch_size
            ),
            num_workers=(
                num_workers
            ),
        )

        del model
        del train_dataset
        del auxiliary_subset
        del in_target_subset

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # IN / OUT reference estimates
    # ========================================================

    in_mean = np.empty(
        num_targets,
        dtype=np.float64,
    )

    out_mean = np.empty(
        num_targets,
        dtype=np.float64,
    )

    for target_index in range(
        num_targets
    ):
        in_values = target_probs_all[
            target_in_mask[
                :,
                target_index
            ],
            target_index,
        ]

        out_values = target_probs_all[
            ~target_in_mask[
                :,
                target_index
            ],
            target_index,
        ]

        if (
            in_values.size == 0
            or out_values.size == 0
        ):
            raise RuntimeError(
                f"Target sample "
                f"{target_index} lacks "
                "IN or OUT observations."
            )

        in_mean[
            target_index
        ] = in_values.mean()

        out_mean[
            target_index
        ] = out_values.mean()

    target_reference_probs = (
        0.5
        * (
            in_mean
            + out_mean
        )
    )

    random_reference_probs = (
        random_probs_all.mean(
            axis=0
        )
    )

    print(
        "Finished shadow estimation. "
        "Mean target reference "
        "probability="
        f"{target_reference_probs.mean():.6f}",
        flush=True,
    )

    return (
        random_dataset,
        target_reference_probs,
        random_reference_probs,
    )


# ============================================================
# RMIA attack
# ============================================================

def attack_zone(
    target_model,
    target_dataset,
    random_dataset,
    target_reference_probs,
    random_reference_probs,
    gamma=0.5,
    eval_batch_size=16,
    num_workers=2,
    device="cuda",
):
    """
    Compute RMIA score.

    Larger score = more member-like under
    the current RMIA definition.
    """

    eps = 1e-12

    print(
        "\nComputing target-model "
        "probabilities...",
        flush=True,
    )

    target_model_probs = (
        get_true_class_probs(
            model=target_model,
            dataset=target_dataset,
            device=device,
            batch_size=(
                eval_batch_size
            ),
            num_workers=(
                num_workers
            ),
        )
    )

    target_model_random_probs = (
        get_true_class_probs(
            model=target_model,
            dataset=random_dataset,
            device=device,
            batch_size=(
                eval_batch_size
            ),
            num_workers=(
                num_workers
            ),
        )
    )

    lr_target = (
        target_model_probs
        / np.maximum(
            target_reference_probs,
            eps,
        )
    )

    lr_random = (
        target_model_random_probs
        / np.maximum(
            random_reference_probs,
            eps,
        )
    )

    relative_ratio = (
        lr_target[:, None]
        / np.maximum(
            lr_random[
                None,
                :
            ],
            eps,
        )
    )

    scores = np.mean(
        relative_ratio
        > float(gamma),
        axis=1,
    ).astype(
        np.float64
    )

    print(
        f"RMIA score mean="
        f"{scores.mean():.6f}, "
        f"min={scores.min():.6f}, "
        f"max={scores.max():.6f}",
        flush=True,
    )

    return scores


# ============================================================
# Public interface called by main_rmia.py
# ============================================================

def RMIA(
    target_dataset,
    target_model,
    shadow_dataset,
    pretrained_path,
    num_classes=2,
    model_name="vit_large_patch16",
    img_size=448,
    num_shadow_models=8,
    shadow_epochs=10,
    shadow_blr=0.001,
    shadow_min_lr=1e-6,
    shadow_warmup_epochs=2,
    shadow_weight_decay=0.05,
    shadow_layer_decay=0.75,
    shadow_label_smoothing=0.1,
    shadow_betas=(0.9, 0.999),
    random_sample_number=200,
    gamma=0.5,
    shadow_train_fraction=0.5,
    shadow_batch_size=24,
    eval_batch_size=16,
    num_workers=2,
    seed=42,
    device="cuda",
):
    """
    Low-memory, architecture-matched RMIA.

    target_dataset / shadow_dataset:
        must yield (image, label).

    Use XYDataset when the underlying dataset
    returns (image, label, path).
    """

    print(
        "\n" + "=" * 80,
        flush=True,
    )

    print(
        "Starting architecture-matched "
        "low-memory RMIA",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Target samples: "
        f"{len(target_dataset)}",
        flush=True,
    )

    print(
        f"Shadow/reference samples: "
        f"{len(shadow_dataset)}",
        flush=True,
    )

    (
        random_dataset,
        target_ref_probs,
        random_ref_probs,
    ) = shadow_zone(
        shadow_dataset=(
            shadow_dataset
        ),
        target_dataset=(
            target_dataset
        ),
        pretrained_path=(
            pretrained_path
        ),
        num_classes=num_classes,
        model_name=model_name,
        img_size=img_size,
        num_shadow_models=(
            num_shadow_models
        ),
        shadow_epochs=(
            shadow_epochs
        ),
        shadow_blr=(
            shadow_blr
        ),
        shadow_min_lr=(
            shadow_min_lr
        ),
        shadow_warmup_epochs=(
            shadow_warmup_epochs
        ),
        shadow_weight_decay=(
            shadow_weight_decay
        ),
        shadow_layer_decay=(
            shadow_layer_decay
        ),
        shadow_label_smoothing=(
            shadow_label_smoothing
        ),
        shadow_betas=(
            shadow_betas
        ),
        random_sample_number=(
            random_sample_number
        ),
        shadow_train_fraction=(
            shadow_train_fraction
        ),
        shadow_batch_size=(
            shadow_batch_size
        ),
        eval_batch_size=(
            eval_batch_size
        ),
        num_workers=num_workers,
        seed=seed,
        device=device,
    )

    scores = attack_zone(
        target_model=target_model,
        target_dataset=target_dataset,
        random_dataset=random_dataset,
        target_reference_probs=(
            target_ref_probs
        ),
        random_reference_probs=(
            random_ref_probs
        ),
        gamma=gamma,
        eval_batch_size=(
            eval_batch_size
        ),
        num_workers=num_workers,
        device=device,
    )

    print(
        "Architecture-matched "
        "low-memory RMIA finished.",
        flush=True,
    )

    return scores