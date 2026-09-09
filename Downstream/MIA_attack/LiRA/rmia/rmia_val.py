import copy
import gc
import math
import os
import sys

import numpy as np
import torch

from sklearn.metrics import roc_auc_score

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
        drop_path_rate=drop_path_rate,
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

    checkpoint_model = checkpoint_model.copy()

    state_dict = model.state_dict()

    # --------------------------------------------------------
    # Remove incompatible classification head
    # --------------------------------------------------------

    for key in (
        "head.weight",
        "head.bias",
    ):
        if (
            key in checkpoint_model
            and (
                key not in state_dict
                or checkpoint_model[key].shape
                != state_dict[key].shape
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

    # --------------------------------------------------------
    # Position embedding interpolation
    # --------------------------------------------------------

    interpolate_pos_embed(
        model,
        checkpoint_model,
    )

    # --------------------------------------------------------
    # Initialize classification head
    # --------------------------------------------------------

    trunc_normal_(
        model.head.weight,
        std=2e-5,
    )

    if model.head.bias is not None:
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
# Validation AUROC
# ============================================================

@torch.no_grad()
def evaluate_validation_auc(
    model,
    val_dataset,
    device,
    batch_size=16,
    num_workers=2,
):
    """
    Evaluate classification AUROC on an independent
    shadow validation set.

    For binary classification:
        score = P(y = 1 | x)

    Returns
    -------
    auc : float
        Validation AUROC.

    accuracy : float
        Validation classification accuracy.
    """

    loader = _make_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        num_workers=num_workers,
    )

    model.eval()

    labels_all = []
    scores_all = []
    preds_all = []

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

        # Binary classification:
        # positive-class probability
        positive_probs = probs[
            :,
            1
        ]

        preds = torch.argmax(
            logits,
            dim=1,
        )

        labels_all.append(
            labels.cpu()
        )

        scores_all.append(
            positive_probs.cpu()
        )

        preds_all.append(
            preds.cpu()
        )

    labels_all = (
        torch.cat(
            labels_all
        )
        .numpy()
        .astype(
            np.int64
        )
    )

    scores_all = (
        torch.cat(
            scores_all
        )
        .numpy()
        .astype(
            np.float64
        )
    )

    preds_all = (
        torch.cat(
            preds_all
        )
        .numpy()
        .astype(
            np.int64
        )
    )

    unique_labels = np.unique(
        labels_all
    )

    # AUROC is undefined if validation happens
    # to contain only one class.
    if len(unique_labels) < 2:
        raise RuntimeError(
            "Shadow validation set contains only "
            f"one class: {unique_labels}. "
            "Increase validation size or use a "
            "stratified split."
        )

    auc = roc_auc_score(
        labels_all,
        scores_all,
    )

    accuracy = np.mean(
        preds_all
        == labels_all
    )

    return (
        float(auc),
        float(accuracy),
    )


# ============================================================
# Train one matched ViT shadow
# ============================================================

def train_shadow_vit(
    model,
    train_dataset,
    val_dataset,
    device,
    epochs=10,
    batch_size=24,
    val_batch_size=16,
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

    Model selection:
        After every epoch, evaluate the model on an
        independent shadow validation set.

        The checkpoint with the highest validation
        classification AUROC is retained.

        The validation set must NOT contain the RMIA
        measurement samples.
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

    print(
        f"  Shadow train samples: "
        f"{len(train_dataset)}",
        flush=True,
    )

    print(
        f"  Shadow validation samples: "
        f"{len(val_dataset)}",
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

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            betas=tuple(
                betas
            ),
        )

        print(
            "  Using layer-wise "
            "learning-rate decay "
            f"(layer_decay={layer_decay}).",
            flush=True,
        )

    except Exception as exc:

        print(
            "  Layer-wise LR decay "
            "failed; using standard AdamW.",
            flush=True,
        )

        print(
            f"  Reason: {exc}",
            flush=True,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=tuple(
                betas
            ),
            weight_decay=float(
                weight_decay
            ),
        )

    # ========================================================
    # Best checkpoint tracking
    # ========================================================

    best_val_auc = -np.inf
    best_val_acc = -np.inf
    best_epoch = -1
    best_state_dict = None

    # ========================================================
    # Training
    # ========================================================

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

            param_group[
                "lr"
            ] = (
                current_lr
                * lr_scale
            )

        # ----------------------------------------------------
        # Train one epoch
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Validation after this epoch
        # ----------------------------------------------------

        val_auc, val_acc = (
            evaluate_validation_auc(
                model=model,
                val_dataset=val_dataset,
                device=device,
                batch_size=val_batch_size,
                num_workers=num_workers,
            )
        )

        print(
            f"    shadow epoch "
            f"{epoch + 1}/{epochs}, "
            f"lr={current_lr:.8e}, "
            f"loss={avg_loss:.6f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_acc={val_acc:.4f}, "
            f"val_auc={val_auc:.6f}",
            flush=True,
        )

        # ----------------------------------------------------
        # Keep checkpoint with highest validation AUROC
        # ----------------------------------------------------

        if val_auc > best_val_auc:

            best_val_auc = float(
                val_auc
            )

            best_val_acc = float(
                val_acc
            )

            best_epoch = (
                epoch + 1
            )

            # Store on CPU to avoid keeping an additional
            # full ViT checkpoint in GPU memory.
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            print(
                f"      -> New best shadow checkpoint: "
                f"epoch={best_epoch}, "
                f"val_auc={best_val_auc:.6f}",
                flush=True,
            )

    # ========================================================
    # Restore best checkpoint
    # ========================================================

    if best_state_dict is None:
        raise RuntimeError(
            "No valid shadow checkpoint was selected."
        )

    print(
        "\n"
        f"  Best shadow checkpoint: "
        f"epoch={best_epoch}/{epochs}, "
        f"val_auc={best_val_auc:.6f}, "
        f"val_acc={best_val_acc:.4f}",
        flush=True,
    )

    model.load_state_dict(
        best_state_dict,
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    del best_state_dict

    return (
        model,
        best_epoch,
        best_val_auc,
    )


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
    shadow_val_fraction=0.2,
    shadow_batch_size=24,
    eval_batch_size=16,
    num_workers=2,
    seed=42,
    device="cuda",
):
    """
    Estimate reference probabilities using K
    architecture-matched ViT shadow models.

    Each shadow model:
        1. samples an auxiliary pool;
        2. splits that pool into train / validation;
        3. adds measurement-IN samples ONLY to training;
        4. trains for shadow_epochs;
        5. selects the checkpoint with highest
           validation AUROC;
        6. uses that best checkpoint for RMIA
           reference probability estimation.

    The measurement target samples are never used
    for validation checkpoint selection.

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

    if not (
        0.0
        < float(
            shadow_val_fraction
        )
        < 1.0
    ):
        raise ValueError(
            "shadow_val_fraction "
            "must be in (0, 1)."
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

    # --------------------------------------------------------
    # Output arrays
    # --------------------------------------------------------

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

    # Track selected epochs for debugging/reporting.
    selected_epochs = np.zeros(
        int(
            num_shadow_models
        ),
        dtype=np.int64,
    )

    selected_val_aucs = np.zeros(
        int(
            num_shadow_models
        ),
        dtype=np.float64,
    )

    print(
        f"RMIA: target={num_targets}, "
        f"shadow_pool={num_shadow}, "
        f"shadow_models={num_shadow_models}, "
        f"random_refs={random_sample_number}, "
        f"shadow_val_fraction="
        f"{shadow_val_fraction}",
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
                num_targets=num_targets,
                shadow_index=shadow_index,
                previous_mask=previous_mask,
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

        previous_mask = current_mask

        # ----------------------------------------------------
        # Sample auxiliary pool
        # ----------------------------------------------------

        num_aux_total = max(
            2,
            int(
                round(
                    num_shadow
                    * float(
                        shadow_train_fraction
                    )
                )
            ),
        )

        num_aux_total = min(
            num_aux_total,
            num_shadow,
        )

        aux_indices = (
            torch.randperm(
                num_shadow,
                generator=generator,
            )[
                :num_aux_total
            ]
            .tolist()
        )

        # ----------------------------------------------------
        # Split auxiliary pool into train / validation
        #
        # IMPORTANT:
        # measurement samples are NOT used for validation.
        # ----------------------------------------------------

        num_aux_val = max(
            1,
            int(
                round(
                    num_aux_total
                    * float(
                        shadow_val_fraction
                    )
                )
            ),
        )

        # Must retain at least one auxiliary
        # sample for training.
        num_aux_val = min(
            num_aux_val,
            num_aux_total - 1,
        )

        num_aux_train = (
            num_aux_total
            - num_aux_val
        )

        auxiliary_train_indices = (
            aux_indices[
                :num_aux_train
            ]
        )

        auxiliary_val_indices = (
            aux_indices[
                num_aux_train:
            ]
        )

        auxiliary_train_subset = Subset(
            shadow_dataset,
            auxiliary_train_indices,
        )

        auxiliary_val_subset = Subset(
            shadow_dataset,
            auxiliary_val_indices,
        )

        in_target_subset = Subset(
            target_dataset,
            in_indices.tolist(),
        )

        # ----------------------------------------------------
        # Shadow training:
        #
        # auxiliary training samples
        # +
        # measurement-IN samples
        #
        # Validation:
        #
        # auxiliary validation samples ONLY
        # ----------------------------------------------------

        train_dataset = ConcatDataset(
            [
                auxiliary_train_subset,
                in_target_subset,
            ]
        )

        print(
            f"  Auxiliary pool: "
            f"{num_aux_total}",
            flush=True,
        )

        print(
            f"  Auxiliary train samples: "
            f"{len(auxiliary_train_subset)}",
            flush=True,
        )

        print(
            f"  Auxiliary validation samples: "
            f"{len(auxiliary_val_subset)}",
            flush=True,
        )

        print(
            f"  Measurement IN samples: "
            f"{len(in_target_subset)}",
            flush=True,
        )

        print(
            f"  Total shadow train samples: "
            f"{len(train_dataset)}",
            flush=True,
        )

        # ----------------------------------------------------
        # Same architecture + public initialization
        # ----------------------------------------------------

        model = build_shadow_model(
            pretrained_path=pretrained_path,
            device=device,
            model_name=model_name,
            img_size=img_size,
            num_classes=num_classes,
            drop_path_rate=0.1,
            global_pool=True,
        )

        # ----------------------------------------------------
        # Victim-matched training + best checkpoint selection
        # ----------------------------------------------------

        (
            model,
            best_epoch,
            best_val_auc,
        ) = train_shadow_vit(
            model=model,
            train_dataset=train_dataset,
            val_dataset=auxiliary_val_subset,
            device=device,
            epochs=shadow_epochs,
            batch_size=shadow_batch_size,
            val_batch_size=eval_batch_size,
            blr=shadow_blr,
            min_lr=shadow_min_lr,
            warmup_epochs=shadow_warmup_epochs,
            weight_decay=shadow_weight_decay,
            layer_decay=shadow_layer_decay,
            label_smoothing=shadow_label_smoothing,
            betas=shadow_betas,
            num_workers=num_workers,
        )

        selected_epochs[
            shadow_index
        ] = best_epoch

        selected_val_aucs[
            shadow_index
        ] = best_val_auc

        # ----------------------------------------------------
        # Evaluate all measurement samples using BEST epoch
        # ----------------------------------------------------

        target_probs_all[
            shadow_index
        ] = get_true_class_probs(
            model=model,
            dataset=target_dataset,
            device=device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
        )

        # ----------------------------------------------------
        # Evaluate fixed random references using BEST epoch
        # ----------------------------------------------------

        random_probs_all[
            shadow_index
        ] = get_true_class_probs(
            model=model,
            dataset=random_dataset,
            device=device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
        )

        print(
            f"  Shadow {shadow_index + 1} "
            f"reference probabilities generated "
            f"using best epoch {best_epoch} "
            f"(val_auc={best_val_auc:.6f}).",
            flush=True,
        )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del model
        del train_dataset
        del auxiliary_train_subset
        del auxiliary_val_subset
        del in_target_subset

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Report selected epochs
    # ========================================================

    print(
        "\n"
        + "=" * 70,
        flush=True,
    )

    print(
        "Shadow best-checkpoint summary",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    for shadow_index in range(
        int(
            num_shadow_models
        )
    ):

        print(
            f"Shadow "
            f"{shadow_index + 1:02d}: "
            f"best_epoch="
            f"{selected_epochs[shadow_index]}, "
            f"val_auc="
            f"{selected_val_aucs[shadow_index]:.6f}",
            flush=True,
        )

    print(
        f"Mean selected epoch: "
        f"{selected_epochs.mean():.2f}",
        flush=True,
    )

    print(
        f"Mean best validation AUROC: "
        f"{selected_val_aucs.mean():.6f}",
        flush=True,
    )

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
            batch_size=eval_batch_size,
            num_workers=num_workers,
        )
    )

    target_model_random_probs = (
        get_true_class_probs(
            model=target_model,
            dataset=random_dataset,
            device=device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
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
        > float(
            gamma
        ),
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

    return (
        scores,
        lr_target,
        lr_random,
    )


# ============================================================
# Public interface
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

    # NEW:
    # Fraction of sampled auxiliary data reserved
    # for shadow validation / checkpoint selection.
    shadow_val_fraction=0.2,

    shadow_batch_size=24,
    eval_batch_size=16,
    num_workers=2,
    seed=42,
    device="cuda",
):
    """
    Low-memory, architecture-matched RMIA with
    validation-based shadow checkpoint selection.

    target_dataset / shadow_dataset:
        must yield (image, label).

    Use XYDataset when the underlying dataset
    returns (image, label, path).

    Shadow model selection:
        Each shadow model is trained for
        `shadow_epochs`, and the checkpoint with
        the highest AUROC on an independent
        auxiliary validation subset is restored
        before RMIA reference probabilities are
        calculated.
    """

    print(
        "\n"
        + "=" * 80,
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

    print(
        f"Shadow validation fraction: "
        f"{shadow_val_fraction}",
        flush=True,
    )

    (
        random_dataset,
        target_ref_probs,
        random_ref_probs,
    ) = shadow_zone(
        shadow_dataset=shadow_dataset,
        target_dataset=target_dataset,
        pretrained_path=pretrained_path,
        num_classes=num_classes,
        model_name=model_name,
        img_size=img_size,
        num_shadow_models=num_shadow_models,
        shadow_epochs=shadow_epochs,
        shadow_blr=shadow_blr,
        shadow_min_lr=shadow_min_lr,
        shadow_warmup_epochs=shadow_warmup_epochs,
        shadow_weight_decay=shadow_weight_decay,
        shadow_layer_decay=shadow_layer_decay,
        shadow_label_smoothing=shadow_label_smoothing,
        shadow_betas=shadow_betas,
        random_sample_number=random_sample_number,
        shadow_train_fraction=shadow_train_fraction,
        shadow_val_fraction=shadow_val_fraction,
        shadow_batch_size=shadow_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed,
        device=device,
    )

    (
        scores,
        lr_target,
        lr_random,
    ) = attack_zone(
        target_model=target_model,
        target_dataset=target_dataset,
        random_dataset=random_dataset,
        target_reference_probs=target_ref_probs,
        random_reference_probs=random_ref_probs,
        gamma=gamma,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        device=device,
    )

    print(
        "Architecture-matched "
        "low-memory RMIA finished.",
        flush=True,
    )

    return (
        scores,
        lr_target,
        lr_random,
    )