"""Gradient-based membership inference attack."""

import os
import sys
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    roc_curve,
)

import matplotlib.pyplot as plt


gradient_scope = "all" # head, all,last_block_head
print(f"gradient_scope:{gradient_scope}")
# ============================================================
# Project imports
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )

from model import load_model_from_checkpoint
from DP.eyedataset import ImageListDataset


# ============================================================
# 1. Configuration
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMAGE_SIZE = 448

# ------------------------------------------------------------
# Plaintext training dataset root.
#
# Member paths in MEMBER_LIST are relative to this directory,
# e.g.
#
# class_0/xxx.jpg
# class_1/yyy.jpg
# ------------------------------------------------------------

DATASET_ROOT = (
    "/path/to/workspace/dataset/"
    "DeepUWF/SH_DR/shrdr/train"
)

# ------------------------------------------------------------
# Non-member dataset
# ------------------------------------------------------------

TEST_ROOT = (
    "/path/to/workspace/dataset/"
    "DeepUWF/SH_DR/shrdr/test"
)

# ------------------------------------------------------------
# Images that actually participated in training.
#
# IMPORTANT:
# This file should correspond to the checkpoint being attacked.
# For model_standalone_ep0.pt, use trainset_0.txt.
# ------------------------------------------------------------

MEMBER_LIST = (
    "/path/to/workspace/weights/"
    "DeepUWFDP/"
    "SHDR_random_attack_results_robust_subspace_moredata/"
    "trainset_0.txt"
)

# ------------------------------------------------------------
# Target model checkpoint
# ------------------------------------------------------------

# CHECKPOINT_PATH = (
#     "/path/to/workspace/weights/"
#     "DeepUWFDP/"
#     "SHDR_random_attack_results_robust_subspace_moredata/"
#     "model_standalone_ep0.pt"
# )


CHECKPOINT_PATH = "/path/to/workspace/weights/DeepUWFDP/SHDR_ep1_adamw/model_standalone_ep0.pt"
#CHECKPOINT_PATH = "/path/to/workspace/weights/DeepUWFDP/SHDR_NoDPBaseline_adamw_savegrad/model_standalone_ep0.pt"
#CHECKPOINT_PATH = "/path/to/workspace/weights/DeepUWFDP/SHDR_ep32_adamw/model_standalone_ep0.pt"
#CHECKPOINT_PATH = "/path/to/workspace/weights/DeepUWFDP/SHDR_ep8_adamw/model_standalone_ep0.pt"


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

# OUTPUT_DIR = Path(
#     f"/path/to/workspace/code/DeepUWF/MIA_attack/experiments/gradient_mia_results_{gradient_scope}"
# )

OUTPUT_DIR = Path(
    f"/path/to/workspace/code/DeepUWF/MIA_attack/experiments/gradient_mia_results_dpprivacyep1epoch0_{gradient_scope}"
)
# OUTPUT_DIR = Path(
#     f"/path/to/workspace/code/DeepUWF/MIA_attack/experiments/gradient_mia_results_baselineepoch0_{gradient_scope}"
# )
# OUTPUT_DIR = Path(
#     f"/path/to/workspace/code/DeepUWF/MIA_attack/experiments/gradient_mia_results_dpprivacyep32epoch0_{gradient_scope}"
# )

# OUTPUT_DIR = Path(
#     f"/path/to/workspace/code/DeepUWF/MIA_attack/experiments/gradient_mia_results_dpprivacyep8epoch0_{gradient_scope}"
# )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ------------------------------------------------------------
# Score cache.
#
# Set USE_SAVED_SCORES=True after the gradient scores have
# already been computed once. This avoids running thousands
# of expensive ViT backward passes again.
# ------------------------------------------------------------

SCORE_PATH = (
    OUTPUT_DIR
    / "gradient_mia_scores.npz"
)

USE_SAVED_SCORES = False

print(f"DATASET_ROOT:{DATASET_ROOT}")
print(f"TEST_ROOT:{TEST_ROOT}")
print(f"MEMBER_LIST:{MEMBER_LIST}")
print(f"CHECKPOINT_PATH:{CHECKPOINT_PATH}")
print(f"OUTPUT_DIR:{OUTPUT_DIR}")
# ============================================================
# 2. Evaluation preprocessing
# ============================================================

eval_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=(
                transforms.InterpolationMode.BICUBIC
            ),
        ),

        transforms.CenterCrop(
            IMAGE_SIZE
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# 3. Prepare Member / Non-member datasets
# ============================================================

member_dataset = ImageListDataset(
    dataset_root=DATASET_ROOT,
    list_file=MEMBER_LIST,
    transform=eval_transform,
)

non_member_dataset = ImageFolder(
    root=TEST_ROOT,
    transform=eval_transform,
)


member_loader = DataLoader(
    member_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)

non_member_loader = DataLoader(
    non_member_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)


print("=" * 70)
print("Gradient-based Membership Inference Attack")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

print(
    f"Member samples: "
    f"{len(member_dataset)}"
)

print(
    f"Non-member samples: "
    f"{len(non_member_dataset)}"
)

print(
    f"Member list: "
    f"{MEMBER_LIST}"
)

print(
    f"Target checkpoint: "
    f"{CHECKPOINT_PATH}"
)


# ============================================================
# 4. Load target model
# ============================================================

model = load_model_from_checkpoint(
    checkpoint_path=CHECKPOINT_PATH,
    num_classes=2,
    img_size=IMAGE_SIZE,
    device=DEVICE,
)

model.eval()


# ============================================================
# 5. Get classification-head parameters
# ============================================================

def get_target_params(
    model,
    gradient_scope="head",
):
    if gradient_scope == "head":
        if hasattr(model, "head"):
            return list(model.head.parameters())
        elif hasattr(model, "classifier"):
            return list(model.classifier.parameters())
        elif hasattr(model, "fc"):
            return list(model.fc.parameters())

    elif gradient_scope == "last_block_head":
        params = []

        if hasattr(model, "blocks"):
            params.extend(
                model.blocks[-1].parameters()
            )

        if hasattr(model, "head"):
            params.extend(
                model.head.parameters()
            )

        return list(params)

    elif gradient_scope == "all":
        return list(
            model.parameters()
        )

    else:
        raise ValueError(
            f"Unknown gradient_scope: "
            f"{gradient_scope}"
        )

def get_classification_head_parameters(
    model,
):
    """
    Find the classification head of the target model.

    Supports common architectures:
        model.head
        model.classifier
        model.fc

    Returns:
        List[nn.Parameter]
    """

    if hasattr(model, "head"):
        head_module = model.head

    elif hasattr(model, "classifier"):
        head_module = model.classifier

    elif hasattr(model, "fc"):
        head_module = model.fc

    else:
        raise RuntimeError(
            "Cannot find classification head. "
            "Expected model.head, "
            "model.classifier, or model.fc."
        )

    # target_params = list(
    #     head_module.parameters()
    # )

    target_params = get_target_params(
        model,
        gradient_scope=gradient_scope,
    )

    if len(target_params) == 0:
        raise RuntimeError(
            "Classification head contains "
            "no parameters."
        )

    for parameter in target_params:
        if not parameter.requires_grad:
            raise RuntimeError(
                "A classification-head parameter has "
                "requires_grad=False. Gradient MIA "
                "requires gradients with respect to "
                "the classification head."
            )

    return target_params


# ============================================================
# 6. Per-sample gradient norm
# ============================================================

def get_gradient_norms(
    model,
    dataloader,
    device,
    log_interval=500,
):
    """
    Compute the classification-head gradient L2 norm
    for every individual image.

    Gradient score:

        g(x) =
            || d L(f(x), y) / d theta_head ||_2

    For membership inference we later define:

        membership_score(x) = -g(x)

    because members are generally expected to have
    smaller loss gradients.

    IMPORTANT:
        batch_size must be 1.
    """

    model.eval()

    criterion = nn.CrossEntropyLoss(
        reduction="mean"
    )

    target_params = (
        get_classification_head_parameters(
            model
        )
    )

    gradient_norms = []

    total_samples = len(
        dataloader.dataset
    )

    for sample_index, batch in enumerate(
        dataloader
    ):
        # ImageListDataset:
        #     image, label, path
        #
        # ImageFolder:
        #     image, label

        if len(batch) == 3:
            inputs, labels, _ = batch

        elif len(batch) == 2:
            inputs, labels = batch

        else:
            raise ValueError(
                "Unexpected batch format. "
                f"Received {len(batch)} elements."
            )

        if inputs.shape[0] != 1:
            raise ValueError(
                "Gradient-based MIA requires "
                "batch_size=1 in order to compute "
                "true per-sample gradients."
            )

        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).long()

        # --------------------------------------------------------
        # Forward
        # --------------------------------------------------------

        outputs = model(
            inputs
        )

        loss = criterion(
            outputs,
            labels,
        )

        # --------------------------------------------------------
        # Compute gradients ONLY with respect to classification
        # head parameters.
        #
        # This avoids storing .grad for the whole ViT.
        # --------------------------------------------------------

        grads = torch.autograd.grad(
            outputs=loss,
            inputs=target_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        squared_norm = 0.0

        for grad in grads:
            if grad is None:
                continue

            squared_norm += (
                grad.detach()
                .float()
                .pow(2)
                .sum()
                .item()
            )

        gradient_norm = (
            squared_norm ** 0.5
        )

        gradient_norms.append(
            gradient_norm
        )

        if (
            sample_index == 0
            or (
                log_interval > 0
                and (sample_index + 1)
                % log_interval == 0
            )
            or sample_index + 1
            == total_samples
        ):
            print(
                f"Processed "
                f"{sample_index + 1}/"
                f"{total_samples} samples"
            )

    return np.asarray(
        gradient_norms,
        dtype=np.float64,
    )


# ============================================================
# 7. Calculate / load gradient scores
# ============================================================

if (
    USE_SAVED_SCORES
    and SCORE_PATH.exists()
):
    print(
        "\nLoading previously calculated "
        f"gradient scores from:\n{SCORE_PATH}"
    )

    saved_scores = np.load(
        SCORE_PATH
    )

    member_gradient_norms = (
        saved_scores[
            "member_norms"
        ]
    )

    non_member_gradient_norms = (
        saved_scores[
            "non_member_norms"
        ]
    )

else:
    print(
        "\nComputing MEMBER gradient norms..."
    )

    member_gradient_norms = (
        get_gradient_norms(
            model=model,
            dataloader=member_loader,
            device=DEVICE,
        )
    )

    print(
        "\nComputing NON-MEMBER gradient norms..."
    )

    non_member_gradient_norms = (
        get_gradient_norms(
            model=model,
            dataloader=non_member_loader,
            device=DEVICE,
        )
    )

    # --------------------------------------------------------
    # Save immediately because gradient extraction is expensive.
    # --------------------------------------------------------

    np.savez(
        SCORE_PATH,
        member_norms=(
            member_gradient_norms
        ),
        non_member_norms=(
            non_member_gradient_norms
        ),
    )

    print(
        "\nGradient scores saved to:"
    )

    print(
        SCORE_PATH
    )


# ============================================================
# 8. Sanity checks
# ============================================================

if (
    len(member_gradient_norms)
    != len(member_dataset)
):
    raise RuntimeError(
        "Member gradient score count does not "
        "match member dataset size."
    )

if (
    len(non_member_gradient_norms)
    != len(non_member_dataset)
):
    raise RuntimeError(
        "Non-member gradient score count does "
        "not match non-member dataset size."
    )

if not np.all(
    np.isfinite(
        member_gradient_norms
    )
):
    raise RuntimeError(
        "Non-finite values found in "
        "member gradient norms."
    )

if not np.all(
    np.isfinite(
        non_member_gradient_norms
    )
):
    raise RuntimeError(
        "Non-finite values found in "
        "non-member gradient norms."
    )


# ============================================================
# 9. Construct membership labels and scores
# ============================================================

# Unified MIA convention:
#
#     Member     = 1
#     Non-member = 0
#
# roc_auc_score assumes a larger score indicates
# a greater probability of class 1.
#
# Since smaller gradient norms are expected for members:
#
#     membership_score = -gradient_norm
#

y_true = np.concatenate(
    [
        np.ones(
            len(
                member_gradient_norms
            ),
            dtype=np.int64,
        ),

        np.zeros(
            len(
                non_member_gradient_norms
            ),
            dtype=np.int64,
        ),
    ]
)

y_scores = np.concatenate(
    [
        -member_gradient_norms,
        -non_member_gradient_norms,
    ]
)


# ============================================================
# 10. AUROC
# ============================================================

gradient_auc = roc_auc_score(
    y_true,
    y_scores,
)

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_scores,
)


# ============================================================
# 11. Accuracy using the best Youden-J threshold
# ============================================================

j_scores = (
    tpr - fpr
)

best_index = int(
    np.argmax(
        j_scores
    )
)

best_threshold = float(
    thresholds[
        best_index
    ]
)

y_pred = (
    y_scores
    >= best_threshold
).astype(
    np.int64
)

gradient_accuracy = accuracy_score(
    y_true,
    y_pred,
)


# ============================================================
# 12. TPR at low FPR
# ============================================================

def calculate_tpr_at_fpr(
    fpr,
    tpr,
    target_fprs,
):
    """
    Return the maximum empirical TPR achievable
    without exceeding each target FPR.
    """

    results = {}

    for target_fpr in target_fprs:
        valid_indices = (
            fpr <= target_fpr
        )

        if np.any(
            valid_indices
        ):
            value = float(
                np.max(
                    tpr[
                        valid_indices
                    ]
                )
            )

        else:
            value = 0.0

        results[
            target_fpr
        ] = value

    return results


target_fprs = [
    1e-3,
    5e-3,
    1e-2,
    5e-2,
    1e-1,
]

low_fpr_results = (
    calculate_tpr_at_fpr(
        fpr=fpr,
        tpr=tpr,
        target_fprs=target_fprs,
    )
)


# ============================================================
# 13. Print results
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Gradient-based MIA Results"
)

print(
    "=" * 70
)

print(
    f"Member samples: "
    f"{len(member_gradient_norms)}"
)

print(
    f"Non-member samples: "
    f"{len(non_member_gradient_norms)}"
)

print()

print(
    "Gradient norm statistics:"
)

print(
    f"  Member mean:       "
    f"{member_gradient_norms.mean():.8f}"
)

print(
    f"  Member median:     "
    f"{np.median(member_gradient_norms):.8f}"
)

print(
    f"  Member std:        "
    f"{member_gradient_norms.std():.8f}"
)

print(
    f"  Non-member mean:   "
    f"{non_member_gradient_norms.mean():.8f}"
)

print(
    f"  Non-member median: "
    f"{np.median(non_member_gradient_norms):.8f}"
)

print(
    f"  Non-member std:    "
    f"{non_member_gradient_norms.std():.8f}"
)

print()

print(
    f"Gradient MIA AUROC: "
    f"{gradient_auc:.6f}"
)

print(
    f"Best-threshold Accuracy: "
    f"{gradient_accuracy:.6f}"
)

print(
    f"Best membership-score threshold: "
    f"{best_threshold:.8f}"
)

print()

print(
    "Low-FPR attack performance:"
)

for target_fpr, target_tpr in (
    low_fpr_results.items()
):
    print(
        f"  TPR @ FPR <= "
        f"{target_fpr:.4g}: "
        f"{target_tpr:.6f}"
    )


# ============================================================
# 14. Standard ROC
# ============================================================

standard_roc_path = (
    OUTPUT_DIR
    / "gradient_mia_roc.png"
)

plt.figure(
    figsize=(7, 7)
)

plt.plot(
    fpr,
    tpr,
    linewidth=2.2,
    label=(
        f"Gradient MIA "
        f"(AUC = {gradient_auc:.4f})"
    ),
)

plt.plot(
    [0.0, 1.0],
    [0.0, 1.0],
    linestyle="--",
    linewidth=1.5,
    label="Random guess",
)

plt.xlim(
    0.0,
    1.0,
)

plt.ylim(
    0.0,
    1.0,
)

plt.xlabel(
    "False Positive Rate",
    fontsize=13,
)

plt.ylabel(
    "True Positive Rate",
    fontsize=13,
)

plt.title(
    "Gradient-based Membership Inference ROC",
    fontsize=14,
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.3,
)

plt.legend(
    loc="lower right",
    fontsize=11,
)

plt.tight_layout()

plt.savefig(
    standard_roc_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "\nStandard ROC saved to:"
)

print(
    standard_roc_path
)


# ============================================================
# 15. LiRA-style semilog ROC
# ============================================================

semilog_roc_path = (
    OUTPUT_DIR
    / "gradient_mia_semilog_roc.png"
)

num_non_members = int(
    np.sum(
        y_true == 0
    )
)

if num_non_members <= 0:
    raise RuntimeError(
        "No non-member samples available."
    )

# Smallest non-zero empirical FPR:
#
#     1 / N_nonmember
#
min_fpr = (
    1.0
    / num_non_members
)

# log(0) cannot be plotted.
positive_fpr_mask = (
    fpr > 0
)

fpr_plot = (
    fpr[
        positive_fpr_mask
    ]
)

tpr_plot = (
    tpr[
        positive_fpr_mask
    ]
)

random_fpr = np.geomspace(
    min_fpr,
    1.0,
    500,
)


plt.figure(
    figsize=(8, 6.5)
)

plt.semilogx(
    fpr_plot,
    tpr_plot,
    linewidth=2.2,
    label=(
        f"Gradient MIA "
        f"(AUC = {gradient_auc:.4f})"
    ),
)

plt.semilogx(
    random_fpr,
    random_fpr,
    linestyle="--",
    linewidth=1.5,
    label="Random guess",
)

plt.xlim(
    min_fpr,
    1.0,
)

plt.ylim(
    0.0,
    1.0,
)

plt.xlabel(
    "False Positive Rate",
    fontsize=13,
)

plt.ylabel(
    "True Positive Rate",
    fontsize=13,
)

plt.title(
    "Gradient-based Membership Inference ROC",
    fontsize=14,
)

plt.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.35,
)

plt.legend(
    loc="lower right",
    fontsize=11,
)

plt.tight_layout()

plt.savefig(
    semilog_roc_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "\nLiRA-style semilog ROC saved to:"
)

print(
    semilog_roc_path
)


# ============================================================
# 16. Gradient norm distribution
# ============================================================

distribution_path = (
    OUTPUT_DIR
    / "gradient_mia_distribution.png"
)

plt.figure(
    figsize=(9, 5.5)
)

plt.hist(
    member_gradient_norms,
    bins=60,
    alpha=0.5,
    label="Members",
    density=True,
)

plt.hist(
    non_member_gradient_norms,
    bins=60,
    alpha=0.5,
    label="Non-members",
    density=True,
)

plt.xlabel(
    "Classification-head Gradient L2 Norm",
    fontsize=12,
)

plt.ylabel(
    "Density",
    fontsize=12,
)

plt.title(
    "Gradient Norm Distribution",
    fontsize=14,
)

plt.legend(
    fontsize=11,
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    distribution_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "\nGradient distribution saved to:"
)

print(
    distribution_path
)


# ============================================================
# 17. Final summary
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Finished"
)

print(
    "=" * 70
)

print(
    f"AUROC: "
    f"{gradient_auc:.6f}"
)

print(
    f"Accuracy: "
    f"{gradient_accuracy:.6f}"
)

print(
    f"Scores: "
    f"{SCORE_PATH}"
)

print(
    f"ROC: "
    f"{standard_roc_path}"
)

print(
    f"Semilog ROC: "
    f"{semilog_roc_path}"
)

print(
    f"Distribution: "
    f"{distribution_path}"
)

# ============================================================
# 16. Log-log ROC
# ============================================================

loglog_roc_path = (
    OUTPUT_DIR
    / "gradient_mia_loglog_roc.png"
)

# ------------------------------------------------------
# Smallest measurable probability
# ------------------------------------------------------

num_non_members = np.sum(
    y_true == 0
)

num_members = np.sum(
    y_true == 1
)

if num_non_members <= 0:
    raise RuntimeError(
        "No non-member samples available."
    )

if num_members <= 0:
    raise RuntimeError(
        "No member samples available."
    )

min_rate = max(
    1.0 / num_non_members,
    1.0 / num_members,
)

print(
    f"\nMinimum rate for log-log ROC = "
    f"{min_rate:.6e}"
)

# ------------------------------------------------------
# log-log cannot display zero
# ------------------------------------------------------

fpr_plot_loglog = np.clip(
    fpr,
    min_rate,
    1.0,
)

tpr_plot_loglog = np.clip(
    tpr,
    min_rate,
    1.0,
)

# ------------------------------------------------------
# Random baseline
# ------------------------------------------------------

random_rates_loglog = np.logspace(
    np.log10(
        min_rate
    ),
    0,
    500,
)

# ------------------------------------------------------
# Plot
# ------------------------------------------------------

plt.figure(
    figsize=(8, 7)
)

plt.plot(
    fpr_plot_loglog,
    tpr_plot_loglog,
    linewidth=2.5,
    label=(
        f"Gradient MIA "
        f"(AUC = {gradient_auc:.4f})"
    ),
)

plt.plot(
    random_rates_loglog,
    random_rates_loglog,
    "--",
    linewidth=1.6,
    label="Random guess",
)

plt.xscale(
    "log"
)

plt.yscale(
    "log"
)

plt.xlim(
    min_rate,
    1.0,
)

plt.ylim(
    min_rate,
    1.0,
)

plt.xlabel(
    "False Positive Rate",
    fontsize=13,
)

plt.ylabel(
    "True Positive Rate",
    fontsize=13,
)

plt.title(
    "Gradient-based Membership Inference ROC",
    fontsize=15,
)

plt.grid(
    True,
    which="both",
    linestyle=":",
    linewidth=0.7,
    alpha=0.6,
)

plt.legend(
    loc="lower right",
    fontsize=11,
)

plt.tight_layout()

plt.savefig(
    loglog_roc_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "\nLog-log ROC saved to:"
)

print(
    loglog_roc_path
)
