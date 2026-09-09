"""Run the RMIA attack with a ViT target model."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import ConcatDataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ============================================================
# Project imports
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DP.eyedataset import ImageListDataset
from model import load_model_from_checkpoint
from rmia.rmia_val import RMIA, XYDataset

# ============================================================
# Configuration
# ============================================================

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RMIA membership inference attack."
    )
    parser.add_argument(
        "--trainedSeed",
        type=int,
        default=0,
        help="Seed of the trained model to attack.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DP1",
        choices=["DP1",  "DP8", "DP32", "ours"],
        help="Model variant to attack.",
    )
    return parser.parse_args()

args = parse_args()
trained_seed = args.trainedSeed
model = args.model

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 448
NUM_CLASSES = 2
print(f"trained_seed:{trained_seed}, model:{model}, device:{device}, IMAGE_SIZE:{IMAGE_SIZE}, NUM_CLASSES:{NUM_CLASSES}")
baseweight = "/path/to/workspace/weights/DeepUWFDP"
PRETRAINED_CHECKPOINT = f"{baseweight}/pretrain_distill_ffm_checkpoint-best.pth"


TRAIN_ROOT = "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/train"
TEST_ROOT = "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/test"

# ============================================================
# Member dataset configuration
# ============================================================

# False:
#     Only samples listed in MEMBER_LIST are regarded as members.
#
# True:
#     ALL images under TRAIN_ROOT are regarded as members.
#
USE_ALL_TRAIN_AS_MEMBER = True

MEMBER_LIST = (
    "/path/to/workspace/weights/DeepUWFDP/"
    "SHDR_random_attack_results_robust_subspace_moredata/trainset_0.txt"
)


############以下是all trainset

if trained_seed == 0: #all ok ep
    if model == "DP1":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep1_adamw_seed0_alltrainset/model_standalone_ep1.pt"
        )
    elif model == "DP8":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep8_adamw_seed0_alltrainset/model_standalone_ep1.pt"
        )
    elif model == "DP32":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep32_adamw_seed0_alltrainset/model_standalone_ep1.pt"
        )
    elif model == "ours":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_random_attack_results_robust_subspace_alltraindata_seed0/model_standalone_ep0.pt"
        )
elif trained_seed == 1:#
    if model == "DP1":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep1_adamw_seed1_alltrainset/model_standalone_ep0.pt"
        )
    elif model == "DP8":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep8_adamw_seed1_alltrainset/model_standalone_ep0.pt"
        )
    elif model == "DP32":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep32_adamw_seed1_alltrainset/model_standalone_ep0.pt"
        )
    elif model == "ours":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_random_attack_results_robust_subspace_alltraindata_seed1/model_standalone_ep0.pt"
        )
elif trained_seed == 2:#
    if model == "DP1":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep1_adamw_seed2_alltrainset/model_standalone_ep1.pt"
        )
    elif model == "DP8":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep8_adamw_seed2_alltrainset/model_standalone_ep1.pt"
        )
    elif model == "DP32":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep32_adamw_seed2_alltrainset/model_standalone_ep1.pt"
        )
    elif model == "ours":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_random_attack_results_robust_subspace_alltraindata_seed2/model_standalone_ep0.pt"
        )
elif trained_seed == 3:#
    if model == "DP1":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep1_adamw_seed3_alltrainset/model_standalone_ep0.pt"
        )
    elif model == "DP8":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep8_adamw_seed3_alltrainset/model_standalone_ep0.pt"
        )
    elif model == "DP32":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep32_adamw_seed3_alltrainset/model_standalone_ep0.pt"
        )
    elif model == "ours":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_random_attack_results_robust_subspace_alltraindata_seed3/model_standalone_ep0.pt"
        )

elif trained_seed == 4:
    if model == "DP1":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep1_adamw_seed4_alltrainset/model_standalone_ep1.pt"
        )
    elif model == "DP8":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep8_adamw_seed4_alltrainset/model_standalone_ep1.pt"
        )
    elif model == "DP32":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_ep32_adamw_seed4_alltrainset/model_standalone_ep1.pt"
        )
    elif model == "ours":
        TARGET_CHECKPOINT = (
            f"{baseweight}/SHDR_random_attack_results_robust_subspace_alltraindata_seed4/model_standalone_ep0.pt"
        )


OUTPUT_DIR = Path(
    f"/path/to/workspace/code/DeepUWF/MIA_attack/experiments/rmia_results_vitshadow_{model}_{trained_seed}_alltrainset_precise_gamma2_bestep"
)

print(f"OUTPUT_DIR:{OUTPUT_DIR},TARGET_CHECKPOINT:{TARGET_CHECKPOINT}")
# Start small for a sanity check, then increase.
#measurement_number = 1000
measurement_number = 4000
perc_member = 0.0
perc_nonmember = 0.20
shadow_epochs = 5
lr_shadow_model = 1e-3
#random_sample_number = 200
random_sample_number = 1000
gamma = 2
shadow_train_fraction = 0.5
eval_batch_size = 16
num_workers = 2


#num_shadow_models = 8
num_shadow_models = 16


shadow_blr = 0.001
shadow_min_lr = 1e-6
shadow_warmup_epochs = 2

shadow_weight_decay = 0.05
shadow_layer_decay = 0.75
shadow_label_smoothing = 0.1
shadow_betas = (0.9, 0.999)




shadow_batch_size = 24


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Transform
# ============================================================
eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ============================================================
# Datasets -- NO full-image materialization
# ============================================================
# member_base = ImageListDataset(
#     dataset_root=TRAIN_ROOT,
#     list_file=MEMBER_LIST,
#     transform=eval_transform,
# )
# nonmember_base = ImageFolder(
#     root=TEST_ROOT,
#     transform=eval_transform,
# )

# # Strip the path field from ImageListDataset so all datasets yield (x, y).
# member_dataset = XYDataset(member_base)
# nonmember_dataset = XYDataset(nonmember_base)


# ============================================================
# Datasets -- NO full-image materialization
# ============================================================

# ------------------------------------------------------------
# Member dataset
# ------------------------------------------------------------
#
# USE_ALL_TRAIN_AS_MEMBER = False:
#     Use only the samples listed in MEMBER_LIST.
#
# USE_ALL_TRAIN_AS_MEMBER = True:
#     Treat the entire TRAIN_ROOT dataset as members.
#
# Both branches are wrapped by XYDataset so that downstream
# RMIA code always receives exactly:
#
#     (image, label)
#
# regardless of whether the underlying dataset additionally
# returns a path.
# ------------------------------------------------------------
######uncomment from below
if USE_ALL_TRAIN_AS_MEMBER:

    print(
        "\nMember source: ALL training samples "
        f"under {TRAIN_ROOT}",
        flush=True,
    )

    member_base = ImageFolder(
        root=TRAIN_ROOT,
        transform=eval_transform,
    )

else:

    print(
        "\nMember source: MEMBER_LIST",
        flush=True,
    )

    print(
        f"Member list: {MEMBER_LIST}",
        flush=True,
    )

    member_base = ImageListDataset(
        dataset_root=TRAIN_ROOT,
        list_file=MEMBER_LIST,
        transform=eval_transform,
    )


# ------------------------------------------------------------
# Non-member dataset
# ------------------------------------------------------------

nonmember_base = ImageFolder(
    root=TEST_ROOT,
    transform=eval_transform,
)


# ------------------------------------------------------------
# Standardize output to (image, label)
# ------------------------------------------------------------

member_dataset = XYDataset(
    member_base
)

nonmember_dataset = XYDataset(
    nonmember_base
)

print("=" * 70, flush=True)
print("RMIA", flush=True)
print("=" * 70, flush=True)
print(f"Member samples: {len(member_dataset)}", flush=True)
print(f"Non-member samples: {len(nonmember_dataset)}", flush=True)

print("=" * 70, flush=True)
print("RMIA", flush=True)
print("=" * 70, flush=True)

print(
    f"USE_ALL_TRAIN_AS_MEMBER: "
    f"{USE_ALL_TRAIN_AS_MEMBER}",
    flush=True,
)

print(
    f"Member samples: "
    f"{len(member_dataset)}",
    flush=True,
)

print(
    f"Non-member samples: "
    f"{len(nonmember_dataset)}",
    flush=True,
)

if USE_ALL_TRAIN_AS_MEMBER:
    print(
        "Member definition: all samples in TRAIN_ROOT.",
        flush=True,
    )
else:
    print(
        f"Member definition: samples listed in {MEMBER_LIST}.",
        flush=True,
    )

# ============================================================
# Select attacker auxiliary data and measurement data by INDEX.
# Images are loaded only when a DataLoader requests them.
# ============================================================
generator = torch.Generator().manual_seed(SEED)
member_perm = torch.randperm(len(member_dataset), generator=generator)
nonmember_perm = torch.randperm(len(nonmember_dataset), generator=generator)

num_attacker_member = int(perc_member * len(member_dataset))
num_attacker_nonmember = int(perc_nonmember * len(nonmember_dataset))

member_measure_start = num_attacker_member
member_measure_end = member_measure_start + measurement_number
nonmember_measure_start = num_attacker_nonmember
nonmember_measure_end = nonmember_measure_start + measurement_number

if member_measure_end > len(member_dataset):
    raise ValueError("Not enough member samples for measurement_number.")
if nonmember_measure_end > len(nonmember_dataset):
    raise ValueError("Not enough non-member samples for measurement_number.")

attacker_member_indices = member_perm[:num_attacker_member].tolist()
attacker_nonmember_indices = nonmember_perm[:num_attacker_nonmember].tolist()
measurement_member_indices = member_perm[
    member_measure_start:member_measure_end
].tolist()
measurement_nonmember_indices = nonmember_perm[
    nonmember_measure_start:nonmember_measure_end
].tolist()

attacker_member_dataset = Subset(member_dataset, attacker_member_indices)
attacker_nonmember_dataset = Subset(nonmember_dataset, attacker_nonmember_indices)
measurement_member_dataset = Subset(member_dataset, measurement_member_indices)
measurement_nonmember_dataset = Subset(
    nonmember_dataset, measurement_nonmember_indices
)

shadow_dataset = ConcatDataset([
    attacker_member_dataset,
    attacker_nonmember_dataset,
])
measurement_dataset = ConcatDataset([
    measurement_member_dataset,
    measurement_nonmember_dataset,
])

if len(shadow_dataset) == 0:
    raise RuntimeError(
        "Shadow/reference dataset is empty. Increase perc_member or perc_nonmember."
    )

# Unified MIA convention: member=1, non-member=0.
measurement_ref = np.concatenate([
    np.ones(len(measurement_member_dataset), dtype=np.int64),
    np.zeros(len(measurement_nonmember_dataset), dtype=np.int64),
])

print("\nAttacker knowledge:", flush=True)
print(f"Known member samples: {len(attacker_member_dataset)}", flush=True)
print(f"Known non-member samples: {len(attacker_nonmember_dataset)}", flush=True)
print(f"Measurement members: {len(measurement_member_dataset)}", flush=True)
print(f"Measurement non-members: {len(measurement_nonmember_dataset)}", flush=True)
print(f"Shadow/reference pool: {len(shadow_dataset)}", flush=True)
print("No full SHDR image tensors are materialized in CPU RAM.", flush=True)

# ============================================================
# Target model
# ============================================================
target_model = load_model_from_checkpoint(
    checkpoint_path=TARGET_CHECKPOINT,
    num_classes=NUM_CLASSES,
    img_size=IMAGE_SIZE,
    device=device,
)
target_model.eval()

# ============================================================
# Run RMIA
# ============================================================


scores, lr_target, lr_random = RMIA(
    target_dataset=measurement_dataset,
    target_model=target_model,
    shadow_dataset=shadow_dataset,

    pretrained_path=PRETRAINED_CHECKPOINT,

    num_classes=NUM_CLASSES,
    model_name="vit_large_patch16",
    img_size=IMAGE_SIZE,

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
    gamma=gamma,

    shadow_train_fraction=shadow_train_fraction,

    shadow_batch_size=shadow_batch_size,
    eval_batch_size=eval_batch_size,

    num_workers=num_workers,

    seed=SEED,
    device=device,
)


scores = np.asarray(scores, dtype=np.float64).reshape(-1)
if len(scores) != len(measurement_ref):
    raise RuntimeError("RMIA score count does not match measurement labels.")

member_score_mean = float(scores[measurement_ref == 1].mean())
nonmember_score_mean = float(scores[measurement_ref == 0].mean())
print(f"\nMean member score: {member_score_mean:.6f}", flush=True)
print(f"Mean non-member score: {nonmember_score_mean:.6f}", flush=True)

roc_auc = roc_auc_score(measurement_ref, scores)
fpr, tpr, thresholds = roc_curve(
    measurement_ref, scores, pos_label=1
)
print(f"RMIA AUROC: {roc_auc:.6f}", flush=True)

reverse_auc = roc_auc_score(
    measurement_ref,
    -scores,
)

print(f"Reverse-score AUROC: {reverse_auc:.6f}")

# ============================================================
# Low-FPR metrics
# ============================================================
target_fprs = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
print("\nLow-FPR attack performance:", flush=True)
for target_fpr in target_fprs:
    valid = fpr <= target_fpr
    target_tpr = float(np.max(tpr[valid])) if np.any(valid) else 0.0
    print(
        f"TPR @ FPR <= {target_fpr:.4g}: {target_tpr:.6f}",
        flush=True,
    )

# np.savez(
#     OUTPUT_DIR / "rmia_scores.npz",
#     scores=scores,
#     membership_labels=measurement_ref,
# )

np.savez(
    OUTPUT_DIR / "rmia_scores.npz",

    scores=scores,
    membership_labels=measurement_ref,

    lr_target=lr_target,
    lr_random=lr_random,

    gamma=np.asarray(
        [gamma],
        dtype=np.float64,
    ),
)


# ============================================================
# Plot helpers
# ============================================================
def plot_standard_roc(fpr, tpr, auc_value, save_path):
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, linewidth=2.2, label=f"RMIA (AUC = {auc_value:.4f})")
    plt.plot([0, 1], [0, 1], "--", linewidth=1.5, label="Random guess")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RMIA Membership Inference ROC")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_semilog_roc(fpr, tpr, auc_value, num_non_members, save_path):
    min_fpr = 1.0 / num_non_members
    mask = fpr > 0
    random_fpr = np.geomspace(min_fpr, 1.0, 500)

    plt.figure(figsize=(8, 6.5))
    plt.semilogx(
        fpr[mask], tpr[mask], linewidth=2.2,
        label=f"RMIA (AUC = {auc_value:.4f})",
    )
    plt.semilogx(
        random_fpr, random_fpr, "--", linewidth=1.5,
        label="Random guess",
    )
    plt.xlim(min_fpr, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RMIA Membership Inference ROC")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loglog_roc(
    fpr, tpr, auc_value, num_members, num_non_members, save_path
):
    min_rate = max(1.0 / num_members, 1.0 / num_non_members)
    fpr_plot = np.clip(fpr, min_rate, 1.0)
    tpr_plot = np.clip(tpr, min_rate, 1.0)
    random_rates = np.geomspace(min_rate, 1.0, 500)

    plt.figure(figsize=(8, 7))
    plt.plot(
        fpr_plot, tpr_plot, linewidth=2.2,
        label=f"RMIA (AUC = {auc_value:.4f})",
    )
    plt.plot(
        random_rates, random_rates, "--", linewidth=1.5,
        label="Random guess",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(min_rate, 1.0)
    plt.ylim(min_rate, 1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RMIA Membership Inference ROC")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


num_members = int(np.sum(measurement_ref == 1))
num_non_members = int(np.sum(measurement_ref == 0))

plot_standard_roc(
    fpr, tpr, roc_auc, OUTPUT_DIR / "rmia_roc.png"
)
plot_semilog_roc(
    fpr, tpr, roc_auc, num_non_members,
    OUTPUT_DIR / "rmia_semilog_roc.png",
)
plot_loglog_roc(
    fpr, tpr, roc_auc, num_members, num_non_members,
    OUTPUT_DIR / "rmia_loglog_roc.png",
)

print("\nRMIA evaluation finished.", flush=True)


gammas = [
    2.0,
]


relative_ratio = (
    lr_target[:, None]
    / np.maximum(
        lr_random[None, :],
        1e-12,
    )
)

for gamma in gammas:

    scores_gamma = np.mean(
        relative_ratio > gamma,
        axis=1,
    )

    auc_gamma = roc_auc_score(
        measurement_ref,
        scores_gamma,
    )

    print(
        f"gamma={gamma:.1f}: "
        f"AUC={auc_gamma:.6f}"
    )
