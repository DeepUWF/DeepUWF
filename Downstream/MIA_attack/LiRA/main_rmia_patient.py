import os
import sys
import re
import csv
from pathlib import Path
from collections import defaultdict

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
        description=(
            "Run RMIA membership inference attack with image-, "
            "patient-eye-, and patient-level evaluation."
        )
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
        default="ours",
        choices=["DP1", "baseline", "DP8", "DP32", "ours"],
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

print(
    f"trained_seed:{trained_seed}, model:{model}, device:{device}, "
    f"IMAGE_SIZE:{IMAGE_SIZE}, NUM_CLASSES:{NUM_CLASSES}"
)

baseweight = "/path/to/workspace/weights/DeepUWFDP"

PRETRAINED_CHECKPOINT = (
    f"{baseweight}/pretrain_distill_ffm_checkpoint-best.pth"
)

TRAIN_ROOT = "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/train"
TEST_ROOT = "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/test"

USE_ALL_TRAIN_AS_MEMBER = True

MEMBER_LIST = (
    "/path/to/workspace/weights/DeepUWFDP/"
    "SHDR_random_attack_results_robust_subspace_moredata/"
    "trainset_0.txt"
)


# ============================================================
# Target checkpoint
# ============================================================

CHECKPOINTS = {
    0: {
        "DP1": f"{baseweight}/SHDR_ep1_adamw_seed0_alltrainset/model_standalone_ep1.pt",
        "baseline": f"{baseweight}/SHDR_NoDPBaseline_adamw_seed0/model_standalone_ep0.pt",
        "DP8": f"{baseweight}/SHDR_ep8_adamw_seed0_alltrainset/model_standalone_ep1.pt",
        "DP32": f"{baseweight}/SHDR_ep32_adamw_seed0_alltrainset/model_standalone_ep1.pt",
        "ours": (
            f"{baseweight}/"
            "SHDR_random_attack_results_robust_subspace_alltraindata_seed0/"
            "model_standalone_ep0.pt"
        ),
    },
    1: {
        "DP1": f"{baseweight}/SHDR_ep1_adamw_seed1_alltrainset/model_standalone_ep0.pt",
        "baseline": f"{baseweight}/SHDR_NoDPBaseline_adamw_seed1/model_standalone_ep0.pt",
        "DP8": f"{baseweight}/SHDR_ep8_adamw_seed1_alltrainset/model_standalone_ep0.pt",
        "DP32": f"{baseweight}/SHDR_ep32_adamw_seed1_alltrainset/model_standalone_ep0.pt",
        "ours": (
            f"{baseweight}/"
            "SHDR_random_attack_results_robust_subspace_alltraindata_seed1/"
            "model_standalone_ep0.pt"
        ),
    },
    2: {
        "DP1": f"{baseweight}/SHDR_ep1_adamw_seed2_alltrainset/model_standalone_ep1.pt",
        "baseline": f"{baseweight}/SHDR_NoDPBaseline_adamw_seed2/model_standalone_ep0.pt",
        "DP8": f"{baseweight}/SHDR_ep8_adamw_seed2_alltrainset/model_standalone_ep1.pt",
        "DP32": f"{baseweight}/SHDR_ep32_adamw_seed2_alltrainset/model_standalone_ep1.pt",
        "ours": (
            f"{baseweight}/"
            "SHDR_random_attack_results_robust_subspace_alltraindata_seed2/"
            "model_standalone_ep0.pt"
        ),
    },
    3: {
        "DP1": f"{baseweight}/SHDR_ep1_adamw_seed3_alltrainset/model_standalone_ep0.pt",
        "baseline": f"{baseweight}/SHDR_NoDPBaseline_adamw_seed3/model_standalone_ep0.pt",
        "DP8": f"{baseweight}/SHDR_ep8_adamw_seed3_alltrainset/model_standalone_ep0.pt",
        "DP32": f"{baseweight}/SHDR_ep32_adamw_seed3_alltrainset/model_standalone_ep0.pt",
        "ours": (
            f"{baseweight}/"
            "SHDR_random_attack_results_robust_subspace_alltraindata_seed3/"
            "model_standalone_ep0.pt"
        ),
    },
    4: {
        "DP1": f"{baseweight}/SHDR_ep1_adamw_seed4_alltrainset/model_standalone_ep1.pt",
        "baseline": f"{baseweight}/SHDR_NoDPBaseline_adamw_seed4/model_standalone_ep0.pt",
        "DP8": f"{baseweight}/SHDR_ep8_adamw_seed4_alltrainset/model_standalone_ep1.pt",
        "DP32": f"{baseweight}/SHDR_ep32_adamw_seed4_alltrainset/model_standalone_ep1.pt",
        "ours": (
            f"{baseweight}/"
            "SHDR_random_attack_results_robust_subspace_alltraindata_seed4/"
            "model_standalone_ep0.pt"
        ),
    },
}

if trained_seed not in CHECKPOINTS:
    raise ValueError(f"Unsupported trained_seed: {trained_seed}")

TARGET_CHECKPOINT = CHECKPOINTS[trained_seed][model]

OUTPUT_DIR = Path(
    "/path/to/workspace/code/DeepUWF/MIA_attack/experiments/"
    f"rmia_results_vitshadow_{model}_{trained_seed}_"
    "alltrainset_precise_gamma2_bestep_patientlevel"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"OUTPUT_DIR:{OUTPUT_DIR}")
print(f"TARGET_CHECKPOINT:{TARGET_CHECKPOINT}")


# ============================================================
# RMIA parameters
# ============================================================

measurement_number = 4000

perc_member = 0.0
perc_nonmember = 0.20

shadow_epochs = 5
random_sample_number = 1000
gamma = 2
shadow_train_fraction = 0.5

eval_batch_size = 16
num_workers = 2
num_shadow_models = 16

shadow_blr = 0.001
shadow_min_lr = 1e-6
shadow_warmup_epochs = 2
shadow_weight_decay = 0.05
shadow_layer_decay = 0.75
shadow_label_smoothing = 0.1
shadow_betas = (0.9, 0.999)
shadow_batch_size = 24


# ============================================================
# Patient / eye parsing
# ============================================================

def extract_patient_eye_from_path(image_path):
    """
    Examples
    --------
    005172-20200508@101552-R3-S.jpg -> patient=005172, eye=R
    027038-20210309@145706-L2-S.jpg -> patient=027038, eye=L
    """
    stem = Path(image_path).stem
    tokens = stem.split("-")

    if not tokens or not tokens[0].strip():
        raise ValueError(f"Cannot parse patient ID from: {image_path}")

    patient_id = tokens[0].strip()
    eye_id = "UNKNOWN"

    for token in tokens[1:]:
        token = token.strip().upper()

        if re.fullmatch(r"R\d*", token):
            eye_id = "R"
            break

        if re.fullmatch(r"L\d*", token):
            eye_id = "L"
            break

    return patient_id, eye_id


def get_base_dataset_path(base_dataset, index):
    if hasattr(base_dataset, "samples"):
        sample = base_dataset.samples[index]
        return str(sample[0] if isinstance(sample, (tuple, list)) else sample)

    for attr in ["image_paths", "paths", "files", "imgs"]:
        if hasattr(base_dataset, attr):
            item = getattr(base_dataset, attr)[index]
            return str(item[0] if isinstance(item, (tuple, list)) else item)

    raise RuntimeError(
        f"Could not recover image path from dataset type {type(base_dataset)}."
    )


def collect_dataset_patient_info(dataset):
    """
    Collect patient IDs and patient-eye IDs from the full dataset.

    This version works with ImageFolder and also custom datasets as
    long as get_base_dataset_path() can recover their image paths.
    """
    patient_ids = set()
    patient_eye_keys = set()
    unknown_eye_paths = []

    for idx in range(len(dataset)):
        image_path = get_base_dataset_path(dataset, idx)
        patient_id, eye_id = extract_patient_eye_from_path(image_path)

        patient_id = str(patient_id)
        eye_id = str(eye_id)

        patient_ids.add(patient_id)
        patient_eye_keys.add((patient_id, eye_id))

        if eye_id == "UNKNOWN":
            unknown_eye_paths.append(image_path)

    return patient_ids, patient_eye_keys, unknown_eye_paths


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
# Datasets
# ============================================================

if USE_ALL_TRAIN_AS_MEMBER:
    print(f"\nMember source: ALL training samples under {TRAIN_ROOT}", flush=True)
    member_base = ImageFolder(
        root=TRAIN_ROOT,
        transform=eval_transform,
    )
else:
    print("\nMember source: MEMBER_LIST", flush=True)
    print(f"Member list: {MEMBER_LIST}", flush=True)

    member_base = ImageListDataset(
        dataset_root=TRAIN_ROOT,
        list_file=MEMBER_LIST,
        transform=eval_transform,
    )

nonmember_base = ImageFolder(
    root=TEST_ROOT,
    transform=eval_transform,
)


# ============================================================
# Global patient / patient-eye membership information
# ============================================================

(
    global_train_patient_ids,
    global_train_patient_eye_keys,
    train_unknown_eye_paths,
) = collect_dataset_patient_info(member_base)

(
    global_test_patient_ids,
    global_test_patient_eye_keys,
    test_unknown_eye_paths,
) = collect_dataset_patient_info(nonmember_base)

global_overlap_patients = global_train_patient_ids & global_test_patient_ids
global_train_only_patients = global_train_patient_ids - global_test_patient_ids
global_test_only_patients = global_test_patient_ids - global_train_patient_ids

global_overlap_patient_eyes = (
    global_train_patient_eye_keys & global_test_patient_eye_keys
)
global_train_only_patient_eyes = (
    global_train_patient_eye_keys - global_test_patient_eye_keys
)
global_test_only_patient_eyes = (
    global_test_patient_eye_keys - global_train_patient_eye_keys
)

print("\n" + "=" * 70)
print("GLOBAL PATIENT SPLIT STATISTICS")
print("=" * 70)

print(f"Train patients: {len(global_train_patient_ids)}")
print(f"Test patients: {len(global_test_patient_ids)}")
print(f"Patients appearing in BOTH train/test: {len(global_overlap_patients)}")
print(f"Train-only patients: {len(global_train_only_patients)}")
print(f"Test-only patients: {len(global_test_only_patients)}")

print(
    "Patient-eyes appearing in BOTH train/test: "
    f"{len(global_overlap_patient_eyes)}"
)

print(f"Unknown-eye train images: {len(train_unknown_eye_paths)}")
print(f"Unknown-eye test images: {len(test_unknown_eye_paths)}")

if train_unknown_eye_paths:
    print("Example unknown-eye TRAIN paths:", train_unknown_eye_paths[:10])

if test_unknown_eye_paths:
    print("Example unknown-eye TEST paths:", test_unknown_eye_paths[:10])


# Standardize outputs to (image, label)
member_dataset = XYDataset(member_base)
nonmember_dataset = XYDataset(nonmember_base)

print("\n" + "=" * 70)
print("RMIA")
print("=" * 70)
print(f"USE_ALL_TRAIN_AS_MEMBER: {USE_ALL_TRAIN_AS_MEMBER}")
print(f"Member samples: {len(member_dataset)}")
print(f"Non-member samples: {len(nonmember_dataset)}")


# ============================================================
# Select auxiliary / measurement samples
# ============================================================

generator = torch.Generator().manual_seed(SEED)

member_perm = torch.randperm(
    len(member_dataset),
    generator=generator,
)

nonmember_perm = torch.randperm(
    len(nonmember_dataset),
    generator=generator,
)

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

attacker_member_dataset = Subset(
    member_dataset,
    attacker_member_indices,
)

attacker_nonmember_dataset = Subset(
    nonmember_dataset,
    attacker_nonmember_indices,
)

measurement_member_dataset = Subset(
    member_dataset,
    measurement_member_indices,
)

measurement_nonmember_dataset = Subset(
    nonmember_dataset,
    measurement_nonmember_indices,
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
        "Shadow/reference dataset is empty. "
        "Increase perc_member or perc_nonmember."
    )


# ============================================================
# Image-level membership labels
# member=1, non-member=0
# ============================================================

measurement_ref = np.concatenate([
    np.ones(len(measurement_member_dataset), dtype=np.int64),
    np.zeros(len(measurement_nonmember_dataset), dtype=np.int64),
])


# ============================================================
# Measurement paths and patient IDs
# ============================================================

measurement_member_paths = [
    get_base_dataset_path(member_base, idx)
    for idx in measurement_member_indices
]

measurement_nonmember_paths = [
    get_base_dataset_path(nonmember_base, idx)
    for idx in measurement_nonmember_indices
]

measurement_paths = measurement_member_paths + measurement_nonmember_paths

assert len(measurement_paths) == len(measurement_ref)

measurement_patient_ids = []
measurement_eye_ids = []

for path in measurement_paths:
    patient_id, eye_id = extract_patient_eye_from_path(path)
    measurement_patient_ids.append(patient_id)
    measurement_eye_ids.append(eye_id)

measurement_patient_ids = np.asarray(
    measurement_patient_ids,
    dtype=object,
)

measurement_eye_ids = np.asarray(
    measurement_eye_ids,
    dtype=object,
)


# ============================================================
# Sanity / diagnostic information
# ============================================================

print("\nFirst 20 patient/eye mappings:")

for i in range(min(20, len(measurement_paths))):
    print(
        f"{i:04d} | membership={measurement_ref[i]} | "
        f"patient={measurement_patient_ids[i]} | "
        f"eye={measurement_eye_ids[i]} | "
        f"{measurement_paths[i]}"
    )

unknown_measurement_idx = np.where(
    measurement_eye_ids == "UNKNOWN"
)[0]

print(
    f"\nMeasurement UNKNOWN eye count: "
    f"{len(unknown_measurement_idx)}"
)

if len(unknown_measurement_idx) > 0:
    print("Example UNKNOWN-eye measurement paths:")

    for idx in unknown_measurement_idx[:20]:
        print(measurement_paths[idx])


measurement_patient_membership_sets = defaultdict(set)

for pid, membership in zip(
    measurement_patient_ids,
    measurement_ref,
):
    measurement_patient_membership_sets[str(pid)].add(int(membership))

measurement_overlap_patients = [
    pid
    for pid, labels in measurement_patient_membership_sets.items()
    if len(labels) > 1
]

print("\nMeasurement subset patient composition:")
print(
    f"Unique measurement patients: "
    f"{len(measurement_patient_membership_sets)}"
)
print(
    "Patients represented by both member and non-member "
    f"measurement images: {len(measurement_overlap_patients)}"
)
print(
    f"GLOBAL train/test overlapping patients: "
    f"{len(global_overlap_patients)}"
)

print("\nAttacker knowledge:")
print(f"Known member samples: {len(attacker_member_dataset)}")
print(f"Known non-member samples: {len(attacker_nonmember_dataset)}")
print(f"Measurement members: {len(measurement_member_dataset)}")
print(f"Measurement non-members: {len(measurement_nonmember_dataset)}")

print(
    f"Measurement unique patients: "
    f"{len(np.unique(measurement_patient_ids))}"
)

print(
    "Measurement unique patient-eyes: "
    f"{len(set(zip(measurement_patient_ids, measurement_eye_ids)))}"
)

print(f"Shadow/reference pool: {len(shadow_dataset)}")


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
#
# IMPORTANT:
# This block is unchanged from the original RMIA pipeline.
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


# ============================================================
# RMIA result
# ============================================================

scores = np.asarray(
    scores,
    dtype=np.float64,
).reshape(-1)

lr_target = np.asarray(
    lr_target,
    dtype=np.float64,
).reshape(-1)

lr_random = np.asarray(
    lr_random,
    dtype=np.float64,
).reshape(-1)

if len(scores) != len(measurement_ref):
    raise RuntimeError(
        "RMIA score count does not match measurement labels."
    )

if len(lr_target) != len(scores):
    raise RuntimeError(
        "lr_target count does not match RMIA score count."
    )

member_score_mean = float(
    scores[measurement_ref == 1].mean()
)

nonmember_score_mean = float(
    scores[measurement_ref == 0].mean()
)

print(f"\nMean member score: {member_score_mean:.6f}")
print(f"Mean non-member score: {nonmember_score_mean:.6f}")


# ============================================================
# Evaluation helpers
# ============================================================

TARGET_FPRS = [
    1e-3,
    5e-3,
    1e-2,
    5e-2,
    1e-1,
]


def evaluate_level(level_name, labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if len(labels) != len(scores):
        raise RuntimeError(
            f"{level_name}: labels/scores length mismatch "
            f"({len(labels)} vs {len(scores)})."
        )

    if len(np.unique(labels)) < 2:
        raise RuntimeError(
            f"{level_name}: only one membership class is present."
        )

    if not np.all(np.isfinite(scores)):
        raise RuntimeError(
            f"{level_name}: scores contain NaN or Inf."
        )

    auc_value = float(
        roc_auc_score(labels, scores)
    )

    reverse_auc = float(
        roc_auc_score(labels, -scores)
    )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    low_fpr_metrics = {}

    for target_fpr in TARGET_FPRS:
        valid = fpr <= target_fpr
        low_fpr_metrics[target_fpr] = (
            float(np.max(tpr[valid]))
            if np.any(valid)
            else 0.0
        )

    n_members = int(np.sum(labels == 1))
    n_nonmembers = int(np.sum(labels == 0))
    min_nonzero_fpr = 1.0 / n_nonmembers

    print("\n" + "=" * 70)
    print(f"{level_name} RMIA")
    print("=" * 70)

    print(f"Number of units: {len(labels)}")
    print(f"Members: {n_members}")
    print(f"Non-members: {n_nonmembers}")
    print(f"AUROC: {auc_value:.6f}")
    print(f"Reverse-score AUROC: {reverse_auc:.6f}")
    print(
        "Minimum resolvable non-zero FPR: "
        f"{min_nonzero_fpr:.6g}"
    )

    print("Low-FPR attack performance:")

    for target_fpr, target_tpr in low_fpr_metrics.items():
        suffix = ""
        if target_fpr < min_nonzero_fpr:
            suffix = " (only FPR=0 is resolvable below this target)"

        print(
            f"TPR @ FPR <= {target_fpr:.4g}: "
            f"{target_tpr:.6f}{suffix}"
        )

    return {
        "level": level_name,
        "labels": labels,
        "scores": scores,
        "auc": auc_value,
        "reverse_auc": reverse_auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "low_fpr": low_fpr_metrics,
        "min_nonzero_fpr": min_nonzero_fpr,
    }


# ============================================================
# Multi-level RMIA evaluation
# ============================================================

# ------------------------------------------------------------
# 1. Image level
# ------------------------------------------------------------

image_result = evaluate_level(
    "Image-level",
    measurement_ref,
    scores,
)


# ------------------------------------------------------------
# 2. Patient-eye level
#
# Score:
#   max RMIA score among sampled images of a patient-eye.
#
# Membership:
#   1 if this patient-eye appears anywhere in the full train set.
# ------------------------------------------------------------

patient_eye_score_dict = defaultdict(list)
patient_eye_path_dict = defaultdict(list)

for score, patient_id, eye_id, image_path in zip(
    scores,
    measurement_patient_ids,
    measurement_eye_ids,
    measurement_paths,
):
    key = (
        str(patient_id),
        str(eye_id),
    )

    patient_eye_score_dict[key].append(
        float(score)
    )

    patient_eye_path_dict[key].append(
        str(image_path)
    )

patient_eye_keys = sorted(
    patient_eye_score_dict.keys()
)

patient_eye_scores = np.asarray(
    [
        max(patient_eye_score_dict[key])
        for key in patient_eye_keys
    ],
    dtype=np.float64,
)

patient_eye_labels = np.asarray(
    [
        int(key in global_train_patient_eye_keys)
        for key in patient_eye_keys
    ],
    dtype=np.int64,
)

patient_eye_result = evaluate_level(
    "Patient-eye-level",
    patient_eye_labels,
    patient_eye_scores,
)


# ------------------------------------------------------------
# 3. Patient level: any-member definition
#
# Score:
#   max RMIA score among sampled images of a patient.
#
# Membership:
#   1 if patient appears anywhere in full TRAIN_ROOT.
#
# Overlap patients are still regarded as members.
# ------------------------------------------------------------

patient_score_dict = defaultdict(list)
patient_path_dict = defaultdict(list)
patient_eye_set_dict = defaultdict(set)

for score, patient_id, eye_id, image_path in zip(
    scores,
    measurement_patient_ids,
    measurement_eye_ids,
    measurement_paths,
):
    patient_id = str(patient_id)

    patient_score_dict[patient_id].append(
        float(score)
    )

    patient_path_dict[patient_id].append(
        str(image_path)
    )

    patient_eye_set_dict[patient_id].add(
        str(eye_id)
    )

patient_ids = sorted(
    patient_score_dict.keys()
)

patient_scores = np.asarray(
    [
        max(patient_score_dict[pid])
        for pid in patient_ids
    ],
    dtype=np.float64,
)

patient_labels = np.asarray(
    [
        int(pid in global_train_patient_ids)
        for pid in patient_ids
    ],
    dtype=np.int64,
)

patient_result = evaluate_level(
    "Patient-level any-member",
    patient_labels,
    patient_scores,
)


# ------------------------------------------------------------
# 4. Strict patient-disjoint level
#
# train-only patient -> member
# test-only patient  -> non-member
# overlap patient    -> excluded
#
# This is the cleaner patient-level evaluation.
# ------------------------------------------------------------

strict_patient_ids = []
strict_patient_scores = []
strict_patient_labels = []

for pid, score in zip(
    patient_ids,
    patient_scores,
):
    if pid in global_train_only_patients:
        strict_patient_ids.append(pid)
        strict_patient_scores.append(float(score))
        strict_patient_labels.append(1)

    elif pid in global_test_only_patients:
        strict_patient_ids.append(pid)
        strict_patient_scores.append(float(score))
        strict_patient_labels.append(0)

strict_patient_ids = np.asarray(
    strict_patient_ids,
    dtype=object,
)

strict_patient_scores = np.asarray(
    strict_patient_scores,
    dtype=np.float64,
)

strict_patient_labels = np.asarray(
    strict_patient_labels,
    dtype=np.int64,
)

print("\nStrict patient-disjoint subset:")
print(f"Patients: {len(strict_patient_labels)}")
print(f"Members: {np.sum(strict_patient_labels == 1)}")
print(f"Non-members: {np.sum(strict_patient_labels == 0)}")

if len(np.unique(strict_patient_labels)) == 2:
    strict_patient_result = evaluate_level(
        "Patient-level strict-disjoint",
        strict_patient_labels,
        strict_patient_scores,
    )
else:
    strict_patient_result = None

    print(
        "WARNING: strict patient-level subset does not contain "
        "both membership classes."
    )


# ============================================================
# CSV output
# ============================================================

def write_csv(path, fieldnames, rows):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Image-level CSV
# ------------------------------------------------------------

image_rows = []

for i, (
    path,
    patient_id,
    eye_id,
    label,
    score,
) in enumerate(
    zip(
        measurement_paths,
        measurement_patient_ids,
        measurement_eye_ids,
        measurement_ref,
        scores,
    )
):
    patient_id = str(patient_id)
    eye_id = str(eye_id)

    patient_split_type = (
        "train_test_overlap"
        if patient_id in global_overlap_patients
        else "train_only"
        if patient_id in global_train_only_patients
        else "test_only"
        if patient_id in global_test_only_patients
        else "unknown"
    )

    eye_key = (
        patient_id,
        eye_id,
    )

    patient_eye_split_type = (
        "train_test_overlap"
        if eye_key in global_overlap_patient_eyes
        else "train_only"
        if eye_key in global_train_only_patient_eyes
        else "test_only"
        if eye_key in global_test_only_patient_eyes
        else "unknown"
    )

    image_rows.append({
        "sample_index": i,
        "image_path": path,
        "patient_id": patient_id,
        "eye_id": eye_id,
        "image_membership": int(label),
        "rmia_score": float(score),
        "patient_split_type": patient_split_type,
        "patient_eye_split_type": patient_eye_split_type,
    })

write_csv(
    OUTPUT_DIR / "image_level_rmia_scores.csv",
    [
        "sample_index",
        "image_path",
        "patient_id",
        "eye_id",
        "image_membership",
        "rmia_score",
        "patient_split_type",
        "patient_eye_split_type",
    ],
    image_rows,
)


# ------------------------------------------------------------
# Patient-eye-level CSV
# ------------------------------------------------------------

patient_eye_rows = []

for key, aggregated_score, membership in zip(
    patient_eye_keys,
    patient_eye_scores,
    patient_eye_labels,
):
    patient_id, eye_id = key
    raw_scores = patient_eye_score_dict[key]
    paths = patient_eye_path_dict[key]

    max_index = int(
        np.argmax(raw_scores)
    )

    if key in global_overlap_patient_eyes:
        split_type = "train_test_overlap"
    elif key in global_train_only_patient_eyes:
        split_type = "train_only"
    elif key in global_test_only_patient_eyes:
        split_type = "test_only"
    else:
        split_type = "unknown"

    patient_eye_rows.append({
        "patient_id": patient_id,
        "eye_id": eye_id,
        "membership": int(membership),
        "split_type": split_type,
        "n_images": len(raw_scores),
        "patient_eye_score_max": float(aggregated_score),
        "patient_eye_score_mean": float(np.mean(raw_scores)),
        "patient_eye_score_q95": float(np.quantile(raw_scores, 0.95)),
        "max_score_image_path": paths[max_index],
    })

write_csv(
    OUTPUT_DIR / "patient_eye_level_rmia_scores.csv",
    [
        "patient_id",
        "eye_id",
        "membership",
        "split_type",
        "n_images",
        "patient_eye_score_max",
        "patient_eye_score_mean",
        "patient_eye_score_q95",
        "max_score_image_path",
    ],
    patient_eye_rows,
)


# ------------------------------------------------------------
# Patient-level CSV: any-member definition
# ------------------------------------------------------------

patient_rows = []

for patient_id, aggregated_score, membership in zip(
    patient_ids,
    patient_scores,
    patient_labels,
):
    raw_scores = patient_score_dict[patient_id]
    paths = patient_path_dict[patient_id]

    max_index = int(
        np.argmax(raw_scores)
    )

    if patient_id in global_overlap_patients:
        split_type = "train_test_overlap"
    elif patient_id in global_train_only_patients:
        split_type = "train_only"
    elif patient_id in global_test_only_patients:
        split_type = "test_only"
    else:
        split_type = "unknown"

    patient_rows.append({
        "patient_id": patient_id,
        "membership": int(membership),
        "split_type": split_type,
        "n_images": len(raw_scores),
        "n_eyes": len(patient_eye_set_dict[patient_id]),
        "eyes": ";".join(
            sorted(patient_eye_set_dict[patient_id])
        ),
        "patient_score_max": float(aggregated_score),
        "patient_score_mean": float(np.mean(raw_scores)),
        "patient_score_q95": float(np.quantile(raw_scores, 0.95)),
        "max_score_image_path": paths[max_index],
    })

write_csv(
    OUTPUT_DIR / "patient_level_rmia_scores.csv",
    [
        "patient_id",
        "membership",
        "split_type",
        "n_images",
        "n_eyes",
        "eyes",
        "patient_score_max",
        "patient_score_mean",
        "patient_score_q95",
        "max_score_image_path",
    ],
    patient_rows,
)


# ------------------------------------------------------------
# Strict patient-disjoint CSV
# ------------------------------------------------------------

strict_patient_rows = []

for pid, score, label in zip(
    strict_patient_ids,
    strict_patient_scores,
    strict_patient_labels,
):
    raw_scores = patient_score_dict[str(pid)]
    paths = patient_path_dict[str(pid)]

    max_index = int(
        np.argmax(raw_scores)
    )

    strict_patient_rows.append({
        "patient_id": str(pid),
        "membership": int(label),
        "split_type": (
            "train_only"
            if int(label) == 1
            else "test_only"
        ),
        "n_images": len(raw_scores),
        "n_eyes": len(
            patient_eye_set_dict[str(pid)]
        ),
        "eyes": ";".join(
            sorted(
                patient_eye_set_dict[str(pid)]
            )
        ),
        "patient_score_max": float(score),
        "patient_score_mean": float(
            np.mean(raw_scores)
        ),
        "patient_score_q95": float(
            np.quantile(raw_scores, 0.95)
        ),
        "max_score_image_path": paths[max_index],
    })

write_csv(
    OUTPUT_DIR / "patient_level_strict_disjoint_rmia_scores.csv",
    [
        "patient_id",
        "membership",
        "split_type",
        "n_images",
        "n_eyes",
        "eyes",
        "patient_score_max",
        "patient_score_mean",
        "patient_score_q95",
        "max_score_image_path",
    ],
    strict_patient_rows,
)


# ============================================================
# Summary CSV
# ============================================================

results_for_summary = [
    image_result,
    patient_eye_result,
    patient_result,
]

if strict_patient_result is not None:
    results_for_summary.append(
        strict_patient_result
    )

summary_rows = []

for result in results_for_summary:
    row = {
        "level": result["level"],
        "n_units": len(result["labels"]),
        "n_members": int(
            np.sum(result["labels"] == 1)
        ),
        "n_nonmembers": int(
            np.sum(result["labels"] == 0)
        ),
        "auroc": result["auc"],
        "reverse_auroc": result["reverse_auc"],
        "min_nonzero_fpr": result["min_nonzero_fpr"],
    }

    for target_fpr in TARGET_FPRS:
        row[
            f"tpr_at_fpr_le_{target_fpr}"
        ] = result["low_fpr"][target_fpr]

    summary_rows.append(row)

summary_fieldnames = [
    "level",
    "n_units",
    "n_members",
    "n_nonmembers",
    "auroc",
    "reverse_auroc",
    "min_nonzero_fpr",
] + [
    f"tpr_at_fpr_le_{fpr_value}"
    for fpr_value in TARGET_FPRS
]

write_csv(
    OUTPUT_DIR / "rmia_multilevel_summary.csv",
    summary_fieldnames,
    summary_rows,
)


# ============================================================
# Save NPZ
# ============================================================

np.savez(
    OUTPUT_DIR / "rmia_scores_multilevel.npz",

    image_scores=scores,
    image_labels=measurement_ref,
    image_paths=np.asarray(
        measurement_paths,
        dtype=object,
    ),
    image_patient_ids=measurement_patient_ids,
    image_eye_ids=measurement_eye_ids,

    patient_eye_scores=patient_eye_scores,
    patient_eye_labels=patient_eye_labels,
    patient_eye_ids=np.asarray(
        patient_eye_keys,
        dtype=object,
    ),

    patient_scores=patient_scores,
    patient_labels=patient_labels,
    patient_ids=np.asarray(
        patient_ids,
        dtype=object,
    ),

    strict_patient_scores=strict_patient_scores,
    strict_patient_labels=strict_patient_labels,
    strict_patient_ids=strict_patient_ids,

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

def plot_standard_roc(result, save_path):
    plt.figure(figsize=(7, 7))

    plt.plot(
        result["fpr"],
        result["tpr"],
        linewidth=2.2,
        label=(
            f'{result["level"]} '
            f'(AUC = {result["auc"]:.4f})'
        ),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=1.5,
        label="Random guess",
    )

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'{result["level"]} RMIA ROC')

    plt.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_semilog_roc(result, save_path):
    fpr = result["fpr"]
    tpr = result["tpr"]

    min_fpr = result["min_nonzero_fpr"]
    valid = fpr > 0

    if not np.any(valid):
        print(
            f'WARNING: {result["level"]} semilog ROC skipped: '
            "no positive FPR points."
        )
        return

    random_fpr = np.geomspace(
        min_fpr,
        1.0,
        500,
    )

    plt.figure(figsize=(8, 6.5))

    plt.semilogx(
        fpr[valid],
        tpr[valid],
        linewidth=2.2,
        label=(
            f'{result["level"]} '
            f'(AUC = {result["auc"]:.4f})'
        ),
    )

    plt.semilogx(
        random_fpr,
        random_fpr,
        "--",
        linewidth=1.5,
        label="Random guess",
    )

    plt.xlim(min_fpr, 1.0)
    plt.ylim(0.0, 1.0)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'{result["level"]} RMIA ROC')

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.35,
    )

    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_loglog_roc(result, save_path):
    fpr = result["fpr"]
    tpr = result["tpr"]

    labels = result["labels"]

    n_members = int(
        np.sum(labels == 1)
    )
    n_nonmembers = int(
        np.sum(labels == 0)
    )

    min_rate = max(
        1.0 / n_members,
        1.0 / n_nonmembers,
    )

    valid = (
        (fpr > 0)
        &
        (tpr > 0)
    )

    if not np.any(valid):
        print(
            f'WARNING: {result["level"]} log-log ROC skipped: '
            "no valid positive ROC points."
        )
        return

    fpr_plot = fpr[valid]
    tpr_plot = tpr[valid]

    random_rates = np.geomspace(
        min_rate,
        1.0,
        500,
    )

    plt.figure(figsize=(8, 7))

    plt.plot(
        fpr_plot,
        tpr_plot,
        linewidth=2.2,
        label=(
            f'{result["level"]} '
            f'(AUC = {result["auc"]:.4f})'
        ),
    )

    plt.plot(
        random_rates,
        random_rates,
        "--",
        linewidth=1.5,
        label="Random guess",
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlim(min_rate, 1.0)
    plt.ylim(min_rate, 1.0)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'{result["level"]} RMIA ROC')

    plt.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.7,
        alpha=0.6,
    )

    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# Individual ROC families
# ============================================================

results_for_plots = [
    (image_result, "image_level"),
    (patient_eye_result, "patient_eye_level"),
    (patient_result, "patient_level_any_member"),
]

if strict_patient_result is not None:
    results_for_plots.append(
        (
            strict_patient_result,
            "patient_level_strict_disjoint",
        )
    )

for result, prefix in results_for_plots:
    plot_standard_roc(
        result,
        OUTPUT_DIR / f"{prefix}_rmia_roc.png",
    )

    plot_semilog_roc(
        result,
        OUTPUT_DIR / f"{prefix}_rmia_semilog_roc.png",
    )

    plot_loglog_roc(
        result,
        OUTPUT_DIR / f"{prefix}_rmia_loglog_roc.png",
    )


# ============================================================
# Combined multi-level ROC
# ============================================================

plt.figure(figsize=(8, 7))

for result, _ in results_for_plots:
    plt.plot(
        result["fpr"],
        result["tpr"],
        linewidth=2.2,
        label=(
            f'{result["level"]} '
            f'(AUC={result["auc"]:.4f})'
        ),
    )

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    linewidth=1.5,
    label="Random guess",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "RMIA Privacy Leakage at Different Aggregation Levels"
)

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)

plt.grid(
    True,
    linestyle="--",
    alpha=0.3,
)

plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rmia_multilevel_roc_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Patient aggregation sensitivity analysis
# ============================================================

patient_mean_scores = np.asarray(
    [
        np.mean(patient_score_dict[pid])
        for pid in patient_ids
    ],
    dtype=np.float64,
)

patient_q95_scores = np.asarray(
    [
        np.quantile(
            patient_score_dict[pid],
            0.95,
        )
        for pid in patient_ids
    ],
    dtype=np.float64,
)

patient_auc_mean = roc_auc_score(
    patient_labels,
    patient_mean_scores,
)

patient_auc_q95 = roc_auc_score(
    patient_labels,
    patient_q95_scores,
)

print("\nPatient aggregation sensitivity analysis:")
print(
    f"Patient AUC (MAX):  "
    f"{patient_result['auc']:.6f}"
)
print(
    f"Patient AUC (MEAN): "
    f"{patient_auc_mean:.6f}"
)
print(
    f"Patient AUC (Q95):  "
    f"{patient_auc_q95:.6f}"
)


# ============================================================
# Strict patient aggregation sensitivity analysis
# ============================================================

if strict_patient_result is not None:
    strict_patient_mean_scores = np.asarray(
        [
            np.mean(
                patient_score_dict[str(pid)]
            )
            for pid in strict_patient_ids
        ],
        dtype=np.float64,
    )

    strict_patient_q95_scores = np.asarray(
        [
            np.quantile(
                patient_score_dict[str(pid)],
                0.95,
            )
            for pid in strict_patient_ids
        ],
        dtype=np.float64,
    )

    strict_patient_auc_mean = roc_auc_score(
        strict_patient_labels,
        strict_patient_mean_scores,
    )

    strict_patient_auc_q95 = roc_auc_score(
        strict_patient_labels,
        strict_patient_q95_scores,
    )

    print(
        "\nStrict patient-disjoint aggregation "
        "sensitivity analysis:"
    )

    print(
        f"Strict patient AUC (MAX):  "
        f"{strict_patient_result['auc']:.6f}"
    )

    print(
        f"Strict patient AUC (MEAN): "
        f"{strict_patient_auc_mean:.6f}"
    )

    print(
        f"Strict patient AUC (Q95):  "
        f"{strict_patient_auc_q95:.6f}"
    )


# ============================================================
# Gamma sensitivity analysis
# ============================================================

gammas = [
    2.0,
]

relative_ratio = (
    lr_target[:, None]
    /
    np.maximum(
        lr_random[None, :],
        1e-12,
    )
)

print("\nGamma sensitivity analysis:")

for gamma_test in gammas:
    scores_gamma = np.mean(
        relative_ratio > gamma_test,
        axis=1,
    )

    auc_gamma = roc_auc_score(
        measurement_ref,
        scores_gamma,
    )

    print(
        f"gamma={gamma_test:.1f}: "
        f"Image-level AUC={auc_gamma:.6f}"
    )


# ============================================================
# Final summary
# ============================================================

print("\n" + "#" * 70)
print("FINAL MULTI-LEVEL RMIA SUMMARY")
print("#" * 70)

print(
    f"Image-level AUROC:                  "
    f"{image_result['auc']:.6f}"
)

print(
    f"Patient-eye-level AUROC:            "
    f"{patient_eye_result['auc']:.6f}"
)

print(
    f"Patient-level any-member AUROC:     "
    f"{patient_result['auc']:.6f}"
)

if strict_patient_result is not None:
    print(
        f"Patient-level strict-disjoint AUROC:"
        f" {strict_patient_result['auc']:.6f}"
    )

print("\nOutputs saved to:")
print(OUTPUT_DIR)

print(
    "\nRMIA evaluation finished.",
    flush=True,
)