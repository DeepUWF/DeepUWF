import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve


# ============================================================
# 1. Configuration
# ============================================================

ROOT = (
        "/path/to/workspace/code/DeepUWF/"
    "MIA_attack/experiments"
)

# IMPORTANT:
# These should be the SAME train/test roots used in the RMIA
# patient-level evaluation.
TRAIN_ROOT = (
    "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/train"
)

TEST_ROOT = (
    "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/test"
)

# Change the seed here if needed.
TRAINED_SEED = 0


experiments = {
    "Ours": (
        "rmia_results_vitshadow_ours_0_alltrainset_precise_gamma2_bestep_patientlevel"
    ),

    "DP-Privacy EP1": (
        "rmia_results_vitshadow_DP1_0_alltrainset_precise_gamma2_bestep_patientlevel"
    ),

    "DP-Privacy EP8": (
        "rmia_results_vitshadow_DP8_0_alltrainset_precise_gamma2_bestep_patientlevel"
    ),

    "DP-Privacy EP32": (
        "rmia_results_vitshadow_DP32_0_alltrainset_precise_gamma2_bestep_patientlevel"
    ),
}


# ============================================================
# 2. Patient ID parser
#
# Must match the parser used in the RMIA script.
#
# Example:
#   005172-20200508@101552-R3-S.jpg
#       patient = 005172
# ============================================================

def extract_patient_id(image_path):
    stem = Path(image_path).stem
    tokens = stem.split("-")

    if not tokens or not tokens[0].strip():
        raise ValueError(
            f"Cannot parse patient ID from: {image_path}"
        )

    return tokens[0].strip()


# ============================================================
# 3. Scan ALL patient IDs from train/test
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def collect_patient_ids(root):
    patient_ids = set()

    for path in Path(root).rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            patient_ids.add(
                extract_patient_id(path)
            )

    return patient_ids


print()
print("=" * 72)
print("Scanning global train/test patient IDs")
print("=" * 72)

global_train_patient_ids = collect_patient_ids(
    TRAIN_ROOT
)

global_test_patient_ids = collect_patient_ids(
    TEST_ROOT
)


global_overlap_patients = (
    global_train_patient_ids
    & global_test_patient_ids
)

global_train_only_patients = (
    global_train_patient_ids
    - global_test_patient_ids
)

global_test_only_patients = (
    global_test_patient_ids
    - global_train_patient_ids
)


print(
    f"Train patients:       "
    f"{len(global_train_patient_ids)}"
)

print(
    f"Test patients:        "
    f"{len(global_test_patient_ids)}"
)

print(
    f"Train-only patients:  "
    f"{len(global_train_only_patients)}"
)

print(
    f"Test-only patients:   "
    f"{len(global_test_only_patients)}"
)

print(
    f"Overlap patients:     "
    f"{len(global_overlap_patients)}"
)


# ============================================================
# 4. Global plotting parameters
# ============================================================

# plt.rcParams.update(
#     {
#         "font.size": 16,
#         "axes.titlesize": 21,
#         "axes.labelsize": 19,
#         "xtick.labelsize": 15,
#         "ytick.labelsize": 15,
#         "legend.fontsize": 14,
#         "axes.linewidth": 1.2,
#     }
# )


plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 26,
        "axes.labelsize": 25,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "legend.fontsize": 18,
        "axes.linewidth": 1.6,
    }
)

# ============================================================
# 5. Load experiments and construct strict patient-disjoint
#    labels
# ============================================================

results = {}

global_min_rate = 0.0


for display_name, basename in experiments.items():

    npz_path = os.path.join(
        ROOT,
        basename,
        "rmia_scores_multilevel.npz",
    )

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"File not found:\n{npz_path}"
        )

    print()
    print("=" * 72)
    print(display_name)
    print("=" * 72)
    print(npz_path)

    data = np.load(
        npz_path,
        allow_pickle=True,
    )

    # --------------------------------------------------------
    # Load patient-level scores / IDs
    # --------------------------------------------------------

    patient_scores = np.asarray(
        data["patient_scores"],
        dtype=np.float64,
    ).reshape(-1)

    patient_ids = np.asarray(
        data["patient_ids"],
        dtype=object,
    ).reshape(-1)

    patient_ids = np.asarray(
        [str(pid) for pid in patient_ids],
        dtype=object,
    )

    if len(patient_scores) != len(patient_ids):
        raise ValueError(
            f"{display_name}: patient_scores and patient_ids "
            f"have different lengths: "
            f"{len(patient_scores)} vs {len(patient_ids)}"
        )

    # --------------------------------------------------------
    # Strict patient-disjoint subset
    #
    # train-only -> member = 1
    # test-only  -> member = 0
    # overlap    -> excluded
    # --------------------------------------------------------

    strict_scores = []
    strict_labels = []
    strict_ids = []

    n_overlap_excluded = 0
    n_unknown_excluded = 0

    for pid, score in zip(
        patient_ids,
        patient_scores,
    ):

        if pid in global_train_only_patients:
            strict_ids.append(pid)
            strict_scores.append(float(score))
            strict_labels.append(1)

        elif pid in global_test_only_patients:
            strict_ids.append(pid)
            strict_scores.append(float(score))
            strict_labels.append(0)

        elif pid in global_overlap_patients:
            n_overlap_excluded += 1

        else:
            n_unknown_excluded += 1


    strict_scores = np.asarray(
        strict_scores,
        dtype=np.float64,
    )

    strict_labels = np.asarray(
        strict_labels,
        dtype=np.int64,
    )

    strict_ids = np.asarray(
        strict_ids,
        dtype=object,
    )


    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    unique_labels = np.unique(
        strict_labels
    )

    if not np.array_equal(
        unique_labels,
        np.array([0, 1]),
    ):
        raise ValueError(
            f"{display_name}: expected strict labels {{0,1}}, "
            f"got {unique_labels}"
        )


    num_members = int(
        np.sum(strict_labels == 1)
    )

    num_non_members = int(
        np.sum(strict_labels == 0)
    )


    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    auc_value = roc_auc_score(
        strict_labels,
        strict_scores,
    )

    reverse_auc = roc_auc_score(
        strict_labels,
        -strict_scores,
    )


    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        strict_labels,
        strict_scores,
        pos_label=1,
    )


    # --------------------------------------------------------
    # Empirical resolution
    # --------------------------------------------------------

    min_fpr = (
        1.0 / num_non_members
    )

    min_tpr = (
        1.0 / num_members
    )

    min_rate = max(
        min_fpr,
        min_tpr,
    )

    global_min_rate = max(
        global_min_rate,
        min_rate,
    )


    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print(
        f"All scored patients:       "
        f"{len(patient_ids)}"
    )

    print(
        f"Strict-disjoint patients:  "
        f"{len(strict_labels)}"
    )

    print(
        f"Members (train-only):      "
        f"{num_members}"
    )

    print(
        f"Non-members (test-only):   "
        f"{num_non_members}"
    )

    print(
        f"Overlap excluded:          "
        f"{n_overlap_excluded}"
    )

    print(
        f"Unknown excluded:          "
        f"{n_unknown_excluded}"
    )

    print(
        f"Strict patient AUC:        "
        f"{auc_value:.6f}"
    )

    print(
        f"Reverse AUC:               "
        f"{reverse_auc:.6f}"
    )

    print(
        f"Min rate:                  "
        f"{min_rate:.6e}"
    )


    results[display_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_value,
        "reverse_auc": reverse_auc,
        "min_rate": min_rate,
        "num_members": num_members,
        "num_non_members": num_non_members,
        "strict_ids": strict_ids,
        "strict_scores": strict_scores,
        "strict_labels": strict_labels,
    }


# ============================================================
# 6. Common resolution
# ============================================================

print()
print("=" * 72)

print(
    f"Common minimum rate: "
    f"{global_min_rate:.6e}"
)

print("=" * 72)


# ============================================================
# 7. Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(10.2, 8.8)
)


# ============================================================
# 8. Plot DP methods first
#
# Ours will be plotted last so it remains visible.
# ============================================================

plot_order = [
    "DP-Privacy EP32",
    "DP-Privacy EP8",
    "DP-Privacy EP1",
]

curve_handles = {}


for display_name in plot_order:

    result = results[
        display_name
    ]

    fpr = result["fpr"]
    tpr = result["tpr"]
    auc_value = result["auc"]

    # Log axes cannot display zero.
    # Only modify plotting copies.
    fpr_plot = np.clip(
        fpr,
        global_min_rate,
        1.0,
    )

    tpr_plot = np.clip(
        tpr,
        global_min_rate,
        1.0,
    )

    handle, = ax.plot(
        fpr_plot,
        tpr_plot,
        linewidth=2.0,
        alpha=0.72,
        zorder=3,
        label=(
            f"{display_name} "
            f"(AUC = {auc_value:.4f})"
        ),
    )

    curve_handles[
        display_name
    ] = handle


# ============================================================
# 9. Random guessing
# ============================================================

random_rates = np.geomspace(
    global_min_rate,
    1.0,
    500,
)

random_handle, = ax.plot(
    random_rates,
    random_rates,
    linestyle="--",
    linewidth=2.0,
    alpha=0.85,
    zorder=2,
    label="Random guess",
)


# ============================================================
# 10. Plot Ours LAST
# ============================================================

ours_result = results[
    "Ours"
]

ours_fpr = np.clip(
    ours_result["fpr"],
    global_min_rate,
    1.0,
)

ours_tpr = np.clip(
    ours_result["tpr"],
    global_min_rate,
    1.0,
)

ours_auc = ours_result[
    "auc"
]

ours_handle, = ax.plot(
    ours_fpr,
    ours_tpr,
    linewidth=3.4,
    alpha=1.0,
    zorder=10,
    label=(
        f"Ours "
        f"(AUC = {ours_auc:.4f})"
    ),
)

curve_handles[
    "Ours"
] = ours_handle


# ============================================================
# 11. Log-log axes
# ============================================================

ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)

ax.set_xlim(
    global_min_rate,
    1.0,
)

ax.set_ylim(
    global_min_rate,
    1.0,
)


# ============================================================
# 12. Labels and title
# ============================================================

ax.set_xlabel(
    "False Positive Rate",
    # fontsize=20,
    # labelpad=10,
    fontsize=26,
    labelpad=12,

)

ax.set_ylabel(
    "True Positive Rate",
    fontsize=26,
    labelpad=12,
)

ax.set_title(
    "Patient-level RMIA Membership Inference ROC",
    # fontsize=22,
    # pad=16,
    fontsize=27,
    pad=18,
)


# ============================================================
# 13. Tick formatting
# ============================================================

# ax.tick_params(
#     axis="both",
#     which="major",
#     labelsize=16,
#     width=1.2,
#     length=6,
# )

# ax.tick_params(
#     axis="both",
#     which="minor",
#     width=0.8,
#     length=3,
# )

ax.tick_params(
    axis="both",
    which="major",
    labelsize=22,
    width=1.5,
    length=7,
)

ax.tick_params(
    axis="both",
    which="minor",
    width=1.1,
    length=4,
)

# ============================================================
# 14. Grid
# ============================================================

ax.grid(
    True,
    which="major",
    linestyle=":",
    linewidth=0.85,
    alpha=0.48,
)

ax.grid(
    True,
    which="minor",
    linestyle=":",
    linewidth=0.55,
    alpha=0.27,
)


# ============================================================
# 15. Legend
#
# Ours
# DP1
# DP8
# DP32
# Random guess
# ============================================================

legend_order = [
    "Ours",
    "DP-Privacy EP1",
    "DP-Privacy EP8",
    "DP-Privacy EP32",
]

legend_handles = [
    curve_handles[name]
    for name in legend_order
]

legend_labels = [
    (
        f"{name} "
        f"(AUC = {results[name]['auc']:.4f})"
    )
    for name in legend_order
]

legend_handles.append(
    random_handle
)

legend_labels.append(
    "Random guess"
)


# legend = ax.legend(
#     legend_handles,
#     legend_labels,
#     loc="lower right",
#     fontsize=14,
#     frameon=True,
#     fancybox=True,
#     framealpha=0.93,
#     borderpad=0.8,
#     labelspacing=0.45,
#     handlelength=2.8,
# )

legend = ax.legend(
    legend_handles,
    legend_labels,
    loc="lower right",
    fontsize=18,
    frameon=True,
    fancybox=True,
    framealpha=0.93,
    borderpad=0.8,
    labelspacing=0.45,
    handlelength=2.8,
)

legend.get_frame().set_linewidth(
    1.0
)


# ============================================================
# 16. Spine formatting
# ============================================================

# for spine in ax.spines.values():
#     spine.set_linewidth(
#         1.25
#     )
for spine in ax.spines.values():
    spine.set_linewidth(1.6)

# ============================================================
# 17. Layout
# ============================================================

fig.tight_layout()


# ============================================================
# 18. Save
# ============================================================

output_png = os.path.join(
    ROOT,
    (
        f"rmia_strict_patient_disjoint_loglog_roc_"
        f"seed{TRAINED_SEED}_gamma2_bestep_larger.png"
    ),
)

output_pdf = os.path.join(
    ROOT,
    (
        f"rmia_strict_patient_disjoint_loglog_roc_"
        f"seed{TRAINED_SEED}_gamma2_bestep_larger.pdf"
    ),
)


fig.savefig(
    output_png,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    output_pdf,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# 19. Final output
# ============================================================

print()
print("=" * 72)
print("Strict patient-disjoint RMIA comparison")
print("=" * 72)

for name in [
    "Ours",
    "DP-Privacy EP1",
    "DP-Privacy EP8",
    "DP-Privacy EP32",
]:
    print(
        f"{name:<18s}: "
        f"AUC = {results[name]['auc']:.6f}"
    )

print()

print(
    f"Common minimum rate: "
    f"{global_min_rate:.6e}"
)

print()

print(
    output_png
)

print(
    output_pdf
)

print("=" * 72)