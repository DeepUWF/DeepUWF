import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
)


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(
    "/path/to/workspace/code/"
    "DeepUWF/MIA_attack/experiments"
)

# ------------------------------------------------------------
# Five confidence-MIA result files
#
# Change the paths below to your actual directories.
# ------------------------------------------------------------

experiments = {
    "Ours": (
        BASE_DIR
        / "confidence_mia_results_v2"
        / "confidence_mia_oof_predictions.npz"
    ),

    "DP-Privacy EP1": (
        BASE_DIR
        / "confidence_mia_results_SHDR_ep1_adamw_softmax_results_trainset"
        / "confidence_mia_oof_predictions.npz"
    ),

    "DP-Privacy EP8": (
        BASE_DIR
        / "confidence_mia_results_SHDR_ep8_adamw_softmax_results_trainset"
        / "confidence_mia_oof_predictions.npz"
    ),

    "DP-Privacy EP32": (
        BASE_DIR
        / "confidence_mia_results_SHDR_ep32_adamw_softmax_results_trainset"
        / "confidence_mia_oof_predictions.npz"
    ),
}


# ============================================================
# 2. Select attack model
# ============================================================

# Options:
#
#     "mlp"
#     "transformer"
#
ATTACK_MODEL = "transformer"


if ATTACK_MODEL == "mlp":
    prediction_key = "mlp_predictions"
    label_key = "mlp_labels"

elif ATTACK_MODEL == "transformer":
    prediction_key = "transformer_predictions"
    label_key = "transformer_labels"

else:
    raise ValueError(
        "ATTACK_MODEL must be "
        "'mlp' or 'transformer'."
    )


# ============================================================
# 3. Load all experiments
# ============================================================

roc_results = {}

num_members_all = []
num_nonmembers_all = []


for experiment_name, npz_path in experiments.items():

    if not npz_path.exists():
        raise FileNotFoundError(
            f"File not found:\n{npz_path}"
        )

    data = np.load(
        npz_path
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        experiment_name
    )

    print(
        npz_path
    )

    print(
        "Available keys:"
    )

    print(
        list(
            data.keys()
        )
    )


    predictions = np.asarray(
        data[
            prediction_key
        ],
        dtype=np.float64,
    ).reshape(-1)

    labels = np.asarray(
        data[
            label_key
        ],
        dtype=np.int64,
    ).reshape(-1)


    if len(predictions) != len(labels):
        raise ValueError(
            f"{experiment_name}: "
            "prediction/label length mismatch."
        )


    unique_labels = np.unique(
        labels
    )

    if not np.array_equal(
        unique_labels,
        np.array(
            [0, 1]
        ),
    ):
        raise ValueError(
            f"{experiment_name}: "
            f"expected labels {{0,1}}, "
            f"got {unique_labels}"
        )


    num_members = int(
        np.sum(
            labels == 1
        )
    )

    num_nonmembers = int(
        np.sum(
            labels == 0
        )
    )

    num_members_all.append(
        num_members
    )

    num_nonmembers_all.append(
        num_nonmembers
    )


    auc_value = roc_auc_score(
        labels,
        predictions,
    )

    fpr, tpr, _ = roc_curve(
        labels,
        predictions,
        pos_label=1,
    )


    roc_results[
        experiment_name
    ] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_value,
        "num_members": num_members,
        "num_nonmembers": num_nonmembers,
    }


    print(
        f"Members:     "
        f"{num_members}"
    )

    print(
        f"Non-members: "
        f"{num_nonmembers}"
    )

    print(
        f"AUROC:       "
        f"{auc_value:.6f}"
    )


# ============================================================
# 4. Common minimum measurable rate
# ============================================================
#
# To compare all five methods on the SAME log-log axes,
# use the most conservative empirical resolution.
#
# For a method:
#
#     min FPR = 1 / N_nonmember
#     min TPR = 1 / N_member
#
# A common lower bound should therefore be:
#
#     max(
#         1 / minimum N_nonmember,
#         1 / minimum N_member
#     )
#
# ============================================================

common_num_members = min(
    num_members_all
)

common_num_nonmembers = min(
    num_nonmembers_all
)


min_rate = max(
    1.0
    / common_num_members,

    1.0
    / common_num_nonmembers,
)


print(
    "\n"
    + "=" * 70
)

print(
    f"Common minimum rate: "
    f"{min_rate:.6e}"
)

print(
    f"Common member count: "
    f"{common_num_members}"
)

print(
    f"Common non-member count: "
    f"{common_num_nonmembers}"
)


# ============================================================
# 5. Plot five log-log ROC curves
# ============================================================

plt.figure(
    figsize=(9, 8)
)


for experiment_name, result in roc_results.items():

    fpr = result[
        "fpr"
    ]

    tpr = result[
        "tpr"
    ]

    auc_value = result[
        "auc"
    ]

    # --------------------------------------------------------
    # Same convention as your previous confidence / gradient
    # log-log plots:
    #
    # log axes cannot represent zero, so clip only the
    # plotting copies.
    # --------------------------------------------------------

    fpr_plot = np.clip(
        fpr,
        min_rate,
        1.0,
    )

    tpr_plot = np.clip(
        tpr,
        min_rate,
        1.0,
    )


    plt.plot(
        fpr_plot,
        tpr_plot,
        linewidth=2.2,
        label=(
            f"{experiment_name} "
            f"(AUC={auc_value:.4f})"
        ),
    )


# ------------------------------------------------------------
# Random guessing baseline
# ------------------------------------------------------------

random_rates = np.geomspace(
    min_rate,
    1.0,
    500,
)


plt.plot(
    random_rates,
    random_rates,
    linestyle="--",
    linewidth=1.7,
    label="Random guess",
)


# ------------------------------------------------------------
# Log-log axes
# ------------------------------------------------------------

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
    fontsize=14,
)

plt.ylabel(
    "True Positive Rate",
    fontsize=14,
)

plt.title(
    (
        "Confidence-based Membership "
        "Inference ROC"
    ),
    fontsize=16,
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


# ============================================================
# 6. Save
# ============================================================

save_path = (
    BASE_DIR
    / (
        f"confidence_mia_5methods_"
        f"{ATTACK_MODEL}_loglog.png"
    )
)


plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "\nSaved comparison figure:"
)

print(
    save_path
)