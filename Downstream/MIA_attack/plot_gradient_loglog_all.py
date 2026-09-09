import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# ======================================================
# Configurations
# ======================================================
experiments = {
    "DP-Privacy EP32": (
        "/path/to/workspace/code/DeepUWF/MIA_attack/experiments/"
        "gradient_mia_results_dpprivacyep32epoch0_all"
    ),
 # Replace this with your fourth experiment
    "DP-Privacy EP8": (
        "/path/to/workspace/code/DeepUWF/MIA_attack/experiments/"
        "gradient_mia_results_dpprivacyep8epoch0_all"
    ),
    "DP-Privacy EP1": (
        "/path/to/workspace/code/DeepUWF/MIA_attack/experiments/"
        "gradient_mia_results_dpprivacyep1epoch0_all"
    ),
   

    "Ours": (
        "/path/to/workspace/code/DeepUWF/MIA_attack/experiments/"
        "gradient_mia_results_all"
    ),
}


# ======================================================
# Load all ROC curves first
# ======================================================
results = {}

global_min_rate = 1.0

for name, basepath in experiments.items():

    data = np.load(
        f"{basepath}/gradient_mia_scores.npz"
    )

    member_norms = data["member_norms"]
    non_member_norms = data["non_member_norms"]

    print(f"\n{name}")
    print(f"Members     : {len(member_norms)}")
    print(f"Non-members : {len(non_member_norms)}")

    # --------------------------------------------------
    # Smaller gradient -> more likely to be member
    # --------------------------------------------------
    y_true = np.concatenate([
        np.ones(len(member_norms)),
        np.zeros(len(non_member_norms)),
    ])

    y_score = np.concatenate([
        -member_norms,
        -non_member_norms,
    ])

    auc = roc_auc_score(
        y_true,
        y_score,
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score,
        pos_label=1,
    )

    num_non_members = len(non_member_norms)
    num_members = len(member_norms)

    min_rate = max(
        1.0 / num_non_members,
        1.0 / num_members,
    )

    global_min_rate = min(
        global_min_rate,
        min_rate,
    )

    results[name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "min_rate": min_rate,
    }

    print(f"Gradient AUROC = {auc:.4f}")
    print(f"Minimum rate   = {min_rate:.6e}")


# ======================================================
# Plot
# ======================================================
plt.figure(figsize=(8, 7))


for name, result in results.items():

    fpr = result["fpr"]
    tpr = result["tpr"]
    auc = result["auc"]

    # --------------------------------------------------
    # Log scale cannot display zero.
    #
    # Map zero to the bottom of the visible axis.
    # This preserves the vertical drops/rises of the
    # ROC curve at zero.
    # --------------------------------------------------
    fpr_plot = np.where(
        fpr == 0,
        global_min_rate,
        fpr,
    )

    tpr_plot = np.where(
        tpr == 0,
        global_min_rate,
        tpr,
    )

    plt.plot(
        fpr_plot,
        tpr_plot,
        linewidth=2.3,
        drawstyle="steps-post",
        label=f"{name} (AUC={auc:.4f})",
    )


# ======================================================
# Random baseline
# ======================================================
random_rates = np.logspace(
    np.log10(global_min_rate),
    0,
    500,
)

plt.plot(
    random_rates,
    random_rates,
    "--",
    linewidth=1.6,
    label="Random guess",
)


# ======================================================
# Axes
# ======================================================
plt.xscale("log")
plt.yscale("log")

plt.xlim(global_min_rate, 1.0)
plt.ylim(global_min_rate, 1.0)

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
    fontsize=10,
)

plt.tight_layout()


# ======================================================
# Save
# ======================================================
output_path = (
    "/path/to/workspace/code/DeepUWF/MIA_attack/experiments/"
    "gradient_mia_loglog_roc_comparison.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nSaved to: {output_path}")