import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)


# ============================================================
# 1. Configuration
# ============================================================

ROOT = (
    "/path/to/workspace/code/DeepUWF/"
    "MIA_attack/experiments"
)


# ------------------------------------------------------------
# Five independently trained victim-model seeds
# ------------------------------------------------------------

SEEDS = [
    0,
    1,
    2,
    3,
    4,
]


# ============================================================
# 2. Experiment directory templates
# ============================================================

EXPERIMENT_TEMPLATES = {

    "Ours": (
        "rmia_results_vitshadow_"
        "ours_{seed}_alltrainset_precise_gamma2_bestep"
    ),

    "DP-Privacy EP1": (
        "rmia_results_vitshadow_"
        "DP1_{seed}_alltrainset_precise_gamma2_bestep"
    ),

    "DP-Privacy EP8": (
        "rmia_results_vitshadow_"
        "DP8_{seed}_alltrainset_precise_gamma2_bestep"
    ),

    "DP-Privacy EP32": (
        "rmia_results_vitshadow_"
        "DP32_{seed}_alltrainset_precise_gamma2_bestep"
    ),
}


# ============================================================
# 3. Plot colors
# ============================================================

COLORS = {

    # Highlight our method
    "Ours": "#8A3E46",

    "DP-Privacy EP1": "#E45756",
    "DP-Privacy EP8": "#59A95A",
    "DP-Privacy EP32": "#F28E3B",
}


RANDOM_COLOR = "#9575CD"


# ============================================================
# 4. Plot style
# ============================================================

# plt.rcParams.update(
#     {
#         "font.size": 16,

#         "axes.titlesize": 21,
#         "axes.labelsize": 19,

#         "xtick.labelsize": 15,
#         "ytick.labelsize": 15,

#         "legend.fontsize": 13,

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
# 5. Helper:
#    mean + t-based 95% CI
#
# Used only for numerical AUROC statistics,
# NOT for ROC shading.
# ============================================================

def mean_ci_t(
    values,
    confidence=0.95,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(
        values
    )

    mean_value = float(
        np.mean(
            values
        )
    )

    if n < 2:

        return (
            mean_value,
            np.nan,
            np.nan,
        )

    sem = stats.sem(
        values
    )

    critical = stats.t.ppf(
        (
            1.0
            + confidence
        )
        / 2.0,
        df=n - 1,
    )

    margin = (
        critical
        * sem
    )

    return (
        mean_value,

        float(
            mean_value
            - margin
        ),

        float(
            mean_value
            + margin
        ),
    )


# ============================================================
# 6. Helper:
#    prepare ROC curve for interpolation
# ============================================================

def prepare_roc_for_interpolation(
    fpr,
    tpr,
):

    fpr = np.asarray(
        fpr,
        dtype=np.float64,
    )

    tpr = np.asarray(
        tpr,
        dtype=np.float64,
    )


    unique_fpr = np.unique(
        fpr
    )


    max_tpr = np.empty_like(
        unique_fpr
    )


    for index, value in enumerate(
        unique_fpr
    ):

        max_tpr[
            index
        ] = np.max(
            tpr[
                fpr == value
            ]
        )


    return (
        unique_fpr,
        max_tpr,
    )


# ============================================================
# 7. Load every method / every seed
# ============================================================

results = {

    method: {}

    for method in (
        EXPERIMENT_TEMPLATES
    )
}


global_min_rate = 0.0


print()
print("=" * 80)
print("Loading five-seed RMIA results")
print("=" * 80)


for method_name, template in (
    EXPERIMENT_TEMPLATES.items()
):

    print()
    print("#" * 80)
    print(method_name)
    print("#" * 80)


    for seed in SEEDS:

        basename = template.format(
            seed=seed
        )


        npz_path = os.path.join(
            ROOT,
            basename,
            "rmia_scores.npz",
        )


        if not os.path.exists(
            npz_path
        ):

            raise FileNotFoundError(
                "\n"
                f"Missing RMIA result:\n"
                f"method = {method_name}\n"
                f"seed   = {seed}\n"
                f"path   = {npz_path}\n"
            )


        data = np.load(
            npz_path,
            allow_pickle=True,
        )


        scores = np.asarray(
            data[
                "scores"
            ],
            dtype=np.float64,
        ).reshape(
            -1
        )


        labels = np.asarray(
            data[
                "membership_labels"
            ],
            dtype=np.int64,
        ).reshape(
            -1
        )


        # ----------------------------------------------------
        # Basic checks
        # ----------------------------------------------------

        if len(scores) != len(labels):

            raise ValueError(
                f"{method_name}, seed={seed}: "
                "score/label length mismatch: "
                f"{len(scores)} vs {len(labels)}"
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
                f"{method_name}, seed={seed}: "
                f"expected labels {{0,1}}, "
                f"but got {unique_labels}"
            )


        num_members = int(
            np.sum(
                labels == 1
            )
        )


        num_non_members = int(
            np.sum(
                labels == 0
            )
        )


        # ----------------------------------------------------
        # AUROC
        # ----------------------------------------------------

        auc_value = roc_auc_score(
            labels,
            scores,
        )


        reverse_auc = roc_auc_score(
            labels,
            -scores,
        )


        # ----------------------------------------------------
        # ROC
        # ----------------------------------------------------

        fpr, tpr, _ = roc_curve(
            labels,
            scores,
            pos_label=1,
        )


        # ----------------------------------------------------
        # Empirical resolution
        # ----------------------------------------------------

        min_fpr = (
            1.0
            / num_non_members
        )


        min_tpr = (
            1.0
            / num_members
        )


        min_rate = max(
            min_fpr,
            min_tpr,
        )


        global_min_rate = max(
            global_min_rate,
            min_rate,
        )


        results[
            method_name
        ][
            seed
        ] = {

            "scores": (
                scores
            ),

            "labels": (
                labels
            ),

            "fpr": (
                fpr
            ),

            "tpr": (
                tpr
            ),

            "auc": float(
                auc_value
            ),

            "reverse_auc": float(
                reverse_auc
            ),

            "num_members": (
                num_members
            ),

            "num_non_members": (
                num_non_members
            ),

            "min_rate": (
                min_rate
            ),

            "path": (
                npz_path
            ),
        }


        print(
            f"Seed {seed}: "
            f"AUC={auc_value:.6f}, "
            f"reverse={reverse_auc:.6f}, "
            f"N_member={num_members}, "
            f"N_nonmember={num_non_members}"
        )


# ============================================================
# 8. Seed-level AUROC statistics
# ============================================================

summary = {}


print()
print("=" * 80)
print("Five-seed RMIA AUROC summary")
print("=" * 80)


for method_name in (
    EXPERIMENT_TEMPLATES
):

    aucs = np.asarray(
        [
            results[
                method_name
            ][
                seed
            ][
                "auc"
            ]

            for seed in SEEDS
        ],
        dtype=np.float64,
    )


    mean_auc, ci_low, ci_high = (
        mean_ci_t(
            aucs
        )
    )


    std_auc = float(
        np.std(
            aucs,
            ddof=1,
        )
    )


    summary[
        method_name
    ] = {

        "aucs": (
            aucs
        ),

        "mean_auc": (
            mean_auc
        ),

        "std_auc": (
            std_auc
        ),

        "ci_low": (
            ci_low
        ),

        "ci_high": (
            ci_high
        ),
    }


    print()
    print(
        method_name
    )

    print(
        "  seed AUROCs: "
        + ", ".join(
            [
                f"{value:.6f}"

                for value in aucs
            ]
        )
    )

    print(
        f"  Mean AUROC: "
        f"{mean_auc:.6f}"
    )

    print(
        f"  SD:        "
        f"{std_auc:.6f}"
    )

    print(
        f"  95% CI:    "
        f"[{ci_low:.6f}, "
        f"{ci_high:.6f}]"
    )


# ============================================================
# 9. Paired two-sided t-tests
#
# ============================================================

COMPARISONS = [

    (
        "Ours",
        "DP-Privacy EP32",
    ),

    (
        "Ours",
        "DP-Privacy EP8",
    ),

    (
        "Ours",
        "DP-Privacy EP1",
    ),
]


paired_results = {}


print()
print("=" * 80)
print("Paired two-sided t-tests across five seeds")
print("=" * 80)


for method_a, method_b in (
    COMPARISONS
):

    auc_a = summary[
        method_a
    ][
        "aucs"
    ]


    auc_b = summary[
        method_b
    ][
        "aucs"
    ]


    differences = (
        auc_a
        - auc_b
    )


    t_stat, p_value = (
        stats.ttest_rel(
            auc_a,
            auc_b,
        )
    )


    mean_difference, diff_ci_low, diff_ci_high = (
        mean_ci_t(
            differences
        )
    )


    paired_results[
        (
            method_a,
            method_b,
        )
    ] = {

        "mean_difference": (
            mean_difference
        ),

        "ci_low": (
            diff_ci_low
        ),

        "ci_high": (
            diff_ci_high
        ),

        "t": float(
            t_stat
        ),

        "p": float(
            p_value
        ),
    }


    print()
    print(
        f"{method_a} vs {method_b}"
    )

    print(
        "  seed-wise differences: "
        + ", ".join(
            [
                f"{value:+.6f}"

                for value in differences
            ]
        )
    )

    print(
        f"  Mean difference: "
        f"{mean_difference:+.6f}"
    )

    print(
        f"  95% CI of difference: "
        f"[{diff_ci_low:+.6f}, "
        f"{diff_ci_high:+.6f}]"
    )

    print(
        f"  t = "
        f"{t_stat:.6f}"
    )

    print(
        f"  p = "
        f"{p_value:.8f}"
    )


# ============================================================
# 10. Build common log-FPR grid
# ============================================================

common_fpr_grid = np.geomspace(
    global_min_rate,
    1.0,
    600,
)


# ============================================================
# 11. Interpolate ROC curves and calculate ONLY mean ROC
#
# No point-wise SD / CI is calculated for visualization.
# ============================================================

for method_name in (
    EXPERIMENT_TEMPLATES
):

    interpolated_tprs = []


    for seed in SEEDS:

        fpr = results[
            method_name
        ][
            seed
        ][
            "fpr"
        ]


        tpr = results[
            method_name
        ][
            seed
        ][
            "tpr"
        ]


        # ----------------------------------------------------
        # Remove duplicate FPR values
        # ----------------------------------------------------

        fpr_unique, tpr_unique = (
            prepare_roc_for_interpolation(
                fpr,
                tpr,
            )
        )


        # ----------------------------------------------------
        # Log axes cannot represent zero.
        #
        # Only modify plotting/interpolation copies.
        # ----------------------------------------------------

        fpr_plot = np.maximum(
            fpr_unique,
            global_min_rate,
        )


        tpr_plot = np.maximum(
            tpr_unique,
            global_min_rate,
        )


        # Clipping may create repeated FPR values.
        fpr_plot, tpr_plot = (
            prepare_roc_for_interpolation(
                fpr_plot,
                tpr_plot,
            )
        )


        interpolated = np.interp(
            common_fpr_grid,
            fpr_plot,
            tpr_plot,

            left=(
                tpr_plot[0]
            ),

            right=(
                tpr_plot[-1]
            ),
        )


        interpolated_tprs.append(
            interpolated
        )


    interpolated_tprs = np.asarray(
        interpolated_tprs,
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # Only the mean ROC across five seeds
    # --------------------------------------------------------

    mean_tpr = np.mean(
        interpolated_tprs,
        axis=0,
    )


    summary[
        method_name
    ][
        "mean_tpr"
    ] = mean_tpr


# ============================================================
# 12. Create mean ROC figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10.2,
        8.8,
    )
)


curve_handles = {}


# ============================================================
# 13. Plot DP methods first
# ============================================================

plot_order = [

    "DP-Privacy EP32",

    "DP-Privacy EP8",

    "DP-Privacy EP1",
]


for method_name in (
    plot_order
):

    mean_tpr = summary[
        method_name
    ][
        "mean_tpr"
    ]


    mean_auc = summary[
        method_name
    ][
        "mean_auc"
    ]


    handle, = ax.plot(
        common_fpr_grid,
        mean_tpr,

        color=COLORS[
            method_name
        ],

        linewidth=2.1,

        alpha=0.82,

        zorder=3,

        label=(
            f"{method_name} "
            f"(AUC = {mean_auc:.4f})"
        ),
    )


    curve_handles[
        method_name
    ] = handle


# ============================================================
# 14. Random guessing baseline
#
# Note:
# This is the theoretical random-guess line,
# ============================================================

random_handle, = ax.plot(
    common_fpr_grid,
    common_fpr_grid,

    color=RANDOM_COLOR,

    linestyle="--",

    linewidth=2.1,

    alpha=0.90,

    zorder=2,

    label="Random guess",
)


# ============================================================
# 15. Plot Ours LAST and emphasize it
# ============================================================

ours_mean = summary[
    "Ours"
][
    "mean_tpr"
]


ours_auc = summary[
    "Ours"
][
    "mean_auc"
]


ours_handle, = ax.plot(
    common_fpr_grid,
    ours_mean,

    color=COLORS[
        "Ours"
    ],

    linewidth=3.6,

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
# 16. Log-log axes
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
# 17. Labels and title
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
    "RMIA Membership Inference ROC",
    # fontsize=22,
    # pad=16,
    fontsize=27,
    pad=18,
)


# ============================================================
# 18. Ticks
# ============================================================

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
# 19. Grid
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
# 20. Legend
# ============================================================

legend_order = [

    "Ours",

    "DP-Privacy EP1",

    "DP-Privacy EP8",

    "DP-Privacy EP32",
]


legend_handles = [

    curve_handles[
        name
    ]

    for name in (
        legend_order
    )
]


legend_labels = [

    (
        f"{name} "
        f"(AUC = "
        f"{summary[name]['mean_auc']:.4f})"
    )

    for name in (
        legend_order
    )
]


legend_handles.append(
    random_handle
)


legend_labels.append(
    "Random guess"
)


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
# 21. Spines
# ============================================================

for spine in (
    ax.spines.values()
):

    spine.set_linewidth(
        #1.25
        1.6
    )


fig.tight_layout()


# ============================================================
# 22. Save mean ROC figure
# ============================================================

output_png = os.path.join(
    ROOT,
    "rmia_5seed_mean_roc_no_shading_larger.png",
)


output_pdf = os.path.join(
    ROOT,
    "rmia_5seed_mean_roc_no_shading_larger.pdf",
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
# 23. Save seed-level AUROCs
# ============================================================

seed_csv = os.path.join(
    ROOT,
    "rmia_5seed_auroc_results.csv",
)


with open(
    seed_csv,
    "w",
    newline="",
) as f:

    writer = csv.writer(
        f
    )


    writer.writerow(
        [
            "Method",
            "Seed",
            "AUROC",
            "Reverse_AUROC",
        ]
    )


    for method_name in (
        EXPERIMENT_TEMPLATES
    ):

        for seed in SEEDS:

            writer.writerow(
                [
                    method_name,

                    seed,

                    results[
                        method_name
                    ][
                        seed
                    ][
                        "auc"
                    ],

                    results[
                        method_name
                    ][
                        seed
                    ][
                        "reverse_auc"
                    ],
                ]
            )


# ============================================================
# 24. Save AUROC summary statistics
#
# CI is still retained numerically even though
# it is not displayed as ROC shading.
# ============================================================

summary_csv = os.path.join(
    ROOT,
    "rmia_5seed_summary.csv",
)


with open(
    summary_csv,
    "w",
    newline="",
) as f:

    writer = csv.writer(
        f
    )


    writer.writerow(
        [
            "Method",
            "Mean_AUROC",
            "SD",
            "CI95_Lower",
            "CI95_Upper",
        ]
    )


    for method_name in (
        EXPERIMENT_TEMPLATES
    ):

        writer.writerow(
            [
                method_name,

                summary[
                    method_name
                ][
                    "mean_auc"
                ],

                summary[
                    method_name
                ][
                    "std_auc"
                ],

                summary[
                    method_name
                ][
                    "ci_low"
                ],

                summary[
                    method_name
                ][
                    "ci_high"
                ],
            ]
        )


# ============================================================
# 25. Save paired statistical tests
# ============================================================

ptest_csv = os.path.join(
    ROOT,
    "rmia_5seed_paired_ttests.csv",
)


with open(
    ptest_csv,
    "w",
    newline="",
) as f:

    writer = csv.writer(
        f
    )


    writer.writerow(
        [
            "Method_A",
            "Method_B",
            "Mean_AUC_Difference",
            "Difference_CI95_Lower",
            "Difference_CI95_Upper",
            "t_statistic",
            "p_value",
        ]
    )


    for comparison in (
        COMPARISONS
    ):

        method_a, method_b = (
            comparison
        )


        stat_result = paired_results[
            comparison
        ]


        writer.writerow(
            [
                method_a,

                method_b,

                stat_result[
                    "mean_difference"
                ],

                stat_result[
                    "ci_low"
                ],

                stat_result[
                    "ci_high"
                ],

                stat_result[
                    "t"
                ],

                stat_result[
                    "p"
                ],
            ]
        )


# ============================================================
# 26. Final output
# ============================================================

print()
print("=" * 80)
print("Finished five-seed RMIA evaluation")
print("=" * 80)


print(
    f"Common minimum rate: "
    f"{global_min_rate:.6e}"
)


print()

print(
    "Mean ROC figure:"
)

print(
    output_png
)

print(
    output_pdf
)


print()

print(
    "Seed-level AUROCs:"
)

print(
    seed_csv
)


print()

print(
    "Five-seed AUROC summary:"
)

print(
    summary_csv
)


print()

print(
    "Paired statistical tests:"
)

print(
    ptest_csv
)


print("=" * 80)