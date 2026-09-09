import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. Configuration
# ============================================================

NPZ_PATH = (
    "/path/to/workspace/code/DeepUWF/"
    "delong_bootstrap_results_moremoredata.npz"
)

OUTPUT_PATH = (
    "/path/to/workspace/code/DeepUWF/"
    "utility_auroc_comparison.png"
)


# ============================================================
# 2. Method order and colors
# ============================================================

METHODS = [
    "Baseline",
    "DP32",
    "DP8",
    "DP1",
    "Ours",
]

COLORS = {
    "Baseline": "#556672",

    "DP32": "#F0F8FF",
    "DP8": "#4F71BE",
    "DP1": "#6A99D0",

    "Ours": "#8A3E46",
}

DISPLAY_NAMES = {
    "Baseline": "Baseline",
    "DP32": "DP32",
    "DP8": "DP8",
    "DP1": "DP1",
    "Ours": "Ours",
}


# ============================================================
# 3. Load NPZ
# ============================================================

data = np.load(
    NPZ_PATH,
    allow_pickle=True,
)

print("=" * 72)
print("Available arrays in NPZ:")
print("=" * 72)

for key in data.files:
    print(
        f"{key}: "
        f"shape={data[key].shape}"
    )


# ============================================================
# 4. Ground-truth labels
# ============================================================

y_true = np.asarray(
    data["y_true"],
    dtype=np.int64,
)

print()
print(
    f"Number of test samples: {len(y_true)}"
)

print(
    f"Class 0: {np.sum(y_true == 0)}"
)

print(
    f"Class 1: {np.sum(y_true == 1)}"
)


# ============================================================
# 5. Load scores + observed AUROC + bootstrap 95% CI
# ============================================================

method_scores = {}

aucs = []
ci_lower = []
ci_upper = []

print()
print("=" * 72)
print("AUROC and bootstrap 95% CI")
print("=" * 72)

for method in METHODS:

    # --------------------------------------------------------
    # Original scores
    # --------------------------------------------------------

    score_key = (
        f"{method}_scores"
    )

    scores = np.asarray(
        data[score_key],
        dtype=np.float64,
    )

    method_scores[
        method
    ] = scores

    auc = roc_auc_score(
        y_true,
        scores,
    )

    # --------------------------------------------------------
    # Bootstrap AUROCs
    # --------------------------------------------------------

    bootstrap_key = (
        f"{method}_bootstrap_aucs"
    )

    bootstrap_aucs = np.asarray(
        data[bootstrap_key],
        dtype=np.float64,
    )

    lower = np.percentile(
        bootstrap_aucs,
        2.5,
    )

    upper = np.percentile(
        bootstrap_aucs,
        97.5,
    )

    aucs.append(
        auc
    )

    ci_lower.append(
        lower
    )

    ci_upper.append(
        upper
    )

    print(
        f"{method:10s}: "
        f"AUROC={auc:.6f}, "
        f"95% CI=[{lower:.6f}, {upper:.6f}]"
    )


aucs = np.asarray(
    aucs,
    dtype=np.float64,
)

ci_lower = np.asarray(
    ci_lower,
    dtype=np.float64,
)

ci_upper = np.asarray(
    ci_upper,
    dtype=np.float64,
)


# ============================================================
# 6. Convert CI to asymmetric error bars
# ============================================================

lower_errors = (
    aucs
    - ci_lower
)

upper_errors = (
    ci_upper
    - aucs
)

yerr = np.vstack(
    [
        lower_errors,
        upper_errors,
    ]
)


# ============================================================
# 7. DeLong implementation
# ============================================================

def compute_midrank(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    order = np.argsort(
        x
    )

    sorted_x = x[
        order
    ]

    n = len(
        x
    )

    ranks = np.zeros(
        n,
        dtype=np.float64,
    )

    i = 0

    while i < n:

        j = i

        while (
            j < n
            and sorted_x[j] == sorted_x[i]
        ):
            j += 1

        # 1-based midrank
        midrank = (
            0.5
            * (
                i
                + j
                - 1
            )
            + 1.0
        )

        ranks[
            i:j
        ] = midrank

        i = j

    result = np.empty(
        n,
        dtype=np.float64,
    )

    result[
        order
    ] = ranks

    return result


def fast_delong(
    predictions_sorted_transposed,
    label_1_count,
):

    m = int(
        label_1_count
    )

    n = (
        predictions_sorted_transposed.shape[1]
        - m
    )

    positive_examples = (
        predictions_sorted_transposed[
            :,
            :m
        ]
    )

    negative_examples = (
        predictions_sorted_transposed[
            :,
            m:
        ]
    )

    k = (
        predictions_sorted_transposed.shape[0]
    )

    tx = np.empty(
        (k, m),
        dtype=np.float64,
    )

    ty = np.empty(
        (k, n),
        dtype=np.float64,
    )

    tz = np.empty(
        (k, m + n),
        dtype=np.float64,
    )

    for r in range(
        k
    ):

        tx[r] = compute_midrank(
            positive_examples[r]
        )

        ty[r] = compute_midrank(
            negative_examples[r]
        )

        tz[r] = compute_midrank(
            predictions_sorted_transposed[r]
        )

    aucs_delong = (
        tz[
            :,
            :m
        ].sum(
            axis=1
        )
        / m
        / n
        - (
            m + 1.0
        )
        / (
            2.0 * n
        )
    )

    v01 = (
        tz[
            :,
            :m
        ]
        - tx
    ) / n

    v10 = (
        1.0
        - (
            tz[
                :,
                m:
            ]
            - ty
        )
        / m
    )

    sx = np.cov(
        v01
    )

    sy = np.cov(
        v10
    )

    covariance = (
        sx / m
        + sy / n
    )

    return (
        aucs_delong,
        covariance,
    )


def paired_delong_test(
    y_true,
    score_a,
    score_b,
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    score_a = np.asarray(
        score_a,
        dtype=np.float64,
    )

    score_b = np.asarray(
        score_b,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Put positives first for fast DeLong
    # --------------------------------------------------------

    order = np.argsort(
        -y_true
    )

    y_sorted = y_true[
        order
    ]

    predictions_sorted = np.vstack(
        [
            score_a[
                order
            ],
            score_b[
                order
            ],
        ]
    )

    num_positive = int(
        np.sum(
            y_sorted == 1
        )
    )

    aucs_delong, covariance = (
        fast_delong(
            predictions_sorted,
            num_positive,
        )
    )

    # --------------------------------------------------------
    # Sanity check:
    # DeLong AUROC should match sklearn AUROC
    # --------------------------------------------------------

    sklearn_auc_a = roc_auc_score(
        y_true,
        score_a,
    )

    sklearn_auc_b = roc_auc_score(
        y_true,
        score_b,
    )

    if not np.allclose(
        aucs_delong[0],
        sklearn_auc_a,
        atol=1e-8,
    ):
        raise RuntimeError(
            f"DeLong AUC mismatch: "
            f"{aucs_delong[0]} vs "
            f"{sklearn_auc_a}"
        )

    if not np.allclose(
        aucs_delong[1],
        sklearn_auc_b,
        atol=1e-8,
    ):
        raise RuntimeError(
            f"DeLong AUC mismatch: "
            f"{aucs_delong[1]} vs "
            f"{sklearn_auc_b}"
        )

    contrast = np.array(
        [
            1.0,
            -1.0,
        ],
        dtype=np.float64,
    )

    difference = float(
        aucs_delong[0]
        - aucs_delong[1]
    )

    variance = float(
        contrast
        @ covariance
        @ contrast.T
    )

    if variance <= 0:
        raise RuntimeError(
            "Non-positive DeLong variance."
        )

    standard_error = np.sqrt(
        variance
    )

    z_value = (
        difference
        / standard_error
    )

    # Two-sided p-value
    p_value = (
        2.0
        * norm.sf(
            abs(
                z_value
            )
        )
    )

    return {
        "auc_a": float(
            aucs_delong[0]
        ),
        "auc_b": float(
            aucs_delong[1]
        ),
        "difference": difference,
        "z": float(
            z_value
        ),
        "p_value": float(
            p_value
        ),
    }


# ============================================================
# 8. Ours-centered DeLong comparisons
# ============================================================

COMPARISONS = [
    (
        "Ours",
        "Baseline",
    ),
    (
        "Ours",
        "DP32",
    ),
    (
        "Ours",
        "DP8",
    ),
    (
        "Ours",
        "DP1",
    ),
]


delong_results = {}

print()
print("=" * 72)
print("Paired two-sided DeLong tests")
print("=" * 72)

for method_a, method_b in (
    COMPARISONS
):

    result = paired_delong_test(
        y_true=y_true,
        score_a=method_scores[
            method_a
        ],
        score_b=method_scores[
            method_b
        ],
    )

    delong_results[
        (
            method_a,
            method_b,
        )
    ] = result

    print()
    print(
        f"{method_a} vs {method_b}"
    )

    print(
        f"  AUC {method_a}: "
        f"{result['auc_a']:.6f}"
    )

    print(
        f"  AUC {method_b}: "
        f"{result['auc_b']:.6f}"
    )

    print(
        f"  Delta AUC: "
        f"{result['difference']:.6f}"
    )

    print(
        f"  z = "
        f"{result['z']:.4f}"
    )

    print(
        f"  p = "
        f"{result['p_value']:.10g}"
    )


# ============================================================
# 9. p-value formatting
# ============================================================

def format_p_value(
    p_value,
):

    if p_value < 0.0001:
        return "p < 0.0001"

    elif p_value < 0.001:
        return (
            f"p = {p_value:.4f}"
        )

    elif p_value < 0.01:
        return (
            f"p = {p_value:.4f}"
        )

    else:
        return (
            f"p = {p_value:.3f}"
        )


# ============================================================
# 10. Helper to draw p-value bracket
# ============================================================

def add_pvalue_bracket(
    ax,
    x1,
    x2,
    y,
    height,
    text,
    fontsize=9,
):
    """
    Draw:

        ┌───────────────┐
             p=...
    """

    ax.plot(
        [
            x1,
            x1,
            x2,
            x2,
        ],
        [
            y,
            y + height,
            y + height,
            y,
        ],
        linewidth=1.1,
        color="black",
        clip_on=False,
    )

    ax.text(
        (
            x1 + x2
        )
        / 2.0,
        y + height + 0.0010,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )


# ============================================================
# 11. Plot
# ============================================================

x = np.arange(
    len(METHODS)
)


fig, ax = plt.subplots(
    figsize=(7.5, 6.5)
)


bars = ax.bar(
    x,
    aucs,
    width=0.68,

    color=[
        COLORS[method]
        for method in METHODS
    ],

    edgecolor="#34495E",
    linewidth=1.0,

    yerr=yerr,

    error_kw={
        "elinewidth": 1.3,
        "capsize": 4,
        "capthick": 1.3,
        "ecolor": "#6B7280",
    },
)


# ============================================================
# 12. Axis formatting
# ============================================================

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        DISPLAY_NAMES[
            method
        ]
        for method in METHODS
    ],
    fontsize=12,
)


ax.set_ylabel(
    "AUROC",
    fontsize=14,
)


ax.set_title(
    "Classification Performance",
    fontsize=15,
    pad=12,
)


# ------------------------------------------------------------
# Need extra room above bars for p-value brackets.
# ------------------------------------------------------------

ax.set_ylim(
    0.84,
    1.005,
)


ax.tick_params(
    axis="y",
    labelsize=11,
)


ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)


ax.spines[
    "left"
].set_linewidth(
    1.1
)

ax.spines[
    "bottom"
].set_linewidth(
    1.1
)


# ============================================================
# 13. AUROC values above error bars
# ============================================================

for bar, auc, upper in zip(
    bars,
    aucs,
    ci_upper,
):

    x_center = (
        bar.get_x()
        + bar.get_width()
        / 2.0
    )

    ax.text(
        x_center,
        upper + 0.0020,
        f"{auc:.3f}",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )


# ============================================================
# 14. Add p-value brackets
# ============================================================

method_to_x = {
    method: index
    for index, method
    in enumerate(
        METHODS
    )
}


# ------------------------------------------------------------
# Stack brackets at different heights.
#
# Shorter comparisons are placed lower;
# longer comparisons are placed higher.
# ------------------------------------------------------------

PLOT_COMPARISONS = [
    (
        "Ours",
        "DP1",
        0.947,
    ),
    (
        "Ours",
        "DP8",
        0.960,
    ),
    (
        "Ours",
        "DP32",
        0.973,
    ),
    (
        "Ours",
        "Baseline",
        0.986,
    ),
]


for (
    method_a,
    method_b,
    bracket_y,
) in PLOT_COMPARISONS:

    p_value = (
        delong_results[
            (
                method_a,
                method_b,
            )
        ][
            "p_value"
        ]
    )

    x1 = method_to_x[
        method_a
    ]

    x2 = method_to_x[
        method_b
    ]

    # Always draw left -> right
    if x1 > x2:
        x1, x2 = (
            x2,
            x1,
        )

    add_pvalue_bracket(
        ax=ax,
        x1=x1,
        x2=x2,
        y=bracket_y,
        height=0.0025,
        text=format_p_value(
            p_value
        ),
        fontsize=8.8,
    )


# ============================================================
# 15. Layout
# ============================================================

plt.tight_layout()


# ============================================================
# 16. Save
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=600,
    bbox_inches="tight",
)


PDF_PATH = (
    OUTPUT_PATH.replace(
        ".png",
        ".pdf",
    )
)

plt.savefig(
    PDF_PATH,
    bbox_inches="tight",
)


plt.show()


print()
print("=" * 72)
print("Saved figures")
print("=" * 72)

print(
    OUTPUT_PATH
)

print(
    PDF_PATH
)