"""Confidence-based membership inference attack."""

import json
import os
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)

from sklearn.model_selection import (
    StratifiedKFold,
)

from scipy import stats

import matplotlib.pyplot as plt

print("Running confidence-based membership inference attack")
# ============================================================
# 0. Global configuration
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Using device: {device}"
)


# ============================================================
# Unified MIA convention
#
# Member     = 1
# Non-member = 0
#
# Larger attack score => more likely to be a member
# ============================================================


# ============================================================
# 1. Paths
# ============================================================

#basepathname = "random_attack_results_robust_subspace_correcttrainset/"
basepathname = "SHDR_npdpbaseline_adamw_softmax_results_trainset/"
basepathname = "SHDR_ep8_adamw_softmax_results_trainset/"
basepathname = "SHDR_ep1_adamw_softmax_results_trainset/"
basepathname = "SHDR_ep32_adamw_softmax_results_trainset/"

print(f"basepathname:{basepathname}")
# json_path_train = (
#     "/path/to/workspace/program_output/"
#     "DeepUWF/"
#     "random_attack_results_robust_subspace_correcttrainset/"
#     "softmax_results.json"
# )

# json_path_test = (
#     "/path/to/workspace/program_output/"
#     "DeepUWF/"
#     "random_attack_results_robust_subspace_correcttrainset/"
#     "softmax_results_test.json"
# )

# output_dir = Path(
#     "/path/to/workspace/code/"
#     "DeepUWF/MIA_attack/experiments/confidence_mia_results_v2"
# )

json_path_train = (
    "/path/to/workspace/program_output/"
    "DeepUWF/"
    f"{basepathname}"
    "softmax_results.json"
)

json_path_test = (
    "/path/to/workspace/program_output/"
    "DeepUWF/"
    f"{basepathname}"
    "softmax_results_test.json"
)

output_dir = Path(
    "/path/to/workspace/code/"
    f"DeepUWF/MIA_attack/experiments/confidence_mia_results_{basepathname}"
)


output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Load softmax vectors
# ============================================================

def load_softmax_results(
    json_path,
):
    """
    Load softmax output vectors from JSON.

    Expected JSON format:

        {
            "class_0/example1.png": [p0, p1],
            "class_1/example2.png": [p0, p1]
        }

    Returns:
        np.ndarray [N, C]
    """

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    softmax_probs = []

    for image_name in sorted(
        data.keys()
    ):
        softmax_probs.append(
            data[image_name]
        )

    probs = np.asarray(
        softmax_probs,
        dtype=np.float32,
    )

    if probs.ndim != 2:
        raise ValueError(
            f"Expected a 2-D softmax array, "
            f"but received shape {probs.shape}"
        )

    if len(probs) == 0:
        raise ValueError(
            f"No samples found in {json_path}"
        )

    if not np.all(
        np.isfinite(probs)
    ):
        raise ValueError(
            f"Non-finite softmax values found in "
            f"{json_path}"
        )

    return probs


member_probs = load_softmax_results(
    json_path_train
)

non_member_probs = (
    load_softmax_results(
        json_path_test
    )
)


print(
    "=" * 70
)

print(
    "Confidence-vector Membership Inference Attack"
)

print(
    "=" * 70
)

print(
    f"Member samples: "
    f"{len(member_probs)}"
)

print(
    f"Non-member samples: "
    f"{len(non_member_probs)}"
)

print(
    f"Softmax feature dimension: "
    f"{member_probs.shape[1]}"
)

print(
    f"Example member softmax: "
    f"{member_probs[0]}"
)

print(
    f"Example non-member softmax: "
    f"{non_member_probs[0]}"
)


if (
    member_probs.shape[1]
    != non_member_probs.shape[1]
):
    raise ValueError(
        "Member and non-member softmax "
        "dimensions do not match."
    )


# ============================================================
# 3. Build MIA dataset
# ============================================================

X = np.vstack(
    [
        member_probs,
        non_member_probs,
    ]
).astype(
    np.float32
)

# Member = 1
# Non-member = 0
y = np.concatenate(
    [
        np.ones(
            len(member_probs),
            dtype=np.float32,
        ),

        np.zeros(
            len(non_member_probs),
            dtype=np.float32,
        ),
    ]
)


print()

print(
    f"Total MIA samples: "
    f"{len(X)}"
)

print(
    f"Member ratio: "
    f"{np.mean(y):.4f}"
)

majority_accuracy = max(
    np.mean(y == 1),
    np.mean(y == 0),
)

print(
    f"Majority-class baseline accuracy: "
    f"{majority_accuracy:.4f}"
)


# ============================================================
# 4. MLP attack model
# ============================================================

class AttackModel(
    nn.Module
):
    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                64,
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                32,
            ),
            nn.ReLU(),

            nn.Linear(
                32,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


# ============================================================
# 5. Transformer attack model
# ============================================================

class MultiHeadAttention(
    nn.Module
):
    def __init__(
        self,
        input_dim,
        num_heads=2,
    ):
        super().__init__()

        self.num_heads = (
            num_heads
        )

        self.input_dim = (
            input_dim
        )

        if (
            input_dim
            % num_heads
            != 0
        ):
            raise ValueError(
                "input_dim must be divisible "
                "by num_heads"
            )

        self.head_dim = (
            input_dim
            // num_heads
        )

        self.query = nn.Linear(
            input_dim,
            input_dim,
        )

        self.key = nn.Linear(
            input_dim,
            input_dim,
        )

        self.value = nn.Linear(
            input_dim,
            input_dim,
        )

        self.out = nn.Linear(
            input_dim,
            input_dim,
        )

        self.scale = (
            self.head_dim
            ** 0.5
        )

    def forward(
        self,
        x,
    ):
        batch_size = (
            x.size(0)
        )

        Q = (
            self.query(x)
            .view(
                batch_size,
                -1,
                self.num_heads,
                self.head_dim,
            )
            .transpose(
                1,
                2,
            )
        )

        K = (
            self.key(x)
            .view(
                batch_size,
                -1,
                self.num_heads,
                self.head_dim,
            )
            .transpose(
                1,
                2,
            )
        )

        V = (
            self.value(x)
            .view(
                batch_size,
                -1,
                self.num_heads,
                self.head_dim,
            )
            .transpose(
                1,
                2,
            )
        )

        scores = (
            torch.matmul(
                Q,
                K.transpose(
                    -2,
                    -1,
                ),
            )
            / self.scale
        )

        attention = (
            torch.softmax(
                scores,
                dim=-1,
            )
        )

        out = torch.matmul(
            attention,
            V,
        )

        out = (
            out.transpose(
                1,
                2,
            )
            .contiguous()
            .view(
                batch_size,
                -1,
                self.input_dim,
            )
        )

        return self.out(
            out
        )


class TransformerBlock(
    nn.Module
):
    def __init__(
        self,
        input_dim,
        num_heads=2,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = (
            MultiHeadAttention(
                input_dim=input_dim,
                num_heads=num_heads,
            )
        )

        self.norm1 = nn.LayerNorm(
            input_dim
        )

        self.norm2 = nn.LayerNorm(
            input_dim
        )

        self.feed_forward = (
            nn.Sequential(
                nn.Linear(
                    input_dim,
                    input_dim * 4,
                ),
                nn.ReLU(),

                nn.Linear(
                    input_dim * 4,
                    input_dim,
                ),
            )
        )

        self.dropout = (
            nn.Dropout(
                dropout
            )
        )

    def forward(
        self,
        x,
    ):
        attn_output = (
            self.attention(
                x
            )
        )

        x = self.norm1(
            x
            + self.dropout(
                attn_output
            )
        )

        ff_output = (
            self.feed_forward(
                x
            )
        )

        x = self.norm2(
            x
            + self.dropout(
                ff_output
            )
        )

        return x


class AttackModel2(
    nn.Module
):
    def __init__(
        self,
        input_dim,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = (
            input_dim
        )

        self.embedding = (
            nn.Linear(
                1,
                input_dim,
            )
        )

        self.transformer_blocks = (
            nn.ModuleList(
                [
                    TransformerBlock(
                        input_dim=input_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                    for _ in range(
                        num_layers
                    )
                ]
            )
        )

        self.classifier = (
            nn.Sequential(
                nn.Linear(
                    input_dim,
                    64,
                ),
                nn.ReLU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    64,
                    32,
                ),
                nn.ReLU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    32,
                    1,
                ),
                nn.Sigmoid(),
            )
        )

    def forward(
        self,
        x,
    ):
        # [B, C]
        #
        # ->
        #
        # [B, C, 1]

        x = x.unsqueeze(
            -1
        )

        x = self.embedding(
            x
        )

        for block in (
            self.transformer_blocks
        ):
            x = block(
                x
            )

        # Global average over
        # confidence-vector tokens.
        x = torch.mean(
            x,
            dim=1,
        )

        return self.classifier(
            x
        )


# ============================================================
# 6. Stratified 5-fold CV
# ============================================================

N_SPLITS = 5
ATTACK_EPOCHS = 50
ATTACK_LR = 1e-3

input_dim = (
    X.shape[1]
)

num_heads = min(
    input_dim,
    2,
)

if (
    input_dim
    % num_heads
    != 0
):
    num_heads = 1


kfold = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)


mlp_accuracies = []
mlp_balanced_accuracies = []
mlp_aucs = []

transformer_accuracies = []
transformer_balanced_accuracies = []
transformer_aucs = []


# Out-of-fold predictions
all_mlp_predictions = []
all_mlp_labels = []

all_transformer_predictions = []
all_transformer_labels = []


criterion = nn.BCELoss()


print()

print(
    f"Starting {N_SPLITS}-fold "
    f"stratified cross-validation"
)

print(
    "=" * 70
)


for fold, (
    train_idx,
    test_idx,
) in enumerate(
    kfold.split(
        X,
        y,
    )
):
    print()

    print(
        f"Fold {fold + 1}/{N_SPLITS}"
    )

    print(
        "-" * 50
    )

    X_train = (
        X[train_idx]
    )

    X_test = (
        X[test_idx]
    )

    y_train = (
        y[train_idx]
    )

    y_test = (
        y[test_idx]
    )


    print(
        f"Train member/non-member: "
        f"{int(np.sum(y_train == 1))}/"
        f"{int(np.sum(y_train == 0))}"
    )

    print(
        f"Test member/non-member: "
        f"{int(np.sum(y_test == 1))}/"
        f"{int(np.sum(y_test == 0))}"
    )


    X_train_tensor = (
        torch.from_numpy(
            X_train
        )
        .float()
        .to(
            device
        )
    )

    y_train_tensor = (
        torch.from_numpy(
            y_train
        )
        .float()
        .unsqueeze(
            1
        )
        .to(
            device
        )
    )

    X_test_tensor = (
        torch.from_numpy(
            X_test
        )
        .float()
        .to(
            device
        )
    )


    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    mlp_model = (
        AttackModel(
            input_dim=input_dim
        )
        .to(
            device
        )
    )

    mlp_optimizer = (
        optim.Adam(
            mlp_model.parameters(),
            lr=ATTACK_LR,
        )
    )


    print(
        "Training MLP attack model..."
    )

    for epoch in range(
        ATTACK_EPOCHS
    ):
        mlp_model.train()

        mlp_optimizer.zero_grad(
            set_to_none=True
        )

        outputs = (
            mlp_model(
                X_train_tensor
            )
        )

        loss = criterion(
            outputs,
            y_train_tensor,
        )

        loss.backward()

        mlp_optimizer.step()


    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    transformer_model = (
        AttackModel2(
            input_dim=input_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=0.1,
        )
        .to(
            device
        )
    )

    transformer_optimizer = (
        optim.Adam(
            transformer_model.parameters(),
            lr=ATTACK_LR,
        )
    )


    print(
        "Training Transformer attack model..."
    )

    for epoch in range(
        ATTACK_EPOCHS
    ):
        transformer_model.train()

        transformer_optimizer.zero_grad(
            set_to_none=True
        )

        outputs = (
            transformer_model(
                X_train_tensor
            )
        )

        loss = criterion(
            outputs,
            y_train_tensor,
        )

        loss.backward()

        transformer_optimizer.step()


    # --------------------------------------------------------
    # Evaluate MLP
    # --------------------------------------------------------

    mlp_model.eval()

    with torch.no_grad():
        predictions = (
            mlp_model(
                X_test_tensor
            )
            .squeeze(
                1
            )
            .detach()
            .cpu()
            .numpy()
        )

    pred_labels = (
        predictions
        >= 0.5
    ).astype(
        np.int64
    )


    mlp_accuracy = (
        accuracy_score(
            y_test,
            pred_labels,
        )
    )

    mlp_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            pred_labels,
        )
    )

    mlp_auc = (
        roc_auc_score(
            y_test,
            predictions,
        )
    )


    mlp_accuracies.append(
        mlp_accuracy
    )

    mlp_balanced_accuracies.append(
        mlp_balanced_accuracy
    )

    mlp_aucs.append(
        mlp_auc
    )


    all_mlp_predictions.extend(
        predictions.tolist()
    )

    all_mlp_labels.extend(
        y_test.tolist()
    )


    # --------------------------------------------------------
    # Evaluate Transformer
    # --------------------------------------------------------

    transformer_model.eval()

    with torch.no_grad():
        predictions2 = (
            transformer_model(
                X_test_tensor
            )
            .squeeze(
                1
            )
            .detach()
            .cpu()
            .numpy()
        )

    pred_labels2 = (
        predictions2
        >= 0.5
    ).astype(
        np.int64
    )


    transformer_accuracy = (
        accuracy_score(
            y_test,
            pred_labels2,
        )
    )

    transformer_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            pred_labels2,
        )
    )

    transformer_auc = (
        roc_auc_score(
            y_test,
            predictions2,
        )
    )


    transformer_accuracies.append(
        transformer_accuracy
    )

    transformer_balanced_accuracies.append(
        transformer_balanced_accuracy
    )

    transformer_aucs.append(
        transformer_auc
    )


    all_transformer_predictions.extend(
        predictions2.tolist()
    )

    all_transformer_labels.extend(
        y_test.tolist()
    )


    print(
        f"MLP:"
    )

    print(
        f"  Accuracy:          "
        f"{mlp_accuracy:.4f}"
    )

    print(
        f"  Balanced Accuracy: "
        f"{mlp_balanced_accuracy:.4f}"
    )

    print(
        f"  AUROC:             "
        f"{mlp_auc:.4f}"
    )


    print(
        f"Transformer:"
    )

    print(
        f"  Accuracy:          "
        f"{transformer_accuracy:.4f}"
    )

    print(
        f"  Balanced Accuracy: "
        f"{transformer_balanced_accuracy:.4f}"
    )

    print(
        f"  AUROC:             "
        f"{transformer_auc:.4f}"
    )


# ============================================================
# 7. CV statistics
# ============================================================

mlp_acc_mean = float(
    np.mean(
        mlp_accuracies
    )
)

mlp_acc_std = float(
    np.std(
        mlp_accuracies
    )
)

mlp_bal_acc_mean = float(
    np.mean(
        mlp_balanced_accuracies
    )
)

mlp_bal_acc_std = float(
    np.std(
        mlp_balanced_accuracies
    )
)

mlp_auc_mean = float(
    np.mean(
        mlp_aucs
    )
)

mlp_auc_std = float(
    np.std(
        mlp_aucs
    )
)


transformer_acc_mean = float(
    np.mean(
        transformer_accuracies
    )
)

transformer_acc_std = float(
    np.std(
        transformer_accuracies
    )
)

transformer_bal_acc_mean = float(
    np.mean(
        transformer_balanced_accuracies
    )
)

transformer_bal_acc_std = float(
    np.std(
        transformer_balanced_accuracies
    )
)

transformer_auc_mean = float(
    np.mean(
        transformer_aucs
    )
)

transformer_auc_std = float(
    np.std(
        transformer_aucs
    )
)


# ============================================================
# 8. Pooled out-of-fold statistics
# ============================================================

all_mlp_predictions = np.asarray(
    all_mlp_predictions,
    dtype=np.float64,
)

all_mlp_labels = np.asarray(
    all_mlp_labels,
    dtype=np.int64,
)

all_transformer_predictions = (
    np.asarray(
        all_transformer_predictions,
        dtype=np.float64,
    )
)

all_transformer_labels = (
    np.asarray(
        all_transformer_labels,
        dtype=np.int64,
    )
)


mlp_overall_auc = (
    roc_auc_score(
        all_mlp_labels,
        all_mlp_predictions,
    )
)

transformer_overall_auc = (
    roc_auc_score(
        all_transformer_labels,
        all_transformer_predictions,
    )
)


print()

print(
    "=" * 70
)

print(
    f"{N_SPLITS}-Fold Cross-Validation Results"
)

print(
    "=" * 70
)


print(
    "\n[MLP]"
)

print(
    f"Accuracy: "
    f"{mlp_acc_mean:.4f} "
    f"± {mlp_acc_std:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{mlp_bal_acc_mean:.4f} "
    f"± {mlp_bal_acc_std:.4f}"
)

print(
    f"Fold AUROC: "
    f"{mlp_auc_mean:.4f} "
    f"± {mlp_auc_std:.4f}"
)

print(
    f"Pooled OOF AUROC: "
    f"{mlp_overall_auc:.4f}"
)


print(
    "\n[Transformer]"
)

print(
    f"Accuracy: "
    f"{transformer_acc_mean:.4f} "
    f"± {transformer_acc_std:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{transformer_bal_acc_mean:.4f} "
    f"± {transformer_bal_acc_std:.4f}"
)

print(
    f"Fold AUROC: "
    f"{transformer_auc_mean:.4f} "
    f"± {transformer_auc_std:.4f}"
)

print(
    f"Pooled OOF AUROC: "
    f"{transformer_overall_auc:.4f}"
)


print(
    "\nFold MLP AUROCs:"
)

print(
    [
        f"{value:.4f}"
        for value in mlp_aucs
    ]
)

print(
    "Fold Transformer AUROCs:"
)

print(
    [
        f"{value:.4f}"
        for value in transformer_aucs
    ]
)


# ============================================================
# 9. Statistical comparison
# ============================================================

t_stat_auc, p_value_auc = (
    stats.ttest_rel(
        mlp_aucs,
        transformer_aucs,
    )
)

t_stat_bal_acc, p_value_bal_acc = (
    stats.ttest_rel(
        mlp_balanced_accuracies,
        transformer_balanced_accuracies,
    )
)


print()

print(
    "[Paired Statistical Tests]"
)

print(
    f"AUROC paired t-test: "
    f"t={t_stat_auc:.4f}, "
    f"p={p_value_auc:.6f}"
)

print(
    f"Balanced Accuracy paired t-test: "
    f"t={t_stat_bal_acc:.4f}, "
    f"p={p_value_bal_acc:.6f}"
)


# ============================================================
# 10. Low-FPR TPR
# ============================================================

def get_tpr_at_fpr(
    labels,
    predictions,
    target_fprs=(
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
    ),
):
    """
    Maximum empirical TPR achievable while
    FPR <= target_fpr.
    """

    labels = np.asarray(
        labels
    )

    predictions = np.asarray(
        predictions
    )

    fpr, tpr, _ = roc_curve(
        labels,
        predictions,
        pos_label=1,
    )

    results = {}

    for target_fpr in (
        target_fprs
    ):
        valid = (
            fpr
            <= target_fpr
        )

        if np.any(
            valid
        ):
            results[
                target_fpr
            ] = float(
                np.max(
                    tpr[
                        valid
                    ]
                )
            )

        else:
            results[
                target_fpr
            ] = 0.0

    return results


mlp_low_fpr = (
    get_tpr_at_fpr(
        all_mlp_labels,
        all_mlp_predictions,
    )
)

transformer_low_fpr = (
    get_tpr_at_fpr(
        all_transformer_labels,
        all_transformer_predictions,
    )
)


print()

print(
    "[Low-FPR TPR]"
)

for target_fpr in (
    mlp_low_fpr
):
    print(
        f"FPR <= {target_fpr:.4g}: "
        f"MLP TPR="
        f"{mlp_low_fpr[target_fpr]:.6f}, "
        f"Transformer TPR="
        f"{transformer_low_fpr[target_fpr]:.6f}"
    )


# ============================================================
# 11. Standard ROC
# ============================================================

def plot_standard_roc(
    mlp_predictions,
    mlp_labels,
    transformer_predictions,
    transformer_labels,
    mlp_auc,
    transformer_auc,
    save_path,
):
    mlp_fpr, mlp_tpr, _ = (
        roc_curve(
            mlp_labels,
            mlp_predictions,
            pos_label=1,
        )
    )

    transformer_fpr, transformer_tpr, _ = (
        roc_curve(
            transformer_labels,
            transformer_predictions,
            pos_label=1,
        )
    )


    plt.figure(
        figsize=(8, 7)
    )

    plt.plot(
        mlp_fpr,
        mlp_tpr,
        linewidth=2.2,
        label=(
            f"MLP "
            f"(AUC = {mlp_auc:.4f})"
        ),
    )

    plt.plot(
        transformer_fpr,
        transformer_tpr,
        linewidth=2.2,
        label=(
            f"Transformer "
            f"(AUC = {transformer_auc:.4f})"
        ),
    )

    plt.plot(
        [0, 1],
        [0, 1],
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
        "Confidence-based Membership Inference ROC",
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
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 12. Log-log ROC
# ============================================================

def plot_loglog_roc(
    mlp_predictions,
    mlp_labels,
    transformer_predictions,
    transformer_labels,
    mlp_auc,
    transformer_auc,
    save_path,
):
    mlp_fpr, mlp_tpr, _ = (
        roc_curve(
            mlp_labels,
            mlp_predictions,
            pos_label=1,
        )
    )

    trans_fpr, trans_tpr, _ = (
        roc_curve(
            transformer_labels,
            transformer_predictions,
            pos_label=1,
        )
    )


    num_nonmember = min(
        int(
            np.sum(
                np.asarray(
                    mlp_labels
                )
                == 0
            )
        ),
        int(
            np.sum(
                np.asarray(
                    transformer_labels
                )
                == 0
            )
        ),
    )

    num_member = min(
        int(
            np.sum(
                np.asarray(
                    mlp_labels
                )
                == 1
            )
        ),
        int(
            np.sum(
                np.asarray(
                    transformer_labels
                )
                == 1
            )
        ),
    )


    min_rate = max(
        1.0 / num_nonmember,
        1.0 / num_member,
    )


    # Plot copies only.
    mlp_fpr_plot = np.clip(
        mlp_fpr,
        min_rate,
        1.0,
    )

    mlp_tpr_plot = np.clip(
        mlp_tpr,
        min_rate,
        1.0,
    )

    trans_fpr_plot = np.clip(
        trans_fpr,
        min_rate,
        1.0,
    )

    trans_tpr_plot = np.clip(
        trans_tpr,
        min_rate,
        1.0,
    )


    random_rate = np.geomspace(
        min_rate,
        1.0,
        500,
    )


    plt.figure(
        figsize=(8, 7)
    )

    plt.plot(
        mlp_fpr_plot,
        mlp_tpr_plot,
        linewidth=2.2,
        label=(
            f"MLP "
            f"(AUC = {mlp_auc:.4f})"
        ),
    )

    plt.plot(
        trans_fpr_plot,
        trans_tpr_plot,
        linewidth=2.2,
        label=(
            f"Transformer "
            f"(AUC = {transformer_auc:.4f})"
        ),
    )

    plt.plot(
        random_rate,
        random_rate,
        linestyle="--",
        linewidth=1.5,
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
        "Confidence-based Membership Inference ROC",
        fontsize=14,
    )

    plt.grid(
        True,
        which="both",
        linestyle=":",
        alpha=0.5,
    )

    plt.legend(
        loc="lower right",
        fontsize=11,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 13. LiRA-style semilog ROC
# ============================================================

def plot_semilog_roc(
    mlp_predictions,
    mlp_labels,
    transformer_predictions,
    transformer_labels,
    mlp_auc,
    transformer_auc,
    save_path,
):
    mlp_fpr, mlp_tpr, _ = (
        roc_curve(
            mlp_labels,
            mlp_predictions,
            pos_label=1,
        )
    )

    trans_fpr, trans_tpr, _ = (
        roc_curve(
            transformer_labels,
            transformer_predictions,
            pos_label=1,
        )
    )


    mlp_nonmember_count = int(
        np.sum(
            np.asarray(
                mlp_labels
            )
            == 0
        )
    )

    trans_nonmember_count = int(
        np.sum(
            np.asarray(
                transformer_labels
            )
            == 0
        )
    )


    min_fpr = max(
        1.0 / mlp_nonmember_count,
        1.0 / trans_nonmember_count,
    )


    mlp_mask = (
        mlp_fpr
        > 0
    )

    trans_mask = (
        trans_fpr
        > 0
    )


    mlp_fpr_plot = (
        mlp_fpr[
            mlp_mask
        ]
    )

    mlp_tpr_plot = (
        mlp_tpr[
            mlp_mask
        ]
    )

    trans_fpr_plot = (
        trans_fpr[
            trans_mask
        ]
    )

    trans_tpr_plot = (
        trans_tpr[
            trans_mask
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
        mlp_fpr_plot,
        mlp_tpr_plot,
        linewidth=2.2,
        label=(
            f"MLP "
            f"(AUC = {mlp_auc:.4f})"
        ),
    )

    plt.semilogx(
        trans_fpr_plot,
        trans_tpr_plot,
        linewidth=2.2,
        label=(
            f"Transformer "
            f"(AUC = {transformer_auc:.4f})"
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
        "Confidence-based Membership Inference ROC",
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
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 14. Save figures
# ============================================================

standard_roc_path = (
    output_dir
    / "mia_attack_roc_curves.png"
)

loglog_roc_path = (
    output_dir
    / "mia_attack_roc_curves_loglog.png"
)

semilog_roc_path = (
    output_dir
    / "mia_lira_semilog_roc.png"
)


plot_standard_roc(
    mlp_predictions=(
        all_mlp_predictions
    ),
    mlp_labels=(
        all_mlp_labels
    ),
    transformer_predictions=(
        all_transformer_predictions
    ),
    transformer_labels=(
        all_transformer_labels
    ),
    mlp_auc=(
        mlp_overall_auc
    ),
    transformer_auc=(
        transformer_overall_auc
    ),
    save_path=(
        standard_roc_path
    ),
)


plot_loglog_roc(
    mlp_predictions=(
        all_mlp_predictions
    ),
    mlp_labels=(
        all_mlp_labels
    ),
    transformer_predictions=(
        all_transformer_predictions
    ),
    transformer_labels=(
        all_transformer_labels
    ),
    mlp_auc=(
        mlp_overall_auc
    ),
    transformer_auc=(
        transformer_overall_auc
    ),
    save_path=(
        loglog_roc_path
    ),
)


plot_semilog_roc(
    mlp_predictions=(
        all_mlp_predictions
    ),
    mlp_labels=(
        all_mlp_labels
    ),
    transformer_predictions=(
        all_transformer_predictions
    ),
    transformer_labels=(
        all_transformer_labels
    ),
    mlp_auc=(
        mlp_overall_auc
    ),
    transformer_auc=(
        transformer_overall_auc
    ),
    save_path=(
        semilog_roc_path
    ),
)


print()

print(
    "Saved figures:"
)

print(
    standard_roc_path
)

print(
    loglog_roc_path
)

print(
    semilog_roc_path
)


# ============================================================
# 15. Save out-of-fold attack predictions
# ============================================================

prediction_path = (
    output_dir
    / "confidence_mia_oof_predictions.npz"
)


np.savez(
    prediction_path,

    mlp_predictions=(
        all_mlp_predictions
    ),

    mlp_labels=(
        all_mlp_labels
    ),

    transformer_predictions=(
        all_transformer_predictions
    ),

    transformer_labels=(
        all_transformer_labels
    ),
)


print(
    f"\nOOF predictions saved to: "
    f"{prediction_path}"
)


# ============================================================
# 16. Final interpretation
# ============================================================

strongest_auc = max(
    mlp_overall_auc,
    transformer_overall_auc,
)


print()

print(
    "=" * 70
)

print(
    "Final MIA Summary"
)

print(
    "=" * 70
)

print(
    f"MLP pooled AUROC: "
    f"{mlp_overall_auc:.6f}"
)

print(
    f"Transformer pooled AUROC: "
    f"{transformer_overall_auc:.6f}"
)

print(
    f"Strongest attack AUROC: "
    f"{strongest_auc:.6f}"
)


if strongest_auc < 0.55:
    print(
        "Conclusion: confidence-vector MIA "
        "is close to random guessing."
    )

elif strongest_auc < 0.60:
    print(
        "Conclusion: weak membership signal "
        "is detectable."
    )

else:
    print(
        "Conclusion: confidence-vector MIA "
        "detects meaningful membership leakage."
    )
