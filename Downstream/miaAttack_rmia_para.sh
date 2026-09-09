#!/bin/bash
#SBATCH --job-name=rmia
#SBATCH --partition=gpu
#SBATCH --nodelist=elab-gpu-ubuntu04
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


# ============================================================
# User configuration
# ============================================================

HOME_DIR="/path/to/workspace"
LOG_DIR="${HOME_DIR}/logs"

PYTHON_EXEC="${HOME_DIR}/miniconda3/envs/diffusers311/bin/python"

#TRAIN_SCRIPT="/path/to/workspace/code/DeepUWF/MIA_attack/LiRA/main_rmia_vit_val.py"
TRAIN_SCRIPT="/path/to/workspace/code/DeepUWF/MIA_attack/LiRA/main_rmia_patient.py"


# ============================================================
# Configure experiments
# ============================================================

trainedSeeds=(
0
)

models=(
    "DP32" #baseline,ours
)


mkdir -p "$LOG_DIR"


# ============================================================
# Cache directories
# ============================================================

export HF_HOME="${HOME_DIR}/hf-cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

export PIP_CACHE_DIR="${HOME_DIR}/pip-cache"
export XDG_CACHE_HOME="${HOME_DIR}/xdg-cache"
export TORCH_HOME="${HOME_DIR}/torch-cache"


mkdir -p \
    "$HF_HOME" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" \
    "$PIP_CACHE_DIR" \
    "$XDG_CACHE_HOME" \
    "$TORCH_HOME"


# ============================================================
# Master log
# ============================================================

MASTER_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

#MASTER_LOG="${LOG_DIR}/${MASTER_TIMESTAMP}_rmia_master_job${SLURM_JOB_ID:-unknown}.out"
MASTER_LOG="${LOG_DIR}/${MASTER_TIMESTAMP}_rmia_master_patient_job${SLURM_JOB_ID:-unknown}.out"

exec > "$MASTER_LOG" 2>&1


echo "============================================================"
echo "RMIA batch job started"
echo "============================================================"
echo "Time:         $(date)"
echo "Host:         $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-UNSET}"
echo "Master log:   ${MASTER_LOG}"
echo "============================================================"


# ============================================================
# Environment check
# ============================================================

if [ ! -x "$PYTHON_EXEC" ]; then
    echo "ERROR: Python executable not found:"
    echo "$PYTHON_EXEC"
    exit 1
fi


if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: RMIA script not found:"
    echo "$TRAIN_SCRIPT"
    exit 1
fi


echo
echo "Using Python:"
echo "$PYTHON_EXEC"

"$PYTHON_EXEC" -V


"$PYTHON_EXEC" -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())

if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))
"


echo
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-UNSET}"

nvidia-smi || true


# ============================================================
# Experiment information
# ============================================================

NUM_SEEDS=${#trainedSeeds[@]}
NUM_MODELS=${#models[@]}
TOTAL_RUNS=$((NUM_SEEDS * NUM_MODELS))


echo
echo "============================================================"
echo "Experiment configuration"
echo "============================================================"

echo "Seeds:"
printf '  %s\n' "${trainedSeeds[@]}"

echo
echo "Models:"
printf '  %s\n' "${models[@]}"

echo
echo "Total runs: ${TOTAL_RUNS}"


# ============================================================
# Launch all experiments
#
# IMPORTANT:
# "&" means DO NOT wait for the current experiment.
# ============================================================

RUN_INDEX=0

for trainedSeed in "${trainedSeeds[@]}"; do

    for model in "${models[@]}"; do

        RUN_INDEX=$((RUN_INDEX + 1))

        TIMESTAMP=$(date +%Y%m%d_%H%M%S)

        #LOG_NAME="${TIMESTAMP}_miaAttack_rmia_precise_alltrainset_bestrp_seed${trainedSeed}_${model}"
        LOG_NAME="${TIMESTAMP}_miaAttack_rmia_precise_alltrainset_patient_bestrp_seed${trainedSeed}_${model}"

        LOG_FILE="${LOG_DIR}/${LOG_NAME}.out"


        echo
        echo "============================================================"
        echo "Launching ${RUN_INDEX}/${TOTAL_RUNS}"
        echo "trainedSeed = ${trainedSeed}"
        echo "model       = ${model}"
        echo "log         = ${LOG_FILE}"
        echo "============================================================"


        # ----------------------------------------------------
        # Launch and immediately continue to next configuration
        # ----------------------------------------------------

        srun \
            --exclusive \
            "$PYTHON_EXEC" -u "$TRAIN_SCRIPT" \
            --trainedSeed "$trainedSeed" \
            --model "$model" \
            > "$LOG_FILE" 2>&1 &


        PID=$!

        echo "Launched background PID: ${PID}"

    done

done


# ============================================================
# All configurations have now been launched
# ============================================================

echo
echo "============================================================"
echo "All ${TOTAL_RUNS} experiments have been launched."
echo "Waiting for all background processes to finish..."
echo "============================================================"


# Keep the SLURM allocation alive until all srun processes finish.
wait


echo
echo "============================================================"
echo "All RMIA experiments finished"
echo "Time: $(date)"
echo "============================================================"
