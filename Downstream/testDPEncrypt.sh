#!/bin/bash
#SBATCH --job-name=genNAE          
#SBATCH --partition=gpu           
#SBATCH --nodelist=elab-gpu-ubuntu03                 
#SBATCH --gres=gpu:1                
#SBATCH --mem=10G                     
#SBATCH --output=/dev/null       
#SBATCH --error=/dev/null        

# ====== 用户可配置变量 ======

HOME_DIR="/path/to/workspace"
LOG_DIR="${HOME_DIR}/logs"
LOG_NAME="testDP_SHDR_random_attack_results_robust_subspace_pixrecons"



mkdir -p "$LOG_DIR"

# ---- 缓存目录设置 ----
export HF_HOME="${HOME_DIR}/hf-cache"
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PIP_CACHE_DIR="${HOME_DIR}/pip-cache"
export XDG_CACHE_HOME="${HOME_DIR}/xdg-cache"
export TORCH_HOME="${HOME_DIR}/torch-cache"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
 "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME"

# ---- 日志重定向 ----
echo "TARGET=$TARGET V1=$V1 DATASET=$DATASET MAXQ=$MAXQ EPSILON=$EPSILON"
echo "Logging to: ${LOG_DIR}/${LOG_NAME}.out"

exec > "${LOG_DIR}/${LOG_NAME}.out" 2>&1


echo "Using diffusers311 environment"
PYTHON_EXEC="${HOME_DIR}/miniconda3/envs/diffusers311/bin/python"


if [ ! -x "$PYTHON_EXEC" ]; then
    echo "ERROR: Python executable not found or not executable: $PYTHON_EXEC"
    exit 1
fi

echo "=== System Check ==="
echo "Using Python interpreter at: $PYTHON_EXEC"
"$PYTHON_EXEC" -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-UNSET}"
nvidia-smi || true
echo "===================="

MODEL_DIR="/path/to/workspace/weights/DeepUWFDP/SHDR_random_attack_results_robust_subspace_pixcons"


for EP in $(seq 0 5); do
    MODEL_PATH="${MODEL_DIR}/model_standalone_ep${EP}.pt"

    echo "============================================================"
    echo "Testing checkpoint: ${MODEL_PATH}"
    echo "Epoch: ${EP}"
    echo "Time: $(date)"
    echo "============================================================"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "WARNING: checkpoint not found, skip: ${MODEL_PATH}"
        continue
    fi

    srun "$PYTHON_EXEC" -u /path/to/workspace/code/DeepUWF/DP/dp_fl_train_encrypt.py \
        --modedp "test" \
        --modelpath "$MODEL_PATH" \
       --config "/path/to/workspace/code/DeepUWF/DP/dp_fl_config_test_encrypt.yaml" 


    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: failed at epoch ${EP}, model path: ${MODEL_PATH}"
        echo "Exit code: ${EXIT_CODE}"
        # 如果你希望某个 epoch 失败后立刻停止，取消下一行注释
        # exit $EXIT_CODE
    fi

    echo "Finished epoch ${EP}"
    echo
done

echo "All checkpoints finished."

# ---- run ----
