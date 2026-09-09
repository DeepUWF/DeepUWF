#!/bin/bash
#SBATCH --job-name=t1raBase          
#SBATCH --partition=gpu           
#SBATCH --nodelist=elab-gpu-ubuntu03                 
#SBATCH --gres=gpu:1                
#SBATCH --mem=10G                     
#SBATCH --output=/dev/null       
#SBATCH --error=/dev/null        

# ====== 用户可配置变量 ======

# ---- 路径配置 (使用你真实的绝对路径) ----
# 根据文档，你的家目录是 /path/to/workspace
HOME_DIR="/path/to/workspace"
LOG_DIR="${HOME_DIR}/logs"
LOG_NAME="trainDP_SHDR_adamw_nodp_seed0"

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

srun $PYTHON_EXEC -u /path/to/workspace/code/DeepUWF/DP/dp_fl_train.py  --modedp "train"  --config "/path/to/workspace/code/DeepUWF/DP/dp_fl_config_baseline.yaml" 
# ---- run ----
