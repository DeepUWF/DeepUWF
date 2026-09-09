#!/bin/bash
#SBATCH --job-name=mia        
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
LOG_NAME="loadnpz_loglog_comparison_patientSHDR"



mkdir -p "$LOG_DIR"

MASTER_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

MASTER_LOG="${LOG_DIR}/${MASTER_TIMESTAMP}_rmiaAttack_master_job${SLURM_JOB_ID:-unknown}.out"

exec > "$MASTER_LOG" 2>&1


echo "============================================================"
echo "RMIA batch job started"
echo "============================================================"
echo "Time:         $(date)"
echo "Host:         $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-UNSET}"
echo "Master log:   ${MASTER_LOG}"
echo "============================================================"




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

# ---- 核心任务执行 (加回 srun，并使用绝对路径 Python) ----
#srun $PYTHON_EXEC -u /path/to/workspace/code/DeepUWF/MIA_attack/confidence_attack.py  
#srun $PYTHON_EXEC -u /path/to/workspace/code/DeepUWF/MIA_attack/gradient_attack.py  
#srun $PYTHON_EXEC -u /path/to/workspace/code/DeepUWF/MIA_attack/plot_gradient_loglog_all.py  
#srun $PYTHON_EXEC -u /path/to/workspace/code/DeepUWF/MIA_attack/LiRA/loadnpz_loglog_comparison.py

#srun $PYTHON_EXEC -u /path/to/workspace/code/DeepUWF/MIA_attack/plot_confidence_loglog_all.py
#srun $PYTHON_EXEC -u /path/to/workspace/code/DeepUWF/DP/drawPTest_step2_v2.py
#srun $PYTHON_EXEC -u /path/to/workspace/code/DeepUWF/DP/patientAnalysis.py
srun $PYTHON_EXEC -u /path/to/workspace/code/DeepUWF/MIA_attack/LiRA/loadnpz_loglog_comparison_patient.py


# ---- run ----
