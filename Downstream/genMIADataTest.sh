#!/bin/bash
#SBATCH --job-name=subspace_attack          
#SBATCH --partition=gpu           
#SBATCH --nodelist=elab-gpu-ubuntu07               
#SBATCH --gres=gpu:1                
#SBATCH --mem=10G                     
#SBATCH --output=/dev/null       
#SBATCH --error=/dev/null        

# ====== 用户可配置变量 ======

# ---- 路径配置 (使用你真实的绝对路径) ----
# 根据文档，你的家目录是 /path/to/workspace
HOME_DIR="/path/to/workspace"
LOG_DIR="${HOME_DIR}/logs"
#LOG_NAME="trainDP_adamw_nodp"
#LOG_NAME="trainDP_adamw_dp10_12"
#LOG_NAME="trainDP_adamw_dp1_SHDR"
#LOG_NAME="trainDP_adamw_dp8_SHDR"
#LOG_NAME="testDP_adamw_nodp_model_standalone_best"
#LOG_NAME="attackFFM_SHDR_sample"
#LOG_NAME="attackFFM_SHDR_proto"
#LOG_NAME="attackFFM_SHDR_subject"
#LOG_NAME="attackFFM_SHDR_hierar"

#LOG_NAME="attackFFM_SHDR_sample_testset"
#LOG_NAME="attackFFM_SHDR_robustSubspace"
#LOG_NAME="attackFFM_SHDR_hierar_noclassproto"


LOG_NAME="genMIAData_SHDR_robustSubspace_correcttrainset"
LOG_NAME="genMIAData_SHDR_robustSubspace_correcttrainset_moredata_correct"
LOG_NAME="genMIAData_SHDR_robustSubspace_correcttrainset_moredata_testset"
LOG_NAME="genMIAData_SHDR_dp1_testset"
LOG_NAME="genMIAData_SHDR_dp32_testset"
#LOG_NAME="genMIAData_SHDR_dp8_testset"
#LOG_NAME="genMIAData_SHDR_npdpbaseline_testset"
LOG_NAME="genMIAData_SHDR_robustSubspace_correcttrainset_moremoredata_testset"

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

# =================================================================
# 🚨 终极解决方案：直接使用 Conda 环境中 Python 的绝对路径
# =================================================================
# 即使 source 和 conda activate 失败，只要直接调用这个绝对路径，
# 程序就会强制在你需要的 ediffusers311v 环境中运行！

#PYTHON_EXEC="${HOME_DIR}/miniconda3/envs/diffusers311/bin/python"

#echo "=== System Check ==="
#echo "Using Python interpreter at: $PYTHON_EXEC"
#$PYTHON_EXEC -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
#echo "===================="

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
srun $PYTHON_EXEC -u /path/to/workspace/code/DeepUWF/DP/dp_fl_train.py  --modedp "softmax" --train_paths_file "" --config "/path/to/workspace/code/DeepUWF/DP/dp_fl_config_mia_test.yaml"  
# ---- run ----
