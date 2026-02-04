#!/bin/bash
#SBATCH --job-name=test          # 作业名
#SBATCH --partition=gpuA800                # GPU队列
#SBATCH -N 1                               # 1个节点
#SBATCH --ntasks-per-node=1                # 每节点1个任务
#SBATCH --cpus-per-task=8                  # 1:8的GPU:CPU配比
#SBATCH --gres=gpu:1                       # n块GPU
#SBATCH --time=14400                       # 10天
#SBATCH --mail-type=begin                  # 作业开始时发邮件
#SBATCH --mail-type=end                    # 作业结束时发邮件
#SBATCH --mail-user=790335643@qq.com
#SBATCH --output=/share/home/202520143336/outputlog/%j.out                 # 输出文件
#SBATCH --error=/share/home/202520143336/outputlog/%j.err                 # 输出文件


# 打印作业信息
echo "========================================================================"
echo "  HyPE on API-Bank - Cold Boot Test (5 Samples)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"
echo ""

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ag

# 设置 CUDA 环境变量
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX

# 🔥 禁用 bitsandbytes 量化
export BITSANDBYTES_NOWELCOME=1
export DISABLE_BITSANDBYTES=1

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 打印环境信息
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# 进入项目目录
cd /share/home/202520143336/project/hype

# 运行 cold boot（5个样例用于快速测试）
python cold_boot_apibank.py \
    --data-path ./data/apibank/train.json \
    --config config_hpc.yaml \
    --output-dir ./output/apibank_hpc/cold_boot_test \
    --num-tasks 10 \
    --evolution-interval 5 \
    --checkpoint-interval 5

echo ""
echo "========================================================================"
echo "Job completed at: $(date)"
echo "========================================================================"
