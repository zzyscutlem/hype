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


# HyPE on API-Bank SLURM Job Script
# This script runs the complete workflow on HPC cluster

echo "========================================================================" 
echo "  HyPE on API-Bank - SLURM Job"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"
echo ""

# Load modules (adjust based on your HPC environment)
# module load cuda/11.8
# module load cudnn/8.6

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ag

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; torch.cuda.is_available()' 2>/dev/null; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
else
    echo "GPU: N/A"
fi
echo ""

# Change to project directory
cd /share/home/202520143336/project/hype

# Create logs directory
mkdir -p logs

# Run the workflow
echo "Starting HyPE workflow..."
echo ""

./run_apibank_hpc.sh

echo ""
echo "========================================================================"
echo "Job completed at: $(date)"
echo "========================================================================"
