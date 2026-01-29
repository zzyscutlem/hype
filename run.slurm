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

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate xx

# 切换到工作目录
cd /public/home/202520143336/

# 设置环境变量以避免MKL冲突
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

python xx.py


