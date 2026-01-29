#!/bin/bash
#SBATCH --job-name=hype_validation_local
#SBATCH --partition=gpuA800
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=14400
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=790335643@qq.com
#SBATCH --output=/share/home/202520143336/outputlog/%j.out
#SBATCH --error=/share/home/202520143336/outputlog/%j.err

echo "=========================================="
echo "HyPE Optimized Performance Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load environment
source ~/.bashrc
conda activate ag

# Change to project directory
cd /share/home/202520143336/project/ag

echo ""
echo "Python environment:"
which python
python --version

echo ""
echo "CUDA availability:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "Running HyPE Optimized Performance Benchmark"
echo "=========================================="
echo "Optimizations enabled:"
echo "- Early stopping (threshold: 0.7, min iterations: 5)"
echo "- Value prediction caching (size: 100)"
echo "- Reduced search budget (30 vs 100)"
echo "- Reduced tree depth (5 vs 10)"
echo "- Fewer hypotheses per node (2 vs 3)"
echo "Milvus Lite will be started automatically"
echo ""

# Set offline mode environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Run benchmark with 5 tasks per environment (reduced for faster testing)
python benchmark_hype_real_v1.py \
    --config config.yaml \
    --num-tasks 5 \
    --environments toolbench apibank alfworld \
    --output performance_results_optimized_${SLURM_JOB_ID}.json \
    --verbose

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Results saved to: performance_results_optimized_${SLURM_JOB_ID}.json"
    echo ""
    echo "Summary:"
    python -c "
import json
try:
    with open('performance_results_optimized_${SLURM_JOB_ID}.json', 'r') as f:
        results = json.load(f)
    for env_name, env_data in results.get('environments', {}).items():
        if 'error' not in env_data:
            print(f'{env_name}: {env_data.get(\"successful\", 0)}/{env_data.get(\"total_tasks\", 0)} successful ({env_data.get(\"success_rate\", 0)*100:.1f}%)')
            print(f'  Avg time: {env_data.get(\"avg_time\", 0):.2f}s, Avg model calls: {env_data.get(\"avg_model_calls\", 0):.1f}')
except Exception as e:
    print(f'Could not load results: {e}')
"
    echo ""
    echo "✅ Optimized benchmark completed successfully!"
else
    echo "❌ Benchmark failed!"
fi

exit $EXIT_CODE
