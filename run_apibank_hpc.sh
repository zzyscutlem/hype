#!/bin/bash
# HyPE on API-Bank for HPC Environment
# Uses simple in-memory principle storage (no Milvus needed)
# Designed for SLURM-based HPC clusters

set -e  # Exit on error

# Configuration
DATA_DIR="./data/apibank"
OUTPUT_DIR="./output/apibank_hpc"
CONFIG_FILE="config_hpc.yaml"
NUM_TRAIN_TASKS=5   # 只运行10个样例进行测试
NUM_TEST_TASKS=5     # 测试集也只运行5个
EVOLUTION_INTERVAL=5  # 每5个任务进行一次进化
CHECKPOINT_INTERVAL=5 # 每5个任务保存一次检查点

# Conda environment
CONDA_ENV="ag"

# Print banner
echo "========================================================================"
echo "  HyPE on API-Bank: HPC Workflow"
echo "========================================================================"
echo ""

# Activate conda environment
echo "[*] Activating conda environment: ${CONDA_ENV}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
echo "[✓] Conda environment '${CONDA_ENV}' activated"
echo ""

# Check Python and key packages
echo "[*] Checking Python environment..."
python --version
echo "Python path: $(which python)"
echo ""

# Check if required packages are installed
echo "[*] Checking required packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "[✗] PyTorch not found"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || echo "[✗] Transformers not found"
python -c "import sentence_transformers; print(f'Sentence-Transformers: {sentence_transformers.__version__}')" || echo "[✗] Sentence-Transformers not found"
echo "[✓] Package check complete"
echo ""

# Check GPU availability
echo "[*] Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Note about storage
echo "[*] Note: Using Milvus Lite (embedded mode)"
echo "    Principles will be stored in local database file"
echo "[✓] Milvus Lite will be used"
echo ""

# Step 1: Download API-Bank dataset
echo "========================================================================"
echo "  Step 1: Download API-Bank Dataset"
echo "========================================================================"
echo ""

if [ -f "${DATA_DIR}/train.json" ] && [ -f "${DATA_DIR}/test.json" ]; then
    echo "[✓] Dataset already downloaded"
    echo "    Train: ${DATA_DIR}/train.json"
    echo "    Test: ${DATA_DIR}/test.json"
else
    echo "[*] Downloading dataset..."
    python download_apibank.py --output-dir "${DATA_DIR}"
    echo "[✓] Dataset downloaded"
fi
echo ""

# Step 2: Cold boot on training set
echo "========================================================================"
echo "  Step 2: Cold Boot on Training Set"
echo "========================================================================"
echo ""

COLD_BOOT_DIR="${OUTPUT_DIR}/cold_boot"

if [ -f "${COLD_BOOT_DIR}/checkpoints/checkpoint_final.pt" ]; then
    echo "[!] Cold boot checkpoint already exists"
    echo "    Checkpoint: ${COLD_BOOT_DIR}/checkpoints/checkpoint_final.pt"
    echo ""
    read -p "Do you want to run cold boot again? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "[*] Skipping cold boot..."
    else
        echo "[*] Running cold boot..."
        if [ -n "$NUM_TRAIN_TASKS" ]; then
            python cold_boot_apibank.py \
              --data-path "${DATA_DIR}/train.json" \
              --config "${CONFIG_FILE}" \
              --output-dir "${COLD_BOOT_DIR}" \
              --num-tasks ${NUM_TRAIN_TASKS} \
              --evolution-interval ${EVOLUTION_INTERVAL} \
              --checkpoint-interval ${CHECKPOINT_INTERVAL}
        else
            python cold_boot_apibank.py \
              --data-path "${DATA_DIR}/train.json" \
              --config "${CONFIG_FILE}" \
              --output-dir "${COLD_BOOT_DIR}" \
              --evolution-interval ${EVOLUTION_INTERVAL} \
              --checkpoint-interval ${CHECKPOINT_INTERVAL}
        fi
        echo "[✓] Cold boot complete"
    fi
else
    echo "[*] Running cold boot..."
    echo "    This will take approximately 5-10 minutes for ${NUM_TRAIN_TASKS:-all} tasks..."
    echo ""
    
    if [ -n "$NUM_TRAIN_TASKS" ]; then
        python cold_boot_apibank.py \
          --data-path "${DATA_DIR}/train.json" \
          --config "${CONFIG_FILE}" \
          --output-dir "${COLD_BOOT_DIR}" \
          --num-tasks ${NUM_TRAIN_TASKS} \
          --evolution-interval ${EVOLUTION_INTERVAL} \
          --checkpoint-interval ${CHECKPOINT_INTERVAL}
    else
        python cold_boot_apibank.py \
          --data-path "${DATA_DIR}/train.json" \
          --config "${CONFIG_FILE}" \
          --output-dir "${COLD_BOOT_DIR}" \
          --evolution-interval ${EVOLUTION_INTERVAL} \
          --checkpoint-interval ${CHECKPOINT_INTERVAL}
    fi
    echo "[✓] Cold boot complete"
fi
echo ""

# Step 3: Evaluate on test set
echo "========================================================================"
echo "  Step 3: Evaluate on Test Set"
echo "========================================================================"
echo ""

EVAL_DIR="${OUTPUT_DIR}/evaluation"
CHECKPOINT="${COLD_BOOT_DIR}/checkpoints/checkpoint_final.pt"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "[✗] Checkpoint not found: ${CHECKPOINT}"
    echo "    Please run cold boot first."
    exit 1
fi

echo "[*] Running evaluation..."
echo "    This will take approximately 5 minutes for ${NUM_TEST_TASKS:-all} tasks..."
echo ""

if [ -n "$NUM_TEST_TASKS" ]; then
    python evaluate_apibank.py \
      --checkpoint "${CHECKPOINT}" \
      --data-path "${DATA_DIR}/test.json" \
      --config "${CONFIG_FILE}" \
      --output-dir "${EVAL_DIR}" \
      --num-tasks ${NUM_TEST_TASKS}
else
    python evaluate_apibank.py \
      --checkpoint "${CHECKPOINT}" \
      --data-path "${DATA_DIR}/test.json" \
      --config "${CONFIG_FILE}" \
      --output-dir "${EVAL_DIR}"
fi
echo "[✓] Evaluation complete"
echo ""

# Summary
echo "========================================================================"
echo "  Experiment Complete!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  Cold boot output: ${COLD_BOOT_DIR}"
echo "  Evaluation output: ${EVAL_DIR}"
echo ""
echo "Key files:"
echo "  - Cold boot stats: ${COLD_BOOT_DIR}/cold_boot_stats.json"
echo "  - Evaluation metrics: ${EVAL_DIR}/metrics.json"
echo "  - Detailed results: ${EVAL_DIR}/evaluation_results.json"
echo "  - Checkpoints: ${COLD_BOOT_DIR}/checkpoints/"
echo "  - Logs: ${COLD_BOOT_DIR}/logs/ and ${EVAL_DIR}/logs/"
echo ""

# Display metrics if available
if [ -f "${EVAL_DIR}/metrics.json" ]; then
    echo "Evaluation Metrics:"
    python -c "
import json
try:
    with open('${EVAL_DIR}/metrics.json', 'r') as f:
        metrics = json.load(f)
    print(f\"  Success Rate: {metrics.get('success_rate', 0)*100:.1f}%\")
    print(f\"  Average Reward: {metrics.get('avg_reward', 0):.3f}\")
    print(f\"  Average Steps: {metrics.get('avg_steps', 0):.1f}\")
    print(f\"  API Accuracy: {metrics.get('avg_api_accuracy', 0)*100:.1f}%\")
    print(f\"  Completion Rate: {metrics.get('completion_rate', 0)*100:.1f}%\")
except Exception as e:
    print(f\"  Error reading metrics: {e}\")
"
    echo ""
fi

echo "[✓] All experiments complete!"
echo ""
echo "To view detailed logs:"
echo "  tail -f ${COLD_BOOT_DIR}/logs/*.log"
echo "  tail -f ${EVAL_DIR}/logs/*.log"
echo ""
