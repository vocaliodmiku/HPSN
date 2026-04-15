#!/bin/bash
#SBATCH --partition=general-gpu                  # Name of Partition
#SBATCH --ntasks=20                        # Maximum CPU cores for job
#SBATCH --nodes=1                             # Ensure all cores are from the same node
#SBATCH -C gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
##SBATCH --exclude=gpu40,gpu18,gpu20,gpu21,gpu22,gpu48,gpu50
#SBATCH --output=v2_twopass_meg.out


mkdir -p logs results/meg_encoding

module load cuda/12.3
source ~/.bashrc
source ~/miniconda3/bin/activate es

echo "=== MEG Neural Encoding Evaluation ==="
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

MEG_DIR="/scratch/jsm04005/fie24002/DATA/gwilliams2022/download/osfstorage"
CHECKPOINT="runs/v2_twopass/stage2_best.pt"
export CUDA_VISIBLE_DEVICES=1
# -------------------------------------------------------
# Quick test: 1 subject, 1 session (uncomment for testing)
# -------------------------------------------------------
# python meg_encoding.py \
#     --checkpoint "$CHECKPOINT" \
#     --meg_dir "$MEG_DIR" \
#     --output_dir results/meg_encoding_test \
#     --subjects 1 \
#     --sessions 0 \
#     --max_words 100

# -------------------------------------------------------
# Full run: all subjects, both sessions
# -------------------------------------------------------
python meg_encoding.py \
    --checkpoint "$CHECKPOINT" \
    --meg_dir "$MEG_DIR" \
    --output_dir results/meg_encoding \
    --max_words 300 \
    --subjects 1 \
    --log_level INFO \
    --skip_encoding

echo "Done: $(date)"
