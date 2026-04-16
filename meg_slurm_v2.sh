#!/bin/bash
#SBATCH --partition=general-gpu
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH -C gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --exclude=gpu40,gpu18,gpu20,gpu21,gpu22,gpu48,gpu50
#SBATCH --output=meg_eval_v2.out

mkdir -p results/meg_v2_hpsn results/meg_v2_baseline data

module load cuda/12.3
source ~/.bashrc
source ~/miniconda3/bin/activate es

echo "=== MEG Neural Encoding v2 ==="
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

MEG_DIR="/scratch/jsm04005/fie24002/DATA/gwilliams2022/download/osfstorage"
CHECKPOINT="runs/v6_full/stage2_step60000.pt"
FASTLEX_DIR="fastlex"
DENSITY_CSV="data/meg_word_density.csv"

# -------------------------------------------------------
# 0. Clone fastlex & build density CSV (skip if exists)
# -------------------------------------------------------
if [ ! -d "$FASTLEX_DIR" ]; then
    echo ""
    echo "=== Cloning FastLex ==="
    git clone https://github.com/comp-cogneuro-lang/fastlex.git "$FASTLEX_DIR"
    pip install pandas tqdm rapidfuzz nltk
fi

if [ ! -f "$DENSITY_CSV" ]; then
    echo ""
    echo "=== Building density CSV ==="
    python build_density.py \
        --meg_dir "$MEG_DIR" \
        --fastlex_dir "$FASTLEX_DIR" \
        --output "$DENSITY_CSV" \
        --n_jobs 16
fi

# -------------------------------------------------------
# 1. HPSN full model
# -------------------------------------------------------
echo ""
echo "=== Run 1: HPSN full model (v2 evaluation) ==="
python meg_encoding_v2.py \
    --checkpoint "$CHECKPOINT" \
    --meg_dir "$MEG_DIR" \
    --output_dir results/meg_v2_hpsn \
    --density_csv "$DENSITY_CSV" \
    --log_level INFO \
    --tasks 0

# -------------------------------------------------------
# 2. Baseline (bypass novel modules)
# -------------------------------------------------------
echo ""
echo "=== Run 2: Pretrained HuBERT baseline ==="
python meg_encoding_v2.py \
    --checkpoint "$CHECKPOINT" \
    --meg_dir "$MEG_DIR" \
    --output_dir results/meg_v2_baseline \
    --bypass_novel \
    --log_level INFO \
    --skip_correction \
    --skip_inhibition \
    --skip_cpc_n400 \
    --tasks 0

echo ""
echo "Done: $(date)"
