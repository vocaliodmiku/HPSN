
**Standard vertical attention (baseline — current behavior):**
```bash
python train.py --output_dir runs/standard \
    --vertical_attention_type standard \
    --train_split train.clean.100 \
    --stage1_steps 10000 --stage2_steps 80000
```

**Chunked recurrent (scalar attention, shared keys):**
```bash
python train.py --output_dir runs/chunked_scalar \
    --vertical_attention_type chunked \
    --chunk_size 16 \
    --share_topdown_keys \
    --train_split train.clean.100 \
    --stage1_steps 10000 --stage2_steps 80000
```

**Chunked recurrent (per-token attention):**
```bash
python train.py --output_dir runs/chunked_pertoken \
    --vertical_attention_type chunked_per_token \
    --chunk_size 16 \
    --share_topdown_keys \
    --train_split train.clean.100 \
    --stage1_steps 10000 --stage2_steps 80000
```

**Chunked with separate top-down keys:**
```bash
python train.py --output_dir runs/chunked_sepkeys \
    --vertical_attention_type chunked \
    --chunk_size 16 \
    --no_share_topdown_keys \
    --train_split train.clean.100 \
    --stage1_steps 10000 --stage2_steps 80000
```

**Varying chunk size (~160ms / ~640ms):**
```bash
# Smaller chunks (tighter feedback loop)
python train.py --output_dir runs/chunked_cs8 \
    --vertical_attention_type chunked --chunk_size 8 ...

# Larger chunks (coarser feedback)
python train.py --output_dir runs/chunked_cs32 \
    --vertical_attention_type chunked --chunk_size 32 ...
```

**Key flags:**

| Flag | Values | Default | Effect |
|---|---|---|---|
| `--vertical_attention_type` | `standard`, `chunked`, `chunked_per_token` | `standard` | Selects attention module |
| `--chunk_size` | int (frames) | `16` (~320ms) | Temporal resolution of top-down feedback |
| `--share_topdown_keys` / `--no_share_topdown_keys` | bool | shared | Whether BU and TD use same `W_K` |
| `--inhibition_boundary` | 0–3 | 2 (B3/B4) | Where to place inhibition |

All other flags (`--batch_size`, `--fp16`, `--accum_grad`, `--pretrained`, etc.) work the same across all configs. For a quick smoke test, add `--stage1_steps 100 --stage2_steps 0 --batch_size 2 --train_split train.clean.100`.