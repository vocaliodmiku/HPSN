"""HPSN Training Script (v2)
===========================
Two-stage CTC fine-tuning of HuBERT-Large + RefineLayer + Inhibition.

Stage 1: Backbone frozen, train only novel modules (refine layers,
         inhibition, CTC head). ~10K steps with higher LR.

Stage 2: Everything unfrozen (except CNN), joint fine-tuning with
         differential LR. ~80K steps.

Usage:
    # Single GPU
    python train.py --output_dir runs/hpsn_v1

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 train.py --output_dir runs/hpsn_v1
    
    # Quick test on small data
    python train.py --output_dir runs/test --train_split train.clean.100 \
                    --stage1_steps 100 --stage2_steps 200 --batch_size 2
"""

import os
import argparse
import json
import time
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchaudio

from model import HPSNHubert
from data import get_processor, load_librispeech, prepare_dataset, DataCollatorCTCWithPadding, LazyLibriSpeechDataset

from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="HPSN Training")
    
    # Model
    parser.add_argument("--pretrained", type=str, default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--inhibition_rank", type=int, default=64)
    parser.add_argument("--inhibition_boundary", type=int, default=2, 
                        help="Block boundary for inhibition (0=B1/B2, 2=B3/B4)")
    parser.add_argument("--refine_num_heads", type=int, default=4,
                        help="Number of attention heads in RefineLayer")
    
    # Data
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to local LibriSpeech root directory")
    parser.add_argument("--train_split", type=str, default="train.clean.960")
    parser.add_argument("--max_audio_sec", type=float, default=20.0,
                        help="Max audio length in seconds")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Training — Stage 1 (frozen backbone)
    parser.add_argument("--stage1_steps", type=int, default=10000)
    parser.add_argument("--stage1_lr", type=float, default=3e-4)
    parser.add_argument("--stage1_warmup", type=int, default=1000)
    parser.add_argument("--stage1_batch_size", type=int, default=8)
    
    # Training — Stage 2 (joint fine-tuning)
    parser.add_argument("--stage2_steps", type=int, default=80000)
    parser.add_argument("--stage2_lr_backbone", type=float, default=3e-5)
    parser.add_argument("--stage2_lr_novel", type=float, default=1e-4)
    parser.add_argument("--stage2_warmup", type=int, default=2000)
    parser.add_argument("--stage2_batch_size", type=int, default=4)
    
    # Shared
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override both stage batch sizes")
    parser.add_argument("--accum_grad", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--rank_reg_weight", type=float, default=0.01)
    parser.add_argument("--denoise_std", type=float, default=0.0,
                        help="Gaussian noise std for lower-level denoising (0=off)")
    parser.add_argument("--denoise_weight", type=float, default=1.0,
                        help="Weight for denoising loss (active when denoise_std>0)")
    parser.add_argument("--sharpen_weight", type=float, default=0.0,
                        help="Weight for contrastive sharpening loss (0=off)")
    parser.add_argument("--sharpen_temp", type=float, default=0.1,
                        help="Temperature for sharpening contrastive loss")

    # Logging / checkpointing
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=200,
                        help="Number of validation samples for WER eval")
    parser.add_argument(
        "--dry_run_eval_samples", type=int, default=0,
        help="Dry-run WER samples before training (<=0 means full dev sets)",
    )
    
    # Hardware
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.batch_size is not None:
        args.stage1_batch_size = args.batch_size
        args.stage2_batch_size = args.batch_size
    
    return args


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main_process(rank):
    return rank == 0


def log(msg, rank=0):
    if is_main_process(rank):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cosine_schedule(step, total_steps, warmup_steps, lr_start, lr_end=1e-6):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return lr_start * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * progress))


def _build_id2token(processor):
    """Build index → token mapping from the HF tokenizer vocabulary."""
    vocab = processor.tokenizer.get_vocab()
    return {v: k for k, v in vocab.items()}


def _viterbi_decode(pred_ids, id2token, blank=0):
    """Manual CTC greedy decode: collapse repeats, drop blanks, join chars."""
    hyp = pred_ids.unique_consecutive()
    hyp = hyp[hyp != blank]
    return "".join(id2token[int(i)] for i in hyp).lower().replace("|", " ").strip()


@torch.no_grad()
def evaluate_wer(model, processor, val_loader, device, max_samples=200, bypass_novel=False, debug=False):
    """Compute WER on validation set using manual viterbi CTC decode."""
    was_training = model.training
    model.eval()
    id2token = _build_id2token(processor)
    total_edit_distance = 0
    total_length = 0
    count = 0
    
    for batch in val_loader:
        if max_samples is not None and max_samples > 0 and count >= max_samples:
            break
        
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if bypass_novel:
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                bypass_novel=True,
            )
        else:
            outputs = model(input_values=input_values, attention_mask=attention_mask)

        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs.logits
        
        pred_ids = torch.argmax(logits, dim=-1)
        
        # Get reference texts.
        if "transcripts" in batch:
            label_strs = [s.strip().lower() for s in batch["transcripts"]]
            labels = batch["labels"]
        else:
            labels = batch["labels"]
            label_strs = []
            for label_seq in labels:
                valid = label_seq[label_seq >= 0]
                text = "".join(id2token[int(i)] for i in valid).lower().replace("|", " ").strip()
                label_strs.append(text)

        # Manual viterbi CTC decode for predictions.
        pred_strs = [_viterbi_decode(pred_ids[i].cpu(), id2token) for i in range(pred_ids.shape[0])]
        
        if debug and count == 0:
            print(f"[DEBUG] logits shape: {logits.shape}, range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            print(f"[DEBUG] pred_ids[0][:20]: {pred_ids[0][:20].cpu().tolist()}")
            print(f"[DEBUG] labels[0][:20]:   {labels[0][:20].tolist()}")
            print(f"[DEBUG] pred_str[0]:  '{pred_strs[0][:100]}'")
            print(f"[DEBUG] ref_str[0]:   '{label_strs[0][:100]}'")
        
        for pred, ref in zip(pred_strs, label_strs):
            hw = pred.split()
            rw = ref.split()
            total_edit_distance += torchaudio.functional.edit_distance(hw, rw)
            total_length += len(rw)
        count += len(pred_strs)
    
    if total_length == 0:
        model.train(was_training)
        return float("inf")
    
    word_error_rate = total_edit_distance / total_length
    model.train(was_training)
    return word_error_rate


def monitor_model(model, step, rank=0):
    """Log model diagnostics."""
    if not is_main_process(rank):
        return {}
    
    m = model.module if hasattr(model, "module") else model
    
    diagnostics = {
        "inhibition/effective_rank": m.inhibition.effective_rank,
        "inhibition/tau": m.inhibition.tau.item(),
        "inhibition/lambda": m.inhibition.lambda_.item(),
    }
    
    # RefineLayer diagnostics: per-level gate value and attention entropy
    for i, rl in enumerate(m.refine_layers):
        diagnostics[f"refine/level_{i}_gate"] = rl.gate.item()
        w = rl._last_attention_weights
        if w is not None:
            entropy = -(w * (w + 1e-8).log()).sum(-1).mean().item()
            diagnostics[f"refine/level_{i}_entropy"] = entropy
    
    return diagnostics


def train_stage(
    model, optimizer, train_loader, val_clean_loader, val_other_loader,
    processor, device, args, stage_name, total_steps, warmup_steps,
    lr_configs, rank, scaler=None,
):
    """
    Generic training loop for one stage.
    
    lr_configs: list of dicts with 'lr_start' and 'lr_end' per param group
    """
    model.train()
    step = 0
    epoch = 0
    accum_loss = 0.0
    accum_denoise = 0.0
    accum_sharpen = 0.0
    best_wer = float("inf")
    
    log(f"=== {stage_name}: {total_steps} steps, warmup={warmup_steps} ===", rank)
    
    while step < total_steps:
        epoch += 1
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        for batch in train_loader:
            if step >= total_steps:
                break
            
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward
            with torch.amp.autocast('cuda', enabled=args.fp16):
                outputs = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    labels=labels,
                    rank_reg_weight=args.rank_reg_weight,
                    denoise_std=args.denoise_std,
                    denoise_weight=args.denoise_weight,
                    sharpen_weight=args.sharpen_weight,
                    sharpen_temp=args.sharpen_temp,
                )
                loss = outputs["loss"] / args.accum_grad
            
            # NaN detection: skip backward + optimizer step if loss is NaN
            if torch.isnan(loss):
                if step < 10 or step % 100 == 0:
                    log(f"WARNING: NaN loss at step {step}, skipping update", rank)
                optimizer.zero_grad()
                step += 1
                continue
            
            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accum_loss += outputs["loss"].item()  # track raw loss, not pre-divided
            accum_denoise += outputs.get("denoise_loss", torch.zeros(1)).item()
            accum_sharpen += outputs.get("sharpen_loss", torch.zeros(1)).item()
            
            # Optimizer step every accum_grad mini-batches
            if (step + 1) % args.accum_grad == 0 or step == total_steps - 1:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm,
                )
                
                # Skip optimizer step if gradients are NaN
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    if step < 50 or step % 100 == 0:
                        log(f"WARNING: NaN/Inf grad norm ({grad_norm:.4f}) at step {step}, skipping update", rank)
                    optimizer.zero_grad()
                    step += 1
                    continue
                
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update LR (cosine schedule)
                for pg_idx, pg in enumerate(optimizer.param_groups):
                    cfg = lr_configs[min(pg_idx, len(lr_configs) - 1)]
                    pg["lr"] = cosine_schedule(
                        step, total_steps, warmup_steps,
                        cfg["lr_start"], cfg.get("lr_end", 1e-6),
                    )
            
            step += 1
            
            # Logging
            if step % args.log_every == 0 and is_main_process(rank):
                avg_loss = accum_loss / args.log_every
                current_lr = optimizer.param_groups[0]["lr"]
                diag = monitor_model(model, step, rank)

                if args.denoise_std > 0:
                    diag["denoise_loss"] = accum_denoise / args.log_every
                if args.sharpen_weight > 0:
                    diag["sharpen_loss"] = accum_sharpen / args.log_every

                diag_str = " | ".join(f"{k}: {v:.4f}" for k, v in diag.items())
                log(
                    f"{stage_name} step {step}/{total_steps} | "
                    f"loss: {avg_loss:.4f} | lr: {current_lr:.2e} | {diag_str}",
                    rank,
                )
                accum_loss = 0.0
                accum_denoise = 0.0
                accum_sharpen = 0.0
            
            # Evaluation
            if step % args.eval_every == 0:
                m = model.module if hasattr(model, "module") else model
                wer_clean = evaluate_wer(
                    m, processor, val_clean_loader, device, args.eval_samples,
                )
                wer_other = evaluate_wer(
                    m, processor, val_other_loader, device, args.eval_samples,
                )
                
                log(
                    f"{stage_name} step {step} | "
                    f"WER clean: {wer_clean:.4f} | WER other: {wer_other:.4f}",
                    rank,
                )
                
                # Save best
                if wer_other < best_wer and is_main_process(rank):
                    best_wer = wer_other
                    save_path = Path(args.output_dir) / f"{stage_name}_best.pt"
                    m_to_save = model.module if hasattr(model, "module") else model
                    torch.save(m_to_save.state_dict(), save_path)
                    log(f"Saved best model (WER other={best_wer:.4f}) → {save_path}", rank)
                
                model.train()
            
            # Periodic checkpoint
            if step % args.save_every == 0 and is_main_process(rank):
                save_path = Path(args.output_dir) / f"{stage_name}_step{step}.pt"
                m_to_save = model.module if hasattr(model, "module") else model
                torch.save({
                    "model_state_dict": m_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                }, save_path)
    
    return best_wer


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        with open(Path(args.output_dir) / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    
    # === Load processor ===
    log("Loading processor...", rank)
    processor = get_processor(args.pretrained)
    
    # === Build model ===
    log("Building HPSN model...", rank)
    model = HPSNHubert(
        pretrained=args.pretrained,
        refine_num_heads=args.refine_num_heads,
        inhibition_rank=args.inhibition_rank,
        inhibition_boundary=args.inhibition_boundary,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    novel_params = sum(
        p.numel() for n, p in model.named_parameters()
        if any(x in n for x in ["refine_layers", "inhibition", "ctc_head"])
    )
    log(f"Total params: {total_params:,} | Novel params: {novel_params:,} "
        f"({novel_params/total_params*100:.2f}%)", rank)
    
    # === Load data ===
    log(f"Loading data (train={args.train_split})...", rank)
    
    train_ds = LazyLibriSpeechDataset(
        load_librispeech(args.train_split, data_dir=args.data_dir), processor
    )
    val_clean = LazyLibriSpeechDataset(
        load_librispeech("validation.clean", data_dir=args.data_dir), processor
    )
    val_other = LazyLibriSpeechDataset(
        load_librispeech("validation.other", data_dir=args.data_dir), processor
    )
    
    max_input_length = int(args.max_audio_sec * 16000)
    collator = DataCollatorCTCWithPadding(
        processor=processor, max_input_length=max_input_length,
    )
    
    val_clean_loader = DataLoader(
        val_clean, batch_size=args.stage1_batch_size * 2,
        shuffle=False, collate_fn=collator, num_workers=args.num_workers,
    )
    val_other_loader = DataLoader(
        val_other, batch_size=args.stage1_batch_size * 2,
        shuffle=False, collate_fn=collator, num_workers=args.num_workers,
    )
    
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # =========================================================
    # STAGE 1: Frozen backbone, train novel modules only
    # =========================================================
    log("Starting Stage 1: training novel modules (backbone frozen)...", rank)
    model.freeze_backbone()
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Stage 1 trainable params: {trainable:,}", rank)
    
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        if world_size > 1 else None
    )
    train_loader_s1 = DataLoader(
        train_ds, batch_size=args.stage1_batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    
    optimizer_s1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.stage1_lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98), eps=1e-8,
    )
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    train_stage(
        model, optimizer_s1, train_loader_s1, val_clean_loader, val_other_loader,
        processor, device, args,
        stage_name="stage1",
        total_steps=args.stage1_steps,
        warmup_steps=args.stage1_warmup,
        lr_configs=[{"lr_start": args.stage1_lr, "lr_end": 1e-5}],
        rank=rank,
        scaler=scaler,
    )
    
    # =========================================================
    # STAGE 2: Joint fine-tuning with differential LR
    # =========================================================
    log("Starting Stage 2: joint fine-tuning (backbone unfrozen)...", rank)
    
    m = model.module if hasattr(model, "module") else model
    m.unfreeze_backbone()
    
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    log(f"Stage 2 trainable params: {trainable:,}", rank)
    
    param_groups = m.get_param_groups(
        lr_backbone=args.stage2_lr_backbone,
        lr_novel=args.stage2_lr_novel,
    )
    optimizer_s2 = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay,
        betas=(0.9, 0.98), eps=1e-8,
    )
    
    train_loader_s2 = DataLoader(
        train_ds, batch_size=args.stage2_batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    
    train_stage(
        model, optimizer_s2, train_loader_s2, val_clean_loader, val_other_loader,
        processor, device, args,
        stage_name="stage2",
        total_steps=args.stage2_steps,
        warmup_steps=args.stage2_warmup,
        lr_configs=[
            {"lr_start": args.stage2_lr_backbone, "lr_end": 1e-6},
            {"lr_start": args.stage2_lr_novel, "lr_end": 1e-6},
        ],
        rank=rank,
        scaler=scaler,
    )
    
    # === Final evaluation ===
    log("Running final evaluation...", rank)
    m = model.module if hasattr(model, "module") else model
    
    # Load best checkpoint
    best_path = Path(args.output_dir) / "stage2_best.pt"
    if best_path.exists():
        m.load_state_dict(torch.load(best_path, map_location=device))
        log(f"Loaded best model from {best_path}", rank)
    
    wer_clean = evaluate_wer(m, processor, val_clean_loader, device, max_samples=9999)
    wer_other = evaluate_wer(m, processor, val_other_loader, device, max_samples=9999)
    
    log(f"FINAL | WER dev-clean: {wer_clean:.4f} | WER dev-other: {wer_other:.4f}", rank)
    
    if is_main_process(rank):
        results = {
            "wer_dev_clean": wer_clean,
            "wer_dev_other": wer_other,
        }
        with open(Path(args.output_dir) / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
