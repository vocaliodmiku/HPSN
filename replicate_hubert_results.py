"""
Replicate HuBERT-Large LibriSpeech Results
==========================================

Evaluates the pretrained HuBERT CTC checkpoint on LibriSpeech splits and
reports WER.  Uses torchaudio for data loading and decoding, following:
https://github.com/pytorch/audio/blob/main/examples/hubert/evaluate.py

Published WER (with 4-gram LM + beam search):
  test-clean: 1.9%, test-other: 3.3%, dev-clean: 1.5%, dev-other: 3.0%

Expected greedy WER (argmax, no LM):
  test-clean: ~2.5%, test-other: ~5.0%

Usage:
  # Greedy decoding
  python replicate_hubert_results.py \\
      --data_dir /path/to/LibriSpeech

  # With 4-gram LM (replicates published results)
  python replicate_hubert_results.py \\
      --data_dir /path/to/LibriSpeech --use_lm

  # Quick smoke test
  python replicate_hubert_results.py \\
      --data_dir /path/to/LibriSpeech --max_samples 200
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchaudio
from transformers import HubertForCTC, Wav2Vec2Processor

logger = logging.getLogger(__name__)

TARGET_WER = {
    "dev-clean": 1.5,
    "dev-other": 3.0,
    "test-clean": 1.9,
    "test-other": 3.3,
}

TARGET_WER_GREEDY = {
    "dev-clean": 2.1,
    "dev-other": 4.8,
    "test-clean": 2.5,
    "test-other": 5.0,
}


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def _get_id2token(processor: Wav2Vec2Processor) -> Dict[int, str]:
    """Build index → token mapping from the HF tokenizer vocabulary."""
    vocab = processor.tokenizer.get_vocab()
    return {v: k for k, v in vocab.items()}


# ---------------------------------------------------------------------------
# Decoding  (matches torchaudio evaluate.py)
# ---------------------------------------------------------------------------

def _viterbi_decode(
    emission: torch.Tensor, id2token: Dict[int, str], blank_idx: int = 0,
) -> List[str]:
    """Greedy CTC decode — argmax, collapse repeats, drop blanks."""
    hypothesis = emission.argmax(-1).unique_consecutive()
    hypothesis = hypothesis[hypothesis != blank_idx]
    text = (
        "".join(id2token[int(i)] for i in hypothesis)
        .lower()
        .replace("|", " ")
        .strip()
    )
    return text.split()


def _ctc_decode(emission: torch.Tensor, decoder) -> List[str]:
    """LM beam-search decode using torchaudio ctc_decoder."""
    hypothesis = decoder(emission)
    hypothesis = hypothesis[0][0].words
    return [w.lower() for w in hypothesis if w != " "]


# ---------------------------------------------------------------------------
# Data-path resolution for torchaudio.datasets.LIBRISPEECH
# ---------------------------------------------------------------------------

def _resolve_data_path(data_dir: str):
    """
    Return (root, folder_in_archive) for torchaudio.datasets.LIBRISPEECH.

    Handles both layouts:
      data_dir = .../LibriSpeech              (split dirs directly inside)
      data_dir = .../LibriSpeech/LibriSpeech   (nested)
    """
    p = Path(data_dir).resolve()
    if (p / "test-clean").is_dir():
        return str(p.parent), p.name
    if (p / "LibriSpeech" / "test-clean").is_dir():
        return str(p), "LibriSpeech"
    raise FileNotFoundError(
        f"Cannot find LibriSpeech split folders under {data_dir}.\n"
        f"Checked: {p}/test-clean and {p}/LibriSpeech/test-clean"
    )


# ---------------------------------------------------------------------------
# LM decoder setup  (torchaudio ctc_decoder)
# ---------------------------------------------------------------------------

def _build_tokens_file(processor: Wav2Vec2Processor) -> str:
    """Write the HF vocabulary as a torchaudio-compatible tokens file."""
    vocab = processor.tokenizer.get_vocab()
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
    fd, path = tempfile.mkstemp(suffix=".tokens", prefix="hubert_")
    with os.fdopen(fd, "w") as f:
        for token, _ in sorted_tokens:
            f.write(f"{token}\n")
    return path


def build_lm_decoder(processor, args):
    """Build a torchaudio CTC decoder with the LibriSpeech 4-gram LM."""
    from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

    files = download_pretrained_files("librispeech-4-gram")
    tokens_file = _build_tokens_file(processor)

    decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=tokens_file,
        lm=files.lm,
        nbest=1,
        beam_size=args.beam_size,
        beam_size_token=args.beam_size_token,
        beam_threshold=args.beam_threshold,
        lm_weight=args.lm_weight,
        word_score=args.word_score,
        unk_score=float("-inf"),
        sil_score=0,
        blank_token="<pad>",
        sil_token="|",
        unk_word="<unk>",
    )
    return decoder


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_split(
    model,
    id2token,
    dataset,
    device,
    use_fp16=False,
    max_samples=0,
    lm_decoder=None,
):
    """Evaluate one LibriSpeech split.  Returns (WER%, sample_count)."""
    model.eval()
    amp_enabled = use_fp16 and device.type == "cuda"

    total_edit_distance = 0
    total_length = 0
    n = 0

    for idx, (waveform, sample_rate, transcript, _, _, _) in enumerate(dataset):
        if 0 < max_samples <= idx:
            break

        assert sample_rate == 16_000, f"Expected 16 kHz, got {sample_rate}"
        ref_words = transcript.strip().lower().split()

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(input_values=waveform.to(device)).logits

        emission = F.log_softmax(logits, dim=-1)

        if lm_decoder is not None:
            hyp_words = _ctc_decode(emission.cpu(), lm_decoder)
        else:
            hyp_words = _viterbi_decode(emission[0], id2token)

        total_edit_distance += torchaudio.functional.edit_distance(
            hyp_words, ref_words,
        )
        total_length += len(ref_words)
        n += 1

        if idx < 3:
            logger.info(f"  REF:  {' '.join(ref_words)}")
            logger.info(f"  HYP:  {' '.join(hyp_words)}")

        if (idx + 1) % 500 == 0:
            running = total_edit_distance / total_length * 100
            logger.info(f"  ... {idx + 1} samples, running WER: {running:.2f}%")

    wer = (
        total_edit_distance / total_length * 100.0
        if total_length > 0
        else float("inf")
    )
    return wer, n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Replicate HuBERT LibriSpeech WER (torchaudio pipeline)",
    )
    parser.add_argument("--pretrained", default="facebook/hubert-large-ls960-ft")
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to local LibriSpeech root (parent of test-clean/ etc.).",
    )
    parser.add_argument(
        "--split", nargs="+",
        default=["test-clean", "test-other", "dev-clean", "dev-other"],
        choices=["test-clean", "test-other", "dev-clean", "dev-other"],
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_json", default="hubert_replicate_results.json")
    # ---- LM decoding (torchaudio ctc_decoder) ----
    parser.add_argument("--use_lm", action="store_true")
    parser.add_argument("--beam_size", type=int, default=1500)
    parser.add_argument("--beam_size_token", type=int, default=29)
    parser.add_argument("--beam_threshold", type=int, default=100)
    parser.add_argument("--lm_weight", type=float, default=2.46)
    parser.add_argument("--word_score", type=float, default=-0.59)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    fmt = "%(asctime)s %(message)s" if args.debug else "%(message)s"
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device:   {device}")
    logger.info(f"Model:    {args.pretrained}")
    logger.info(f"Decoding: {'LM beam search' if args.use_lm else 'Greedy (viterbi)'}")

    # ---- Model & vocab ----
    model = HubertForCTC.from_pretrained(args.pretrained).to(device).eval()
    processor = Wav2Vec2Processor.from_pretrained(args.pretrained)
    id2token = _get_id2token(processor)

    # ---- LM decoder (optional) ----
    lm_decoder = None
    if args.use_lm:
        lm_decoder = build_lm_decoder(processor, args)

    # ---- Data ----
    root, folder = _resolve_data_path(args.data_dir)
    logger.info(f"Data:     {root}/{folder}\n")

    # ---- Evaluate each split ----
    measured = {}
    for split in args.split:
        logger.info(f"Evaluating {split} ...")
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=split, folder_in_archive=folder, download=False,
        )

        wer, n = evaluate_split(
            model=model,
            id2token=id2token,
            dataset=dataset,
            device=device,
            use_fp16=args.fp16,
            max_samples=args.max_samples,
            lm_decoder=lm_decoder,
        )
        measured[split] = wer
        logger.info(f">>> {split}: WER={wer:.2f}% ({n} samples)\n")

    # ---- Summary ----
    target = TARGET_WER if args.use_lm else TARGET_WER_GREEDY
    decoding_label = "LM beam search" if args.use_lm else "Greedy"

    print(f"\n{'=' * 60}")
    print(f"Target vs Measured  ({decoding_label})")
    print(f"{'=' * 60}")
    for split in args.split:
        t = target[split]
        m = measured[split]
        delta = m - t
        status = "OK" if abs(delta) < 0.5 else ("HIGH" if delta > 0 else "LOW")
        print(
            f"  {split:12s}  target={t:5.1f}%  measured={m:5.2f}%"
            f"  delta={delta:+5.2f}%  [{status}]"
        )

    output = {
        "pretrained": args.pretrained,
        "data_dir": args.data_dir,
        "max_samples": args.max_samples,
        "decoding": "lm_beam_search" if args.use_lm else "greedy",
        "target_wer_percent": {s: target[s] for s in args.split},
        "measured_wer_percent": measured,
    }
    if args.use_lm:
        output.update({
            "beam_size": args.beam_size,
            "lm_weight": args.lm_weight,
            "word_score": args.word_score,
        })

    out = Path(args.output_json)
    out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
    
    
#     Target vs Measured  (Greedy)
# ============================================================
#   test-clean    target=  2.5%  measured= 2.16%  delta=-0.34%  [OK]
#   test-other    target=  5.0%  measured= 4.58%  delta=-0.42%  [OK]
#   dev-clean     target=  2.1%  measured= 2.12%  delta=+0.02%  [OK]
#   dev-other     target=  4.8%  measured= 4.40%  delta=-0.40%  [OK]