"""
Extract Block-Level Representations from HPSN
==============================================
Extracts representations from each block boundary for brain encoding
evaluation (RSA, linear encoding models).

Usage:
    python extract_features.py \
        --checkpoint runs/hpsn_v1/stage2_best.pt \
        --audio_dir /path/to/audio/files \
        --output_dir features/hpsn_v1/
"""

import argparse
import glob
import os

import torch
import torchaudio
import numpy as np
from pathlib import Path

from model import HPSNHubert
from data import get_processor


def extract_block_features(
    model, 
    processor, 
    audio_path: str, 
    device: torch.device,
    layer_norm_features: bool = True,
):
    """
    Extract block-level representations for a single audio file.
    
    Returns:
        dict with keys:
            - block_0 through block_5: numpy arrays of shape (T_enc, 1024)
            - attention_weights: list of (num_sources,) arrays
            - inhibition_effective_rank: float
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.squeeze(0)  # mono
    
    # Process
    inputs = processor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt",
    )
    input_values = inputs.input_values.to(device)
    
    # Forward with block outputs
    with torch.no_grad():
        outputs = model(
            input_values=input_values, 
            return_block_outputs=True,
        )
    
    result = {}
    for i, bo in enumerate(outputs["block_outputs"]):
        result[f"block_{i}"] = bo.squeeze(0).cpu().numpy()  # (T, 1024)
    
    result["logits"] = outputs["logits"].squeeze(0).cpu().numpy()
    result["attention_weights"] = [
        w.squeeze(0).cpu().numpy() if w is not None else None
        for w in outputs["attention_weights"]
    ]
    result["inhibition_effective_rank"] = outputs["inhibition_effective_rank"]
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pretrained", type=str, default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory of audio files, or a single file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--file_ext", type=str, default="wav")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = HPSNHubert(pretrained=args.pretrained).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    
    processor = get_processor(args.pretrained)
    
    # Find audio files
    if os.path.isfile(args.audio_dir):
        audio_files = [args.audio_dir]
    else:
        audio_files = sorted(glob.glob(
            os.path.join(args.audio_dir, f"**/*.{args.file_ext}"), recursive=True,
        ))
    
    print(f"Found {len(audio_files)} audio files")
    os.makedirs(args.output_dir, exist_ok=True)
    
    for audio_path in audio_files:
        name = Path(audio_path).stem
        print(f"  Processing {name}...")
        
        features = extract_block_features(model, processor, audio_path, device)
        
        # Save as npz
        out_path = os.path.join(args.output_dir, f"{name}.npz")
        np.savez_compressed(
            out_path,
            **{k: v for k, v in features.items() if isinstance(v, np.ndarray)},
            attention_weights=np.array(
                [w for w in features["attention_weights"] if w is not None], 
                dtype=object,
            ),
            inhibition_rank=np.array(features["inhibition_effective_rank"]),
        )
    
    print(f"Done. Features saved to {args.output_dir}")


if __name__ == "__main__":
    main()
