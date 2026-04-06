"""
Data Loading for HuBERT Evaluation
====================================
LibriSpeech loading via HuggingFace datasets with CTC-compatible processing.
"""

import io
import os

import numpy as np
import soundfile as sf
import torch
from dataclasses import dataclass
from datasets import Audio, Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor


def get_processor(pretrained: str = "facebook/hubert-large-ls960-ft"):
    """Load the processor (feature extractor + tokenizer) from the fine-tuned model."""
    return Wav2Vec2Processor.from_pretrained(pretrained)


# ---------------------------------------------------------------------------
# Local loading
# ---------------------------------------------------------------------------

def _load_librispeech_local(data_dir, split_dirs):
    """Load LibriSpeech from local directory into a HuggingFace Dataset."""
    records = []
    for split_dir in split_dirs:
        split_path = os.path.join(data_dir, split_dir)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"Split directory not found: {split_path}")
        for root, dirs, files in os.walk(split_path):
            for f in files:
                if f.endswith(".trans.txt"):
                    trans_path = os.path.join(root, f)
                    with open(trans_path) as tf:
                        for line in tf:
                            line = line.strip()
                            if not line:
                                continue
                            utt_id, text = line.split(" ", 1)
                            flac_path = os.path.join(root, utt_id + ".flac")
                            if os.path.exists(flac_path):
                                records.append({"audio_path": flac_path, "text": text})

    ds = Dataset.from_dict({
        "audio_path": [r["audio_path"] for r in records],
        "text": [r["text"] for r in records],
    })
    return ds


_SPLIT_TO_DIRS = {
    "train.clean.100": ["train-clean-100"],
    "train.clean.360": ["train-clean-360"],
    "train.other.500": ["train-other-500"],
    "train.clean.960": ["train-clean-100", "train-clean-360", "train-other-500"],
    "validation.clean": ["dev-clean"],
    "validation.other": ["dev-other"],
    "test.clean": ["test-clean"],
    "test.other": ["test-other"],
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_librispeech(
    split: str = "train.clean.960",
    streaming: bool = False,
    data_dir: str = None,
):
    """
    Load LibriSpeech from HuggingFace Hub or local directory.

    Args:
        split: e.g. "test.clean", "validation.other", "train.clean.960"
        streaming: Whether to use streaming mode (HF Hub only).
        data_dir: Path to local LibriSpeech root. If None, downloads from HF Hub.
    """
    if data_dir is not None:
        if split not in _SPLIT_TO_DIRS:
            raise ValueError(
                f"Unknown split '{split}'. Available: {list(_SPLIT_TO_DIRS.keys())}"
            )
        return _load_librispeech_local(data_dir, _SPLIT_TO_DIRS[split])

    # HuggingFace Hub path
    if split == "train.clean.960":
        from datasets import concatenate_datasets

        ds1 = load_dataset(
            "librispeech_asr", "clean", split="train.100",
            streaming=streaming, trust_remote_code=True,
        )
        ds2 = load_dataset(
            "librispeech_asr", "clean", split="train.360",
            streaming=streaming, trust_remote_code=True,
        )
        ds3 = load_dataset(
            "librispeech_asr", "other", split="train.500",
            streaming=streaming, trust_remote_code=True,
        )
        if not streaming:
            dataset = concatenate_datasets([ds1, ds2, ds3])
        else:
            from datasets import interleave_datasets

            dataset = interleave_datasets([ds1, ds2, ds3])
    else:
        config_map = {
            "validation.clean": ("clean", "validation"),
            "validation.other": ("other", "validation"),
            "test.clean": ("clean", "test"),
            "test.other": ("other", "test"),
        }
        if split in config_map:
            config, actual_split = config_map[split]
        else:
            config, actual_split = "clean", split

        dataset = load_dataset(
            "librispeech_asr", config, split=actual_split,
            streaming=streaming, trust_remote_code=True,
        )

    # Keep audio undecoded so we can decode via soundfile (no torchcodec dependency).
    if not streaming and "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(decode=False))

    return dataset


# ---------------------------------------------------------------------------
# Per-example preparation
# ---------------------------------------------------------------------------

def _load_audio_with_soundfile(batch):
    """Decode one audio example with soundfile from path/bytes/local path key."""
    if "audio_path" in batch:
        array, sr = sf.read(batch["audio_path"])
    elif "audio" in batch:
        audio = batch["audio"]
        if isinstance(audio, dict):
            if audio.get("path"):
                array, sr = sf.read(audio["path"])
            elif audio.get("bytes") is not None:
                array, sr = sf.read(io.BytesIO(audio["bytes"]))
            elif audio.get("array") is not None:
                array = np.asarray(audio["array"])
                sr = int(audio.get("sampling_rate", 16_000))
            else:
                raise ValueError("Unsupported audio record format")
        else:
            array, sr = sf.read(audio)
    else:
        raise KeyError("Batch must contain either 'audio_path' or 'audio'")

    if array.ndim > 1:
        array = array.mean(axis=-1)

    array = array.astype(np.float32)
    if sr != 16_000:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=16_000)

    return array


def prepare_dataset(batch, processor):
    """Process a single example: extract features and encode labels."""
    array = _load_audio_with_soundfile(batch)

    # Processor normalises audio (zero-mean, unit-var)
    inputs = processor(
        array,
        sampling_rate=16_000,
        return_tensors="np",
    )
    batch["input_values"] = inputs.input_values[0]
    batch["input_length"] = len(inputs.input_values[0])

    # HuBERT vocabulary is UPPER-CASE
    text = batch["text"].upper().strip()
    labels = processor.tokenizer(text).input_ids
    batch["labels"] = labels
    batch["transcript"] = text

    return batch


# ---------------------------------------------------------------------------
# Collator  *** FIXED: uses processor.pad() ***
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator for CTC evaluation / training.

    Uses processor.pad() so that padding values and attention masks are
    consistent with what the model expects after feature-extractor
    normalisation.
    """

    processor: Wav2Vec2Processor
    max_input_length: int = 320000  # 20 seconds at 16 kHz

    def __call__(self, features):
        # Filter out utterances that exceed max length; fall back to full
        # batch if all items exceed the limit (avoids returning None which
        # would cause DDP rank desynchronisation).
        filtered = [
            f for f in features
            if len(f["input_values"]) <= self.max_input_length
        ]
        if filtered:
            features = filtered
        # else: accept all (rare; CTC zero_infinity handles unusually long inputs)

        # *** FIX: use processor.pad() instead of manual pad_sequence(padding_value=0.0) ***
        # processor.pad() uses the feature extractor's padding value and
        # produces correct attention masks that are consistent with how the
        # model was trained.
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Pad labels with -100 (CTC ignore index)
        label_features = [f["labels"] for f in features]
        max_label_len = max(len(l) for l in label_features)
        padded_labels = torch.full(
            (len(features), max_label_len), -100, dtype=torch.long,
        )
        for i, l in enumerate(label_features):
            padded_labels[i, : len(l)] = torch.tensor(l, dtype=torch.long)

        batch["labels"] = padded_labels
        batch["transcripts"] = [f["transcript"] for f in features]

        return batch


# ---------------------------------------------------------------------------
# Lazy dataset wrapper (avoids upfront map() and disk cache)
# ---------------------------------------------------------------------------

class LazyLibriSpeechDataset(torch.utils.data.Dataset):
    """Wraps a HF dataset and applies prepare_dataset lazily per item,
    avoiding the memory-intensive upfront .map() + Arrow cache write."""

    def __init__(self, hf_dataset, processor):
        self.ds = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return prepare_dataset(self.ds[idx], self.processor)


# ---------------------------------------------------------------------------
# Convenience loader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    processor,
    train_split: str = "train.clean.960",
    batch_size: int = 4,
    max_input_length: int = 320000,
    num_workers: int = 4,
    data_dir: str = None,
):
    """Create training and validation dataloaders."""
    train_ds = LazyLibriSpeechDataset(
        load_librispeech(train_split, data_dir=data_dir), processor
    )
    val_clean = LazyLibriSpeechDataset(
        load_librispeech("validation.clean", data_dir=data_dir), processor
    )
    val_other = LazyLibriSpeechDataset(
        load_librispeech("validation.other", data_dir=data_dir), processor
    )

    collator = DataCollatorCTCWithPadding(
        processor=processor,
        max_input_length=max_input_length,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_clean_loader = DataLoader(
        val_clean, batch_size=batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=num_workers,
        pin_memory=True,
    )
    val_other_loader = DataLoader(
        val_other, batch_size=batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_clean_loader, val_other_loader