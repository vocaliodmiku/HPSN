"""
MEG-MASC Neural Encoding Evaluation
=====================================
Evaluates brain alignment of HPSN block representations against MEG data
from the MEG-MASC dataset (Gwilliams et al. 2023).

Three evaluation tracks:
  Track A — Linear Encoding: Ridge regression from model features → MEG sensors
  Track B — Hierarchical RSA: Block-level RDMs vs time-resolved neural RDMs
  Track C — Word-locked ERP analysis: N400 prediction with/without inhibition

Pipeline:
  1. Preprocess MEG: bandpass filter, epoch around word onsets
  2. Extract HPSN features for each story's audio
  3. Align model frames to MEG time points
  4. Track A: Ridge regression encoding model
  5. Track B: RSA block-to-time alignment
  6. Track C: N400 analysis at inhibition boundary

Usage:
    # Full pipeline
    python meg_encoding.py \
        --checkpoint runs/v2_twopass/stage2_best.pt \
        --meg_dir /scratch/jsm04005/fie24002/DATA/gwilliams2022/download/osfstorage \
        --output_dir results/meg_encoding

    # Quick test (1 subject, 1 session)
    python meg_encoding.py \
        --checkpoint runs/v2_twopass/stage2_best.pt \
        --meg_dir /scratch/jsm04005/fie24002/DATA/gwilliams2022/download/osfstorage \
        --output_dir results/meg_encoding \
        --subjects 1 --sessions 0
"""

import argparse
import ast
import csv
import json
import logging
import os
from pathlib import Path

import mne
import numpy as np
import torch
import torchaudio
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2Processor

from model import HPSNHubert

logging.getLogger("sklearn").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

mne.set_log_level("WARNING")


# ---------------------------------------------------------------------------
# MEG Preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_meg(meg_con_path, l_freq=0.5, h_freq=30.0, resample_freq=200):
    """
    Load and preprocess a single MEG run.

    Steps:
      1. Load raw KIT .con file
      2. Pick MEG channels (208 magnetometers)
      3. Bandpass filter (0.5–30 Hz)
      4. Resample to 200 Hz (sufficient for ERP analysis up to 100 Hz)

    Returns:
        raw: preprocessed MNE Raw object
    """
    raw = mne.io.read_raw_kit(meg_con_path, preload=True)
    raw.pick_types(meg=True, ref_meg=False, misc=False, stim=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=-1)
    if resample_freq and raw.info["sfreq"] != resample_freq:
        raw.resample(resample_freq, npad="auto")
    return raw


def parse_events(events_tsv_path):
    """
    Parse BIDS events.tsv for word onsets (sentence condition only).

    The 'onset' column gives MEG-relative time. The 'start' field inside
    trial_type gives the word onset *within* the audio clip — this is
    needed to index into model features extracted from that clip.

    Returns:
        word_events: list of dicts with keys:
            onset, duration, start_in_audio, word, story, condition,
            word_index, sound
    """
    word_events = []
    # encoding='utf-8-sig' strips BOM from column names
    with open(events_tsv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            trial = ast.literal_eval(row["trial_type"])
            if trial["kind"] == "word" and trial.get("condition") == "sentence":
                word_events.append({
                    "onset": float(row["onset"]),
                    "duration": float(row["duration"]),
                    "start_in_audio": float(trial.get("start", 0)),
                    "word": trial["word"],
                    "story": trial["story"],
                    "condition": trial["condition"],
                    "word_index": int(trial.get("word_index", 0)),
                    "sound": trial.get("sound", ""),
                })
    return word_events


def epoch_meg_around_words(raw, word_events, tmin=-0.2, tmax=0.8):
    """
    Create MNE Epochs around word onsets.

    Args:
        raw: preprocessed MNE Raw
        word_events: list of dicts from parse_events
        tmin, tmax: epoch window relative to word onset (seconds)

    Returns:
        epochs: MNE Epochs object
        valid_events: subset of word_events that produced valid epochs
    """
    sfreq = raw.info["sfreq"]

    # Build events array: (sample, 0, event_id)
    events_array = []
    valid_events = []
    for i, ev in enumerate(word_events):
        sample = int(ev["onset"] * sfreq)
        if 0 <= sample < raw.n_times:
            events_array.append([sample, 0, 1])
            valid_events.append(ev)

    if not events_array:
        return None, []

    events_array = np.array(events_array, dtype=int)

    epochs = mne.Epochs(
        raw, events_array, event_id=1,
        tmin=tmin, tmax=tmax,
        baseline=(tmin, 0),
        preload=True,
        reject=dict(mag=4e-12),  # reject extreme artifacts
        verbose=False,
    )

    # Update valid_events to match surviving epochs
    kept = epochs.selection
    valid_events = [valid_events[i] for i in kept]

    return epochs, valid_events


# ---------------------------------------------------------------------------
# HPSN Feature Extraction for Audio Stimuli
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_story_features(model, processor, audio_path, device,
                           bypass_novel=False):
    """
    Extract block-level features from HPSN for one audio file.

    Returns:
        raw_features: list of 5 arrays, each (T_enc, D)
        refined_features: list of 5 arrays, each (T_enc, D) (None if bypass)
        frame_times: array of (T_enc,) — time in seconds for each frame
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    waveform = waveform.mean(dim=0, keepdim=True)  # mono

    model.eval()
    out = model(
        input_values=waveform.to(device),
        return_block_outputs=True,
        bypass_novel=bypass_novel,
    )

    raw = out["raw_block_outputs"]       # [o_feat, o_b0, ..., o_b4]
    refined = out["refined_block_outputs"]  # [r0, ..., r4] or None

    num_levels = model.num_blocks
    raw_feats = [raw[b + 1][0].cpu().float().numpy() for b in range(num_levels)]
    if refined is not None:
        ref_feats = [refined[b][0].cpu().float().numpy() for b in range(num_levels)]
    else:
        ref_feats = None

    # HuBERT frame rate: ~20ms per frame (50 Hz)
    T_enc = raw_feats[0].shape[0]
    frame_times = np.arange(T_enc) * 0.02  # 20ms stride

    return raw_feats, ref_feats, frame_times


def align_features_to_word(
    features, frame_times, word_onset, word_duration, method="mean",
):
    """
    Extract feature vector for a word by averaging frames within its span.

    Args:
        features: (T_enc, D) array
        frame_times: (T_enc,) time in seconds
        word_onset: onset time in seconds (relative to audio start)
        word_duration: duration in seconds
        method: "mean" or "onset" (just the onset frame)

    Returns:
        (D,) feature vector
    """
    if method == "onset":
        idx = np.argmin(np.abs(frame_times - word_onset))
        return features[idx]
    else:
        start = word_onset
        end = word_onset + max(word_duration, 0.02)
        mask = (frame_times >= start) & (frame_times <= end)
        if mask.sum() == 0:
            idx = np.argmin(np.abs(frame_times - word_onset))
            return features[idx]
        return features[mask].mean(axis=0)


# ---------------------------------------------------------------------------
# Track A: Linear Encoding Model
# ---------------------------------------------------------------------------

def run_linear_encoding(
    model_features, meg_data, n_splits=5, alphas=None,
):
    """
    Train ridge regression: model features → MEG sensor values.

    Args:
        model_features: dict of {level_name: (N_words, D)} arrays
        meg_data: (N_words, N_sensors, N_times) from epochs

    Returns:
        results: dict of {level_name: {mean_r, per_sensor_r, per_time_r}}
    """
    if alphas is None:
        alphas = np.logspace(0, 5, 8)

    N_words, N_sensors, N_times = meg_data.shape
    results = {}

    for level_name, feats in model_features.items():
        logger.debug(
            f"  Encoding: {level_name} ({feats.shape[1]}D -> {N_sensors} sensors x {N_times} time)"
        )

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        # Flatten MEG to (N_words, N_sensors * N_times) so we fit one RidgeCV
        # per fold instead of one per time point — ~200x faster.
        Y_flat = meg_data.reshape(N_words, N_sensors * N_times)
        pred_flat = np.zeros_like(Y_flat)

        for train_idx, test_idx in kf.split(feats):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(feats[train_idx])
            X_test = scaler.transform(feats[test_idx])
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(X_train, Y_flat[train_idx])
            pred_flat[test_idx] = ridge.predict(X_test)

        # Reshape predictions and compute Pearson r per (sensor, time)
        Y = Y_flat.reshape(N_words, N_sensors, N_times)
        P = pred_flat.reshape(N_words, N_sensors, N_times)
        Y_c = Y - Y.mean(axis=0, keepdims=True)
        P_c = P - P.mean(axis=0, keepdims=True)
        num = np.sum(Y_c * P_c, axis=0)
        den = np.sqrt(np.sum(Y_c ** 2, axis=0) * np.sum(P_c ** 2, axis=0))
        correlations = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        results[level_name] = {
            "mean_r": float(correlations.mean()),
            "peak_r": float(correlations.max()),
            "per_sensor_mean_r": correlations.mean(axis=1).tolist(),
            "per_time_mean_r": correlations.mean(axis=0).tolist(),
        }
        logger.debug(f"    mean r={correlations.mean():.4f}, peak r={correlations.max():.4f}")

    return results


# ---------------------------------------------------------------------------
# Track B: Hierarchical RSA
# ---------------------------------------------------------------------------

def compute_rdm(features, metric="cosine"):
    """
    Compute representational dissimilarity matrix.

    Args:
        features: (N, D) array
    Returns:
        rdm: (N, N) dissimilarity matrix (upper triangle)
    """
    from scipy.spatial.distance import pdist, squareform
    rdm = squareform(pdist(features, metric=metric))
    return rdm


def rdm_upper_tri(rdm):
    """Extract upper triangle of RDM as vector."""
    idx = np.triu_indices(rdm.shape[0], k=1)
    return rdm[idx]


def run_hierarchical_rsa(
    model_features, meg_data, epoch_times,
    window_size_ms=50, step_ms=10, sfreq=200,
):
    """
    Compute RSA between model block RDMs and time-resolved neural RDMs.

    Args:
        model_features: dict of {level_name: (N_words, D)}
        meg_data: (N_words, N_sensors, N_times)
        epoch_times: (N_times,) time vector in seconds
        window_size_ms: sliding window size
        step_ms: sliding window step

    Returns:
        rsa_matrix: {level_name: array of Spearman ρ at each time window}
        time_centers: array of window center times
    """
    N_words, N_sensors, N_times = meg_data.shape
    window_samples = int(window_size_ms / 1000 * sfreq)
    step_samples = int(step_ms / 1000 * sfreq)

    # Compute model RDMs once
    model_rdms = {}
    for name, feats in model_features.items():
        model_rdms[name] = rdm_upper_tri(compute_rdm(feats))

    # Sliding window over time
    time_centers = []
    rsa_results = {name: [] for name in model_features}

    for start in range(0, N_times - window_samples, step_samples):
        end = start + window_samples
        t_center = epoch_times[start + window_samples // 2]
        time_centers.append(float(t_center))

        # Neural RDM: average across sensors within window, then compute RDM
        # Shape: (N_words, N_sensors * window_samples) — concatenate sensor×time
        neural_window = meg_data[:, :, start:end].reshape(N_words, -1)
        neural_rdm_vec = rdm_upper_tri(compute_rdm(neural_window))

        for name, model_rdm_vec in model_rdms.items():
            rho, _ = stats.spearmanr(model_rdm_vec, neural_rdm_vec)
            rsa_results[name].append(float(rho))

    return rsa_results, time_centers


def compute_rsa_noise_ceiling(meg_data, epoch_times, window_size_ms=50,
                              step_ms=10, sfreq=200):
    """
    Split-half noise ceiling for neural RDMs.

    Splits words into odd/even, computes neural RDMs for each half at each
    time window, and returns Spearman correlation between the two halves.
    This gives the maximum achievable RSA for any model.

    Returns:
        noise_ceiling: list of floats (one per time window)
        time_centers: list of floats
    """
    N_words, N_sensors, N_times = meg_data.shape
    window_samples = int(window_size_ms / 1000 * sfreq)
    step_samples = int(step_ms / 1000 * sfreq)

    odd_idx = np.arange(0, N_words, 2)
    even_idx = np.arange(1, N_words, 2)
    n_min = min(len(odd_idx), len(even_idx))
    odd_idx, even_idx = odd_idx[:n_min], even_idx[:n_min]

    time_centers = []
    ceiling = []

    for start in range(0, N_times - window_samples, step_samples):
        end = start + window_samples
        t_center = epoch_times[start + window_samples // 2]
        time_centers.append(float(t_center))

        window_odd = meg_data[odd_idx, :, start:end].reshape(n_min, -1)
        window_even = meg_data[even_idx, :, start:end].reshape(n_min, -1)

        rdm_odd = rdm_upper_tri(compute_rdm(window_odd))
        rdm_even = rdm_upper_tri(compute_rdm(window_even))

        rho, _ = stats.spearmanr(rdm_odd, rdm_even)
        ceiling.append(float(rho))

    return ceiling, time_centers


# ---------------------------------------------------------------------------
# Track C: N400 / Inhibition Analysis
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_inhibited_features(model, processor, audio_path, device):
    """
    Extract features at the inhibition boundary, both before and after inhibition.

    Returns:
        pre_inhib: (T_enc, D) — raw block output at inhibition boundary
        post_inhib: (T_enc, D) — after SubspaceInhibition
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.mean(dim=0, keepdim=True)

    model.eval()
    out = model(
        input_values=waveform.to(device),
        return_block_outputs=True,
        bypass_novel=False,
    )

    raw = out["raw_block_outputs"]
    b = model.inhibition_boundary
    pre_inhib = raw[b + 1][0].cpu().float().numpy()  # o_b before inhibition

    # Apply inhibition manually to get post-inhibition features
    o_b = raw[b + 1].to(device)
    post = model.inhibition(o_b)
    post_inhib = post[0].cpu().float().numpy()

    return pre_inhib, post_inhib


def run_n400_analysis(
    pre_inhib_feats, post_inhib_feats, meg_data, epoch_times,
    n400_window=(0.3, 0.5),
):
    """
    Compare RSA of pre- vs post-inhibition features in the N400 window.

    Args:
        pre_inhib_feats: (N_words, D)
        post_inhib_feats: (N_words, D)
        meg_data: (N_words, N_sensors, N_times)
        epoch_times: (N_times,) time array
        n400_window: (start, end) in seconds

    Returns:
        results dict with RSA values
    """
    # Find N400 time indices
    t_start, t_end = n400_window
    t_mask = (epoch_times >= t_start) & (epoch_times <= t_end)

    if t_mask.sum() == 0:
        return {"error": "N400 window not in epoch range"}

    # Neural features in N400 window
    neural_n400 = meg_data[:, :, t_mask].reshape(meg_data.shape[0], -1)
    neural_rdm = rdm_upper_tri(compute_rdm(neural_n400))

    # Model RDMs
    pre_rdm = rdm_upper_tri(compute_rdm(pre_inhib_feats))
    post_rdm = rdm_upper_tri(compute_rdm(post_inhib_feats))

    rho_pre, p_pre = stats.spearmanr(pre_rdm, neural_rdm)
    rho_post, p_post = stats.spearmanr(post_rdm, neural_rdm)

    return {
        "pre_inhibition_rho": float(rho_pre),
        "pre_inhibition_p": float(p_pre),
        "post_inhibition_rho": float(rho_post),
        "post_inhibition_p": float(p_post),
        "delta_rho": float(rho_post - rho_pre),
        "n400_window": list(n400_window),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def get_task_audio_mapping(meg_dir, sub, ses):
    """
    Map each task to its audio files by parsing events.tsv.

    Returns:
        task_audio: dict {task_id: list of unique audio file paths}
    """
    meg_path = Path(meg_dir) / f"sub-{sub:02d}" / f"ses-{ses}"/ "meg"
    task_audio = {}

    for events_file in sorted(meg_path.glob("*_events.tsv")):
        # Extract task id from filename
        parts = events_file.stem.split("_")
        task_id = [p for p in parts if p.startswith("task-")][0]

        sounds = set()
        with open(events_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                trial = ast.literal_eval(row["trial_type"])
                if trial["kind"] == "sound":
                    sounds.add(trial.get("sound", ""))

        task_audio[task_id] = sorted(sounds)

    return task_audio


def process_one_run(
    model, processor, device, meg_dir,
    sub, ses, task_id,
    tmin=-0.2, tmax=0.8,
    max_words=300,
    bypass_novel=False,
    audio_feature_cache=None,
):
    """
    Process one MEG run: preprocess, extract features, align.

    Returns:
        meg_data: (N_words, N_sensors, N_times) epoch data
        epoch_times: (N_times,) time vector
        model_features: dict of {level_name: (N_words, D)}
        word_events: list of event dicts
        pre_inhib_feats: (N_words, D) or None
        post_inhib_feats: (N_words, D) or None
    """
    sub_str = f"sub-{sub:02d}"
    ses_str = f"ses-{ses}"
    meg_path = Path(meg_dir) / sub_str / ses_str / "meg"

    con_file = meg_path / f"{sub_str}_{ses_str}_{task_id}_meg.con"
    events_file = meg_path / f"{sub_str}_{ses_str}_{task_id}_events.tsv"

    if not con_file.exists() or not events_file.exists():
        logger.warning(f"  Missing files for {sub_str}/{ses_str}/{task_id}")
        return None

    # 1. Preprocess MEG
    raw = load_and_preprocess_meg(str(con_file))
    sfreq = raw.info["sfreq"]

    # 2. Parse events
    word_events = parse_events(str(events_file))
    if not word_events:
        logger.warning(f"  No word events in {events_file.name}")
        return None

    # 3. Epoch around word onsets
    epochs, valid_events = epoch_meg_around_words(raw, word_events, tmin, tmax)
    del raw

    if epochs is None or len(valid_events) == 0:
        logger.warning(f"  No valid epochs for {sub_str}/{ses_str}/{task_id}")
        return None

    # Limit words for memory (0 = no limit)
    if max_words > 0 and len(valid_events) > max_words:
        valid_events = valid_events[:max_words]
        epochs = epochs[:max_words]

    meg_data = epochs.get_data()  # (N, N_sensors, N_times)
    epoch_times = epochs.times
    del epochs

    # 4. Extract features for each unique audio file in this run
    if audio_feature_cache is None:
        audio_feature_cache = {}
    audio_to_feats = audio_feature_cache
    # Build case-insensitive lookup for audio directory once
    audio_dir = Path(meg_dir) / "stimuli" / "audio"
    _audio_lower_map = {}
    if audio_dir.exists():
        for f in audio_dir.iterdir():
            _audio_lower_map[f.name.lower()] = f

    for ev in valid_events:
        sound = ev["sound"]
        if sound not in audio_to_feats:
            audio_path = Path(meg_dir) / sound
            if not audio_path.exists():
                audio_path = audio_dir / Path(sound).name
            # Case-insensitive fallback (events.tsv uses CamelCase,
            # files on disk are lowercase)
            if not audio_path.exists():
                lower = Path(sound).name.lower()
                if lower in _audio_lower_map:
                    audio_path = _audio_lower_map[lower]
            if audio_path.exists():
                raw_f, ref_f, ftimes = extract_story_features(
                    model, processor, str(audio_path), device,
                    bypass_novel=bypass_novel,
                )
                audio_to_feats[sound] = (raw_f, ref_f, ftimes)
            else:
                logger.warning(f"  Audio not found: {sound}")

    # Log audio extraction summary
    unique_sounds = set(ev["sound"] for ev in valid_events)
    found = sum(1 for s in unique_sounds if s in audio_to_feats)
    logger.info(f"  Audio: {found}/{len(unique_sounds)} unique files found, "
                f"{len(valid_events)} word events")

    # 5. Align: get word-level features using start_in_audio
    num_levels = model.num_blocks
    has_refined = not bypass_novel
    raw_word_feats = [[] for _ in range(num_levels)]
    ref_word_feats = [[] for _ in range(num_levels)] if has_refined else None
    keep_mask = []

    for ev in valid_events:
        sound = ev["sound"]
        if sound not in audio_to_feats:
            keep_mask.append(False)
            continue

        raw_f, ref_f, ftimes = audio_to_feats[sound]
        word_onset = ev["start_in_audio"]
        word_dur = ev["duration"]

        for b in range(num_levels):
            raw_word_feats[b].append(
                align_features_to_word(raw_f[b], ftimes, word_onset, word_dur)
            )
            if has_refined:
                ref_word_feats[b].append(
                    align_features_to_word(ref_f[b], ftimes, word_onset, word_dur)
                )

        keep_mask.append(True)

    if sum(keep_mask) < 20:
        logger.warning(f"  Too few aligned words ({sum(keep_mask)})")
        return None

    # Filter meg_data to keep only aligned words
    meg_data = meg_data[np.array(keep_mask)]

    model_features = {}
    for b in range(num_levels):
        model_features[f"raw_level_{b}"] = np.stack(raw_word_feats[b])
        if has_refined:
            model_features[f"refined_level_{b}"] = np.stack(ref_word_feats[b])

    return {
        "meg_data": meg_data,
        "epoch_times": epoch_times,
        "model_features": model_features,
        "word_events": [ev for ev, k in zip(valid_events, keep_mask) if k],
    }


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MEG-MASC Neural Encoding Evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--pretrained", type=str, default="facebook/hubert-large-ls960-ft")
    p.add_argument("--meg_dir", type=str, required=True,
                   help="Path to MEG-MASC BIDS root (contains sub-XX/)")
    p.add_argument("--output_dir", type=str, default="results/meg_encoding")
    p.add_argument("--subjects", nargs="+", type=int, default=None,
                   help="Subject numbers to process (default: all 1-27)")
    p.add_argument("--sessions", nargs="+", type=int, default=None,
                   help="Session numbers (default: 0 and 1)")
    p.add_argument("--max_words", type=int, default=0,
                   help="Max word events per run (0=no limit)")
    p.add_argument("--tmin", type=float, default=-0.2)
    p.add_argument("--tmax", type=float, default=0.8)
    # Model config
    p.add_argument("--refine_num_heads", type=int, default=4)
    p.add_argument("--inhibition_rank", type=int, default=64)
    p.add_argument("--inhibition_boundary", type=int, default=2)
    # Track selection
    p.add_argument("--skip_encoding", action="store_true",
                   help="Skip Track A (linear encoding)")
    p.add_argument("--skip_rsa", action="store_true",
                   help="Skip Track B (hierarchical RSA)")
    p.add_argument("--bypass_novel", action="store_true",
                   help="Skip HPSN refinement (evaluate pretrained HuBERT baseline)")
    p.add_argument(
        "--log_level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: WARNING)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=getattr(logging, args.log_level),
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect model config
    ckpt_dir = Path(args.checkpoint).parent
    args_file = ckpt_dir / "args.json"
    if args_file.exists():
        with open(args_file) as f:
            train_args = json.load(f)
        args.refine_num_heads = train_args.get("refine_num_heads", args.refine_num_heads)
        args.inhibition_rank = train_args.get("inhibition_rank", args.inhibition_rank)
        args.inhibition_boundary = train_args.get("inhibition_boundary", args.inhibition_boundary)

    # Load model
    logger.info("Loading HPSN model...")
    processor = Wav2Vec2Processor.from_pretrained(args.pretrained)
    model = HPSNHubert(
        pretrained=args.pretrained,
        refine_num_heads=args.refine_num_heads,
        inhibition_rank=args.inhibition_rank,
        inhibition_boundary=args.inhibition_boundary,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    logger.info(f"Loaded: {args.checkpoint}")

    # Determine subjects and sessions
    if args.subjects is None:
        args.subjects = list(range(1, 28))
    if args.sessions is None:
        args.sessions = [0, 1]

    # ==================================================================
    # Process each subject × session × task
    # ==================================================================
    all_encoding_results = []
    all_rsa_results = []
    all_n400_results = []
    audio_feature_cache = {}  # cache audio features across all subjects/sessions

    for sub in args.subjects:
        for ses in args.sessions:
            sub_str = f"sub-{sub:02d}"
            ses_str = f"ses-{ses}"
            meg_path = Path(args.meg_dir) / sub_str / ses_str / "meg"

            if not meg_path.exists():
                logger.info(f"Skipping {sub_str}/{ses_str} (not found)")
                continue

            # Find all tasks
            con_files = sorted(meg_path.glob("*_meg.con"))
            for con_file in con_files:
                task_id = [p for p in con_file.stem.split("_") if p.startswith("task-")][0]
                logger.info(f"\nProcessing {sub_str}/{ses_str}/{task_id}...")

                result = process_one_run(
                    model, processor, device, args.meg_dir,
                    sub, ses, task_id,
                    tmin=args.tmin, tmax=args.tmax,
                    max_words=args.max_words,
                    bypass_novel=args.bypass_novel,
                    audio_feature_cache=audio_feature_cache,
                )

                if result is None:
                    continue

                meg_data = result["meg_data"]
                epoch_times = result["epoch_times"]
                feats = result["model_features"]
                N_words = meg_data.shape[0]
                run_id = f"{sub_str}_{ses_str}_{task_id}"

                logger.info(f"  {N_words} word epochs, {meg_data.shape[1]} sensors, "
                            f"{meg_data.shape[2]} time points")

                # Track A: Linear Encoding
                if not args.skip_encoding:
                    logger.info(f"  Track A: Linear Encoding")
                    enc_results = run_linear_encoding(feats, meg_data, n_splits=3)
                    enc_results["run_id"] = run_id
                    enc_results["n_words"] = N_words
                    all_encoding_results.append(enc_results)

                # Track B: RSA
                if not args.skip_rsa:
                    logger.info(f"  Track B: Hierarchical RSA")
                    rsa_results, time_centers = run_hierarchical_rsa(
                        feats, meg_data, epoch_times,
                    )
                    noise_ceiling, _ = compute_rsa_noise_ceiling(
                        meg_data, epoch_times,
                    )
                    all_rsa_results.append({
                        "run_id": run_id,
                        "n_words": N_words,
                        "time_centers": time_centers,
                        "rsa": rsa_results,
                        "noise_ceiling": noise_ceiling,
                    })

                    # Find peak time for each level
                    for name, rho_values in rsa_results.items():
                        peak_idx = np.argmax(rho_values)
                        peak_time = time_centers[peak_idx]
                        peak_rho = rho_values[peak_idx]
                        logger.debug(f"    {name}: peak rho={peak_rho:.4f} at t={peak_time:.3f}s")

                # Track C: N400 at inhibition boundary (skip in bypass mode)
                b = model.inhibition_boundary
                if not args.bypass_novel and f"raw_level_{b}" in feats:
                    logger.info(f"  Track C: N400 analysis")
                    # Use raw and apply inhibition manually
                    # Pre = raw at boundary, Post = inhibited
                    pre_feats = feats[f"raw_level_{b}"]
                    # For post-inhibition, we need to run inhibition on the raw features
                    pre_t = torch.tensor(pre_feats, dtype=torch.float32, device=device)
                    post_t = model.inhibition(pre_t.unsqueeze(0) if pre_t.dim() == 2 else pre_t)
                    post_feats = post_t.squeeze(0).detach().cpu().numpy()

                    n400_res = run_n400_analysis(
                        pre_feats, post_feats, meg_data, epoch_times,
                    )
                    n400_res["run_id"] = run_id
                    all_n400_results.append(n400_res)
                    logger.info(f"    Pre-inhib ρ={n400_res['pre_inhibition_rho']:.4f}, "
                                f"Post-inhib ρ={n400_res['post_inhibition_rho']:.4f}, "
                                f"Δ={n400_res['delta_rho']:+.4f}")

                # Free memory
                del meg_data, feats

    # ==================================================================
    # Aggregate and save results
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)

    aggregate = {}

    # Track A summary
    if all_encoding_results:
        logger.info("\nTrack A — Linear Encoding (mean r across runs):")
        level_rs = {}
        for enc in all_encoding_results:
            for key, val in enc.items():
                if isinstance(val, dict) and "mean_r" in val:
                    level_rs.setdefault(key, []).append(val["mean_r"])

        enc_summary = {}
        for level, rs in sorted(level_rs.items()):
            mean_r = np.mean(rs)
            std_r = np.std(rs)
            enc_summary[level] = {"mean_r": float(mean_r), "std_r": float(std_r), "n_runs": len(rs)}
            logger.info(f"  {level}: r={mean_r:.4f}±{std_r:.4f} ({len(rs)} runs)")
        aggregate["encoding"] = enc_summary

    # Track B summary
    if all_rsa_results:
        logger.info("\nTrack B — RSA Peak Times (mean across runs):")
        rsa_summary = {}
        for rsa in all_rsa_results:
            for name, rho_values in rsa["rsa"].items():
                peak_idx = np.argmax(rho_values)
                peak_time = rsa["time_centers"][peak_idx]
                peak_rho = rho_values[peak_idx]
                rsa_summary.setdefault(name, {"peak_times": [], "peak_rhos": []})
                rsa_summary[name]["peak_times"].append(peak_time)
                rsa_summary[name]["peak_rhos"].append(peak_rho)

        for name, data in sorted(rsa_summary.items()):
            mean_peak_t = np.mean(data["peak_times"])
            mean_peak_rho = np.mean(data["peak_rhos"])
            data["mean_peak_time"] = float(mean_peak_t)
            data["mean_peak_rho"] = float(mean_peak_rho)
            logger.info(f"  {name}: peak t={mean_peak_t:.3f}s, ρ={mean_peak_rho:.4f}")
        aggregate["rsa"] = {k: {"mean_peak_time": v["mean_peak_time"],
                                 "mean_peak_rho": v["mean_peak_rho"]}
                            for k, v in rsa_summary.items()}

        # Noise ceiling summary
        all_ceilings = [r["noise_ceiling"] for r in all_rsa_results
                        if "noise_ceiling" in r]
        if all_ceilings:
            mean_ceiling = np.mean(all_ceilings, axis=0)
            peak_ceil = float(np.max(mean_ceiling))
            mean_ceil = float(np.mean(mean_ceiling))
            aggregate["rsa_noise_ceiling"] = {
                "peak": peak_ceil, "mean": mean_ceil,
            }
            logger.info(f"  Noise ceiling: peak={peak_ceil:.4f}, mean={mean_ceil:.4f}")

    # Track C summary
    if all_n400_results:
        logger.info("\nTrack C — N400 Inhibition Effect:")
        pre_rhos = [r["pre_inhibition_rho"] for r in all_n400_results]
        post_rhos = [r["post_inhibition_rho"] for r in all_n400_results]
        deltas = [r["delta_rho"] for r in all_n400_results]

        n400_summary = {
            "mean_pre_rho": float(np.mean(pre_rhos)),
            "mean_post_rho": float(np.mean(post_rhos)),
            "mean_delta": float(np.mean(deltas)),
            "n_runs": len(all_n400_results),
        }
        # One-sample t-test on delta
        if len(deltas) > 1:
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            n400_summary["ttest_t"] = float(t_stat)
            n400_summary["ttest_p"] = float(p_val)
            logger.info(f"  Δρ = {np.mean(deltas):+.4f} (t={t_stat:.2f}, p={p_val:.4f})")
        else:
            logger.info(f"  Δρ = {np.mean(deltas):+.4f} (single run)")
        aggregate["n400"] = n400_summary

    # Save everything
    output = {
        "args": vars(args),
        "aggregate": aggregate,
        "per_run_encoding": all_encoding_results,
        "per_run_rsa": all_rsa_results,
        "per_run_n400": all_n400_results,
    }

    output_path = Path(args.output_dir) / "meg_encoding_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
