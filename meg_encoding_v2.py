"""
MEG-MASC Neural Encoding Evaluation v2
========================================
Revised evaluation that directly tests HPSN's top-down refinement and
lateral inhibition mechanisms.

Four evaluation tracks:

  Track A — Frame-level Temporal Encoding:
      Ridge regression from model features at frame level (20ms) with
      temporal lags.  Each block b should peak at a different lag —
      testing hierarchical temporal alignment.

  Track B — Correction-Signal Encoding:
      The top-down update correction[b] = refined[b] − raw[b] is regressed
      against MEG.  If refinement is brain-like, these prediction-error
      signals should explain MEG variance beyond raw features.

  Track C — Inhibition × Competition Interaction (N400):
      Words are split by phonological neighborhood density (via FastLex).
      Test: does post-inhibition encoding advantage increase for
      high-density words specifically in the N400 window?

  Track D — CPC Prediction Error vs N400 Amplitude:
      Per-word CPC prediction error ||raw[b] − proj(refined[b+1])|| is
      correlated with N400 amplitude.  Directly tests predictive coding.

Usage:
    python meg_encoding_v2.py \\
        --checkpoint runs/v6_full/stage2_step60000.pt \\
        --meg_dir /path/to/gwilliams2022/osfstorage \\
        --output_dir results/meg_v2 \\
        --density_csv data/meg_word_density.csv \\
        --log_level INFO

    # Quick test
    python meg_encoding_v2.py ... --subjects 1 --sessions 0
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
# MEG Preprocessing  (shared with v1)
# ---------------------------------------------------------------------------

def load_and_preprocess_meg(meg_con_path, l_freq=0.5, h_freq=30.0,
                            resample_freq=200):
    raw = mne.io.read_raw_kit(meg_con_path, preload=True)
    raw.pick_types(meg=True, ref_meg=False, misc=False, stim=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=-1)
    if resample_freq and raw.info["sfreq"] != resample_freq:
        raw.resample(resample_freq, npad="auto")
    return raw


def parse_events(events_tsv_path):
    word_events = []
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
    sfreq = raw.info["sfreq"]
    events_array, valid_events = [], []
    for ev in word_events:
        sample = int(ev["onset"] * sfreq)
        if 0 <= sample < raw.n_times:
            events_array.append([sample, 0, 1])
            valid_events.append(ev)
    if not events_array:
        return None, []
    events_array = np.array(events_array, dtype=int)
    epochs = mne.Epochs(
        raw, events_array, event_id=1,
        tmin=tmin, tmax=tmax, baseline=(tmin, 0),
        preload=True, reject=dict(mag=4e-12), verbose=False,
    )
    kept = epochs.selection
    valid_events = [valid_events[i] for i in kept]
    return epochs, valid_events


# ---------------------------------------------------------------------------
# HPSN Feature Extraction — returns both raw & refined at frame level
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_story_features(model, processor, audio_path, device,
                           bypass_novel=False):
    """
    Returns:
        raw_feats: list of 5 arrays, each (T_enc, D)
        ref_feats: list of 5 arrays, each (T_enc, D) or None
        frame_times: (T_enc,) seconds per frame
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    waveform = waveform.mean(dim=0, keepdim=True)

    model.eval()
    out = model(
        input_values=waveform.to(device),
        return_block_outputs=True,
        bypass_novel=bypass_novel,
    )

    raw = out["raw_block_outputs"]        # [o_feat, o_b0, ..., o_b4]
    refined = out["refined_block_outputs"]  # [r0, ..., r4] or None

    num_levels = model.num_blocks
    raw_feats = [raw[b + 1][0].cpu().float().numpy() for b in range(num_levels)]
    if refined is not None:
        ref_feats = [refined[b][0].cpu().float().numpy()
                     for b in range(num_levels)]
    else:
        ref_feats = None

    T_enc = raw_feats[0].shape[0]
    frame_times = np.arange(T_enc) * 0.02  # 20ms stride

    return raw_feats, ref_feats, frame_times


@torch.inference_mode()
def compute_cpc_errors(model, raw_feats_list, ref_feats_list, device):
    """
    Compute per-frame CPC prediction error for each pair (b+1 → b).

    Args:
        model: HPSNHubert with cpc_projections
        raw_feats_list: list of 5 arrays (T, D) — raw block outputs
        ref_feats_list: list of 5 arrays (T, D) — refined block outputs

    Returns:
        cpc_errors: list of 4 arrays, each (T,) — ||raw[b] - proj(refined[b+1])||
    """
    num_levels = model.num_blocks
    cpc_errors = []
    for b in range(num_levels - 1):
        ref_upper = torch.tensor(ref_feats_list[b + 1], dtype=torch.float32,
                                 device=device).unsqueeze(0)  # (1, T, D)
        raw_lower = torch.tensor(raw_feats_list[b], dtype=torch.float32,
                                 device=device).unsqueeze(0)
        pred = model.cpc_projections[b](ref_upper)  # (1, T, D)
        err = (raw_lower - pred).norm(dim=-1).squeeze(0).cpu().numpy()  # (T,)
        cpc_errors.append(err)
    return cpc_errors


# ---------------------------------------------------------------------------
# Phonological Neighborhood Density
# ---------------------------------------------------------------------------

def load_density_map(density_csv):
    """
    Load word → DAS neighbor count mapping from pre-computed FastLex output.

    Expects a CSV with at least columns: 'Word' (or 'word') and 'das_nb_ph'.
    Returns dict {lowercase_word: das_nb_ph_count}.
    """
    density = {}
    with open(density_csv) as f:
        reader = csv.DictReader(f)
        # Determine column names (FastLex uses capitalised columns from ELP)
        word_col = None
        das_col = None
        for col in reader.fieldnames:
            if col.lower() in ("word", "item"):
                word_col = col
            if col.lower() in ("das_nb_ph",):
                das_col = col
        if word_col is None or das_col is None:
            raise ValueError(
                f"density CSV must have word and das_nb_ph columns, "
                f"found: {reader.fieldnames}"
            )
        for row in reader:
            w = row[word_col].strip().lower()
            try:
                density[w] = int(row[das_col])
            except (ValueError, KeyError):
                continue
    logger.info(f"Loaded density for {len(density)} words")
    return density


# ---------------------------------------------------------------------------
# Feature alignment helpers
# ---------------------------------------------------------------------------

def get_frames_for_word(frame_times, word_onset, word_duration):
    """Return indices of frames within a word's time span."""
    start = word_onset
    end = word_onset + max(word_duration, 0.02)
    mask = (frame_times >= start) & (frame_times <= end)
    if mask.sum() == 0:
        idx = np.argmin(np.abs(frame_times - word_onset))
        return np.array([idx])
    return np.where(mask)[0]


def align_features_to_word(features, frame_times, word_onset, word_duration):
    """Average frames within word span → (D,)."""
    idxs = get_frames_for_word(frame_times, word_onset, word_duration)
    return features[idxs].mean(axis=0)


# ---------------------------------------------------------------------------
# Track A: Frame-level Temporal Encoding with Lags
# ---------------------------------------------------------------------------

def run_frame_encoding(
    model_features, meg_continuous, meg_sfreq,
    frame_times, meg_times_offset,
    lags_ms=None, n_splits=5, alphas=None,
):
    """
    Frame-level ridge regression with temporal lags.

    For each lag τ, predict MEG[t+τ] from model_frame[t].
    Each block level should peak at a different lag.

    Args:
        model_features: dict {level_name: (T_frames, D)}
        meg_continuous: (N_sensors, N_meg_samples) continuous MEG
        meg_sfreq: MEG sample rate (Hz)
        frame_times: (T_frames,) time of each model frame in seconds
        meg_times_offset: time of first MEG sample in seconds
        lags_ms: list of lags to test (default: -50 to 600ms)
        n_splits: CV folds
        alphas: ridge alphas

    Returns:
        results: {level_name: {lag_ms: mean_r_across_sensors}}
    """
    if lags_ms is None:
        lags_ms = list(range(-50, 650, 25))
    if alphas is None:
        alphas = np.logspace(0, 5, 8)

    N_sensors, N_meg = meg_continuous.shape
    results = {}

    for level_name, feats in model_features.items():
        T_frames, D = feats.shape
        lag_results = {}

        for lag in lags_ms:
            lag_samples = int(lag / 1000 * meg_sfreq)

            # Find model frames whose corresponding MEG sample exists
            meg_sample_indices = np.round(
                (frame_times - meg_times_offset) * meg_sfreq
            ).astype(int) + lag_samples

            valid = (meg_sample_indices >= 0) & (meg_sample_indices < N_meg)
            if valid.sum() < 50:
                lag_results[lag] = 0.0
                continue

            X = feats[valid]  # (N_valid, D)
            Y = meg_continuous[:, meg_sample_indices[valid]].T  # (N_valid, N_sensors)

            # Cross-validated ridge
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            pred = np.zeros_like(Y)
            for train_idx, test_idx in kf.split(X):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[train_idx])
                X_te = scaler.transform(X[test_idx])
                ridge = RidgeCV(alphas=alphas)
                ridge.fit(X_tr, Y[train_idx])
                pred[test_idx] = ridge.predict(X_te)

            # Pearson r per sensor
            Y_c = Y - Y.mean(axis=0, keepdims=True)
            P_c = pred - pred.mean(axis=0, keepdims=True)
            num = np.sum(Y_c * P_c, axis=0)
            den = np.sqrt(np.sum(Y_c ** 2, axis=0) * np.sum(P_c ** 2, axis=0))
            r = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            lag_results[lag] = float(r.mean())

        results[level_name] = lag_results
        # Report peak
        peak_lag = max(lag_results, key=lag_results.get)
        logger.info(f"    {level_name}: peak r={lag_results[peak_lag]:.4f} "
                     f"at lag={peak_lag}ms")

    return results


# ---------------------------------------------------------------------------
# Track B: Correction-Signal Encoding
# ---------------------------------------------------------------------------

def run_correction_encoding(
    raw_features, refined_features, meg_data, n_splits=3, alphas=None,
):
    """
    Encode using correction signal: correction[b] = refined[b] − raw[b].

    Also encodes raw[b] alone for comparison.  The key question: does the
    correction signal add brain-predictive variance beyond the raw features?

    Args:
        raw_features: dict {f"raw_level_{b}": (N_words, D)}
        refined_features: dict {f"refined_level_{b}": (N_words, D)}
        meg_data: (N_words, N_sensors, N_times)

    Returns:
        results dict with raw_r, correction_r, combined_r per level
    """
    if alphas is None:
        alphas = np.logspace(0, 5, 8)

    N_words, N_sensors, N_times = meg_data.shape
    Y_flat = meg_data.reshape(N_words, N_sensors * N_times)
    results = {}
    num_levels = len(raw_features)

    for b in range(num_levels):
        raw_key = f"raw_level_{b}"
        ref_key = f"refined_level_{b}"
        if raw_key not in raw_features or ref_key not in refined_features:
            continue

        raw_f = raw_features[raw_key]
        ref_f = refined_features[ref_key]
        correction = ref_f - raw_f  # (N_words, D) — the top-down update

        level_results = {}
        for feat_name, feat_data in [("raw", raw_f),
                                      ("correction", correction),
                                      ("combined", np.concatenate(
                                          [raw_f, correction], axis=1))]:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            pred = np.zeros_like(Y_flat)
            for train_idx, test_idx in kf.split(feat_data):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(feat_data[train_idx])
                X_te = scaler.transform(feat_data[test_idx])
                ridge = RidgeCV(alphas=alphas)
                ridge.fit(X_tr, Y_flat[train_idx])
                pred[test_idx] = ridge.predict(X_te)

            Y = Y_flat.reshape(N_words, N_sensors, N_times)
            P = pred.reshape(N_words, N_sensors, N_times)
            Y_c = Y - Y.mean(axis=0, keepdims=True)
            P_c = P - P.mean(axis=0, keepdims=True)
            num = np.sum(Y_c * P_c, axis=0)
            den = np.sqrt(np.sum(Y_c ** 2, axis=0) * np.sum(P_c ** 2, axis=0))
            corr = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            level_results[f"{feat_name}_mean_r"] = float(corr.mean())

        results[f"level_{b}"] = level_results
        logger.info(f"    level_{b}: raw={level_results['raw_mean_r']:.4f}  "
                     f"correction={level_results['correction_mean_r']:.4f}  "
                     f"combined={level_results['combined_mean_r']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Track C: Inhibition × Competition Interaction
# ---------------------------------------------------------------------------

def run_inhibition_competition(
    raw_feats_at_boundary, refined_feats_at_boundary,
    meg_data, epoch_times, word_events, density_map,
    n400_window=(0.3, 0.5), n_splits=3, alphas=None,
):
    """
    Test interaction: does post-inhibition encoding advantage increase
    for high-density (high competition) words in the N400 window?

    Splits words into HIGH / LOW density (median split), then computes
    linear encoding r for pre- and post-inhibition features in each group.

    Args:
        raw_feats_at_boundary: (N_words, D) — raw at inhibition boundary
        refined_feats_at_boundary: (N_words, D) — refined (post-inhibition)
        meg_data: (N_words, N_sensors, N_times)
        epoch_times: (N_times,)
        word_events: list of dicts with 'word' key
        density_map: {word: das_nb_ph}
        n400_window: (t_start, t_end) in seconds

    Returns:
        results dict
    """
    if alphas is None:
        alphas = np.logspace(0, 5, 8)

    # Look up density for each word
    densities = []
    for ev in word_events:
        w = ev["word"].strip().lower()
        densities.append(density_map.get(w, None))

    n_found = sum(1 for d in densities if d is not None)
    n_total = len(densities)
    logger.info(f"    Density lookup: {n_found}/{n_total} words matched")

    if n_found < 40:
        return {"error": f"Too few words with density ({n_found})"}

    # Replace missing with median, then median-split
    known = [d for d in densities if d is not None]
    median_density = float(np.median(known))
    densities_filled = [d if d is not None else median_density for d in densities]
    densities_arr = np.array(densities_filled)

    high_mask = densities_arr >= median_density
    low_mask = ~high_mask

    # Restrict MEG to N400 window
    t_mask = (epoch_times >= n400_window[0]) & (epoch_times <= n400_window[1])
    if t_mask.sum() == 0:
        return {"error": "N400 window outside epoch range"}

    meg_n400 = meg_data[:, :, t_mask]  # (N, sensors, n400_times)
    N_words, N_sensors, N_t = meg_n400.shape
    Y_flat = meg_n400.reshape(N_words, N_sensors * N_t)

    results = {}
    pre_feats = raw_feats_at_boundary
    post_feats = refined_feats_at_boundary

    for group_name, mask in [("high_density", high_mask),
                              ("low_density", low_mask)]:
        if mask.sum() < 20:
            results[group_name] = {"error": f"Too few words ({mask.sum()})"}
            continue

        group_Y = Y_flat[mask]
        group_pre = pre_feats[mask]
        group_post = post_feats[mask]

        for feat_name, feat_data in [("pre_inhibition", group_pre),
                                      ("post_inhibition", group_post)]:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            pred = np.zeros_like(group_Y)
            for train_idx, test_idx in kf.split(feat_data):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(feat_data[train_idx])
                X_te = scaler.transform(feat_data[test_idx])
                ridge = RidgeCV(alphas=alphas)
                ridge.fit(X_tr, group_Y[train_idx])
                pred[test_idx] = ridge.predict(X_te)

            Y_r = group_Y
            Y_c = Y_r - Y_r.mean(axis=0, keepdims=True)
            P_c = pred - pred.mean(axis=0, keepdims=True)
            num = np.sum(Y_c * P_c, axis=0)
            den = np.sqrt(np.sum(Y_c ** 2, axis=0) * np.sum(P_c ** 2, axis=0))
            r = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            results.setdefault(group_name, {})[feat_name + "_r"] = float(r.mean())

        delta = (results[group_name].get("post_inhibition_r", 0) -
                 results[group_name].get("pre_inhibition_r", 0))
        results[group_name]["delta_r"] = delta
        logger.info(f"    {group_name} (n={mask.sum()}): "
                     f"pre={results[group_name].get('pre_inhibition_r', 0):.4f}  "
                     f"post={results[group_name].get('post_inhibition_r', 0):.4f}  "
                     f"Δ={delta:+.4f}")

    # Interaction: high delta − low delta
    high_delta = results.get("high_density", {}).get("delta_r", 0)
    low_delta = results.get("low_density", {}).get("delta_r", 0)
    results["interaction"] = high_delta - low_delta
    results["median_density"] = median_density
    results["n400_window"] = list(n400_window)
    logger.info(f"    Interaction (high Δ − low Δ): {results['interaction']:+.4f}")

    return results


# ---------------------------------------------------------------------------
# Track D: CPC Prediction Error vs N400
# ---------------------------------------------------------------------------

def run_cpc_n400(
    cpc_errors_per_word, meg_data, epoch_times,
    n400_window=(0.3, 0.5),
):
    """
    Correlate per-word CPC prediction error with N400 amplitude.

    The N400 is the mean sensor amplitude in the 300–500ms window
    (negative = larger N400).

    Args:
        cpc_errors_per_word: dict {pair_name: (N_words,)} CPC error per word
        meg_data: (N_words, N_sensors, N_times)
        epoch_times: (N_times,)

    Returns:
        results: {pair_name: {rho, p, n_words}}
    """
    t_mask = (epoch_times >= n400_window[0]) & (epoch_times <= n400_window[1])
    if t_mask.sum() == 0:
        return {"error": "N400 window outside epoch range"}

    # N400 amplitude: mean across sensors and N400 time window, per word
    n400_amp = meg_data[:, :, t_mask].mean(axis=(1, 2))  # (N_words,)

    results = {}
    for pair_name, errors in cpc_errors_per_word.items():
        rho, p = stats.spearmanr(errors, n400_amp)
        results[pair_name] = {
            "rho": float(rho),
            "p": float(p),
            "n_words": int(len(errors)),
        }
        logger.info(f"    {pair_name}: ρ={rho:.4f}, p={p:.4f} (n={len(errors)})")

    return results


# ---------------------------------------------------------------------------
# Run processing for one MEG run
# ---------------------------------------------------------------------------

def process_one_run(
    model, processor, device, meg_dir,
    sub, ses, task_id,
    tmin=-0.2, tmax=0.8,
    max_words=0,
    bypass_novel=False,
    audio_feature_cache=None,
):
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

    # 3. Epoch around word onsets (for word-level tracks B, C, D)
    epochs, valid_events = epoch_meg_around_words(raw, word_events, tmin, tmax)

    # Also keep continuous MEG for Track A (frame-level encoding)
    meg_continuous = raw.get_data()  # (N_sensors, N_total_samples)
    raw_start_time = raw.times[0]   # should be 0
    del raw

    if epochs is None or len(valid_events) == 0:
        logger.warning(f"  No valid epochs for {sub_str}/{ses_str}/{task_id}")
        return None

    if max_words > 0 and len(valid_events) > max_words:
        valid_events = valid_events[:max_words]
        epochs = epochs[:max_words]

    meg_data = epochs.get_data()  # (N, N_sensors, N_times)
    epoch_times = epochs.times
    del epochs

    # 4. Extract features per audio file (with caching)
    if audio_feature_cache is None:
        audio_feature_cache = {}
    audio_to_feats = audio_feature_cache

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
            if not audio_path.exists():
                lower = Path(sound).name.lower()
                if lower in _audio_lower_map:
                    audio_path = _audio_lower_map[lower]
            if audio_path.exists():
                raw_f, ref_f, ftimes = extract_story_features(
                    model, processor, str(audio_path), device,
                    bypass_novel=bypass_novel,
                )
                # Also compute CPC errors if refinement is active
                cpc_err = None
                if ref_f is not None:
                    cpc_err = compute_cpc_errors(model, raw_f, ref_f, device)
                audio_to_feats[sound] = (raw_f, ref_f, ftimes, cpc_err)
            else:
                logger.warning(f"  Audio not found: {sound}")

    unique_sounds = set(ev["sound"] for ev in valid_events)
    found = sum(1 for s in unique_sounds if s in audio_to_feats)
    logger.info(f"  Audio: {found}/{len(unique_sounds)} files, "
                f"{len(valid_events)} word events")

    # 5. Word-level feature alignment
    num_levels = model.num_blocks
    has_refined = not bypass_novel
    raw_word_feats = [[] for _ in range(num_levels)]
    ref_word_feats = [[] for _ in range(num_levels)] if has_refined else None
    cpc_word_errors = [[] for _ in range(num_levels - 1)] if has_refined else None
    keep_mask = []

    for ev in valid_events:
        sound = ev["sound"]
        if sound not in audio_to_feats:
            keep_mask.append(False)
            continue

        raw_f, ref_f, ftimes, cpc_err = audio_to_feats[sound]
        onset = ev["start_in_audio"]
        dur = ev["duration"]

        frame_idxs = get_frames_for_word(ftimes, onset, dur)

        for b in range(num_levels):
            raw_word_feats[b].append(raw_f[b][frame_idxs].mean(axis=0))
            if has_refined:
                ref_word_feats[b].append(ref_f[b][frame_idxs].mean(axis=0))

        # CPC error per word: mean error across frames in word span
        if has_refined and cpc_err is not None:
            for pair in range(num_levels - 1):
                cpc_word_errors[pair].append(
                    float(cpc_err[pair][frame_idxs].mean())
                )

        keep_mask.append(True)

    if sum(keep_mask) < 20:
        logger.warning(f"  Too few aligned words ({sum(keep_mask)})")
        return None

    meg_data = meg_data[np.array(keep_mask)]
    aligned_events = [ev for ev, k in zip(valid_events, keep_mask) if k]

    raw_features = {}
    ref_features = {}
    for b in range(num_levels):
        raw_features[f"raw_level_{b}"] = np.stack(raw_word_feats[b])
        if has_refined:
            ref_features[f"refined_level_{b}"] = np.stack(ref_word_feats[b])

    cpc_errors_dict = None
    if has_refined and cpc_word_errors is not None:
        cpc_errors_dict = {}
        for pair in range(num_levels - 1):
            cpc_errors_dict[f"r{pair+1}_predicts_o{pair}"] = np.array(
                cpc_word_errors[pair]
            )

    # Build frame-level features per story for Track A
    # Pick the first (or only) story's features for continuous alignment
    frame_level = {}
    first_sound = aligned_events[0]["sound"]
    if first_sound in audio_to_feats:
        raw_f, ref_f, ftimes, _ = audio_to_feats[first_sound]
        for b in range(num_levels):
            frame_level[f"raw_level_{b}"] = raw_f[b]
            if has_refined:
                frame_level[f"refined_level_{b}"] = ref_f[b]
        frame_level["frame_times"] = ftimes

    return {
        "meg_data": meg_data,
        "epoch_times": epoch_times,
        "raw_features": raw_features,
        "refined_features": ref_features,
        "cpc_errors": cpc_errors_dict,
        "word_events": aligned_events,
        "meg_continuous": meg_continuous,
        "meg_sfreq": sfreq,
        "meg_start_time": raw_start_time,
        "frame_level": frame_level,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MEG-MASC Encoding v2 (frame-level + correction + inhibition×density)"
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--pretrained", type=str,
                   default="facebook/hubert-large-ls960-ft")
    p.add_argument("--meg_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="results/meg_v2")
    p.add_argument("--density_csv", type=str, default=None,
                   help="FastLex output CSV with das_nb_ph column")
    p.add_argument("--subjects", nargs="+", type=int, default=None)
    p.add_argument("--sessions", nargs="+", type=int, default=None)
    p.add_argument("--max_words", type=int, default=0)
    p.add_argument("--tmin", type=float, default=-0.2)
    p.add_argument("--tmax", type=float, default=0.8)
    p.add_argument("--refine_num_heads", type=int, default=4)
    p.add_argument("--inhibition_rank", type=int, default=64)
    p.add_argument("--inhibition_boundary", type=int, default=2)
    p.add_argument("--skip_frame_encoding", action="store_true",
                   help="Skip Track A")
    p.add_argument("--skip_correction", action="store_true",
                   help="Skip Track B")
    p.add_argument("--skip_inhibition", action="store_true",
                   help="Skip Track C")
    p.add_argument("--skip_cpc_n400", action="store_true",
                   help="Skip Track D")
    p.add_argument("--bypass_novel", action="store_true")
    p.add_argument("--tasks", nargs="+", type=int, default=None,
                   help="Task indices to process (default: all)")
    p.add_argument("--log_level", type=str, default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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
        args.refine_num_heads = train_args.get("refine_num_heads",
                                                args.refine_num_heads)
        args.inhibition_rank = train_args.get("inhibition_rank",
                                               args.inhibition_rank)
        args.inhibition_boundary = train_args.get("inhibition_boundary",
                                                   args.inhibition_boundary)

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

    # Density map for Track C
    density_map = None
    if args.density_csv:
        density_map = load_density_map(args.density_csv)

    # Subjects / sessions
    if args.subjects is None:
        args.subjects = list(range(1, 28))
    if args.sessions is None:
        args.sessions = [0, 1]

    # =================================================================
    # Main loop
    # =================================================================
    all_frame_enc = []
    all_correction = []
    all_inhibition = []
    all_cpc_n400 = []
    audio_feature_cache = {}

    for sub in args.subjects:
        for ses in args.sessions:
            sub_str = f"sub-{sub:02d}"
            ses_str = f"ses-{ses}"
            meg_path = (Path(args.meg_dir) / sub_str / ses_str / "meg")

            if not meg_path.exists():
                logger.info(f"Skipping {sub_str}/{ses_str} (not found)")
                continue

            con_files = sorted(meg_path.glob("*_meg.con"))
            for con_file in con_files:
                task_id = [p for p in con_file.stem.split("_")
                           if p.startswith("task-")][0]

                # Filter tasks if specified
                if args.tasks is not None:
                    task_num = int(task_id.split("-")[1])
                    if task_num not in args.tasks:
                        continue

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
                raw_feats = result["raw_features"]
                ref_feats = result["refined_features"]
                events = result["word_events"]
                N_words = meg_data.shape[0]
                run_id = f"{sub_str}_{ses_str}_{task_id}"

                logger.info(f"  {N_words} words, {meg_data.shape[1]} sensors, "
                             f"{meg_data.shape[2]} time points")

                # Track A: Frame-level temporal encoding
                if (not args.skip_frame_encoding
                        and result["frame_level"]
                        and "frame_times" in result["frame_level"]):
                    logger.info(f"  Track A: Frame-level Temporal Encoding")
                    fl = result["frame_level"]
                    frame_feats = {k: v for k, v in fl.items()
                                   if k != "frame_times"}
                    enc = run_frame_encoding(
                        frame_feats,
                        result["meg_continuous"],
                        result["meg_sfreq"],
                        fl["frame_times"],
                        result["meg_start_time"],
                    )
                    enc["run_id"] = run_id
                    all_frame_enc.append(enc)

                # Track B: Correction-signal encoding
                if not args.skip_correction and ref_feats:
                    logger.info(f"  Track B: Correction-Signal Encoding")
                    corr = run_correction_encoding(
                        raw_feats, ref_feats, meg_data,
                    )
                    corr["run_id"] = run_id
                    all_correction.append(corr)

                # Track C: Inhibition × competition
                b = model.inhibition_boundary
                if (not args.skip_inhibition and density_map
                        and ref_feats
                        and f"raw_level_{b}" in raw_feats
                        and f"refined_level_{b}" in ref_feats):
                    logger.info(f"  Track C: Inhibition × Competition")
                    inhib = run_inhibition_competition(
                        raw_feats[f"raw_level_{b}"],
                        ref_feats[f"refined_level_{b}"],
                        meg_data, epoch_times, events, density_map,
                    )
                    inhib["run_id"] = run_id
                    all_inhibition.append(inhib)

                # Track D: CPC prediction error vs N400
                if (not args.skip_cpc_n400
                        and result["cpc_errors"] is not None):
                    logger.info(f"  Track D: CPC Error vs N400")
                    cpc_res = run_cpc_n400(
                        result["cpc_errors"], meg_data, epoch_times,
                    )
                    cpc_res["run_id"] = run_id
                    all_cpc_n400.append(cpc_res)

                del meg_data, raw_feats, ref_feats
                del result["meg_continuous"]

    # =================================================================
    # Aggregate
    # =================================================================
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)
    aggregate = {}

    # Track A: peak lag per level
    if all_frame_enc:
        logger.info("\nTrack A — Frame-level Encoding (peak lags):")
        level_lags = {}
        for enc in all_frame_enc:
            for key, lag_dict in enc.items():
                if key == "run_id" or not isinstance(lag_dict, dict):
                    continue
                peak_lag = max(lag_dict, key=lag_dict.get)
                peak_r = lag_dict[peak_lag]
                level_lags.setdefault(key, {"peak_lags": [], "peak_rs": []})
                level_lags[key]["peak_lags"].append(peak_lag)
                level_lags[key]["peak_rs"].append(peak_r)

        enc_summary = {}
        for name, data in sorted(level_lags.items()):
            mean_lag = float(np.mean(data["peak_lags"]))
            mean_r = float(np.mean(data["peak_rs"]))
            enc_summary[name] = {"mean_peak_lag_ms": mean_lag,
                                  "mean_peak_r": mean_r}
            logger.info(f"  {name}: peak lag={mean_lag:.0f}ms, r={mean_r:.4f}")
        aggregate["frame_encoding"] = enc_summary

    # Track B: correction signal
    if all_correction:
        logger.info("\nTrack B — Correction-Signal Encoding:")
        corr_summary = {}
        for res in all_correction:
            for key, vals in res.items():
                if key == "run_id" or not isinstance(vals, dict):
                    continue
                corr_summary.setdefault(key, {"raw": [], "correction": [],
                                               "combined": []})
                for k2 in ("raw_mean_r", "correction_mean_r",
                            "combined_mean_r"):
                    short = k2.replace("_mean_r", "")
                    if k2 in vals:
                        corr_summary[key][short].append(vals[k2])

        for name, data in sorted(corr_summary.items()):
            means = {k: float(np.mean(v)) for k, v in data.items() if v}
            corr_summary[name] = means
            logger.info(f"  {name}: raw={means.get('raw', 0):.4f}, "
                         f"correction={means.get('correction', 0):.4f}, "
                         f"combined={means.get('combined', 0):.4f}")
        aggregate["correction_encoding"] = corr_summary

    # Track C: inhibition × competition
    if all_inhibition:
        logger.info("\nTrack C — Inhibition × Competition:")
        high_deltas = []
        low_deltas = []
        interactions = []
        for res in all_inhibition:
            if "error" in res:
                continue
            hd = res.get("high_density", {})
            ld = res.get("low_density", {})
            if "delta_r" in hd:
                high_deltas.append(hd["delta_r"])
            if "delta_r" in ld:
                low_deltas.append(ld["delta_r"])
            if "interaction" in res:
                interactions.append(res["interaction"])

        inhib_summary = {}
        if high_deltas:
            inhib_summary["high_density_mean_delta"] = float(
                np.mean(high_deltas))
        if low_deltas:
            inhib_summary["low_density_mean_delta"] = float(
                np.mean(low_deltas))
        if interactions:
            inhib_summary["mean_interaction"] = float(np.mean(interactions))
            if len(interactions) > 1:
                t, p = stats.ttest_1samp(interactions, 0)
                inhib_summary["interaction_t"] = float(t)
                inhib_summary["interaction_p"] = float(p)
                logger.info(
                    f"  Interaction: {np.mean(interactions):+.4f} "
                    f"(t={t:.2f}, p={p:.4f})")
            else:
                logger.info(
                    f"  Interaction: {np.mean(interactions):+.4f} (1 run)")
        aggregate["inhibition_competition"] = inhib_summary

    # Track D: CPC vs N400
    if all_cpc_n400:
        logger.info("\nTrack D — CPC Prediction Error vs N400:")
        pair_rhos = {}
        for res in all_cpc_n400:
            for key, vals in res.items():
                if key == "run_id" or not isinstance(vals, dict):
                    continue
                pair_rhos.setdefault(key, []).append(vals.get("rho", 0))

        cpc_summary = {}
        for pair, rhos in sorted(pair_rhos.items()):
            mean_rho = float(np.mean(rhos))
            cpc_summary[pair] = {"mean_rho": mean_rho, "n_runs": len(rhos)}
            if len(rhos) > 1:
                t, p = stats.ttest_1samp(rhos, 0)
                cpc_summary[pair]["t"] = float(t)
                cpc_summary[pair]["p"] = float(p)
                logger.info(f"  {pair}: ρ={mean_rho:.4f} (t={t:.2f}, p={p:.4f})")
            else:
                logger.info(f"  {pair}: ρ={mean_rho:.4f}")
        aggregate["cpc_n400"] = cpc_summary

    # Save
    output = {
        "args": vars(args),
        "aggregate": aggregate,
        "per_run_frame_encoding": all_frame_enc,
        "per_run_correction": all_correction,
        "per_run_inhibition": all_inhibition,
        "per_run_cpc_n400": all_cpc_n400,
    }

    output_path = Path(args.output_dir) / "meg_encoding_v2_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, 'tolist')
                  else str(x))
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
