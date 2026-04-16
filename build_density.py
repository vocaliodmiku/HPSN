#!/usr/bin/env python
"""
Build phonological neighborhood density lookup for MEG-MASC words.

Steps:
  1. Extract all unique words from MEG-MASC events.tsv files.
  2. Look up CMU pronunciations (ARPABET) for each word.
  3. Run FastLex to compute DAS neighbor counts (phonological).
  4. Save density CSV for use by meg_encoding_v2.py.

Usage:
    python build_density.py \
        --meg_dir /path/to/gwilliams2022/osfstorage \
        --fastlex_dir /path/to/fastlex \
        --output data/meg_word_density.csv

Prerequisites:
    pip install nltk pandas tqdm rapidfuzz
    git clone https://github.com/comp-cogneuro-lang/fastlex.git
"""

import argparse
import ast
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_meg_words(meg_dir):
    """Get all unique words from MEG-MASC events.tsv files."""
    words = set()
    meg_dir = Path(meg_dir)
    for events_file in sorted(meg_dir.rglob("*_events.tsv")):
        with open(events_file, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                trial = ast.literal_eval(row["trial_type"])
                if trial["kind"] == "word":
                    w = trial.get("word", "").strip()
                    if w:
                        words.add(w.lower())
    return sorted(words)


def get_cmu_pronunciations(words):
    """Look up ARPABET pronunciations from NLTK's CMU Pronouncing Dict."""
    import nltk
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict
    cmu = cmudict.dict()

    result = {}
    missing = []
    for w in words:
        if w in cmu:
            # Take first pronunciation, strip stress digits
            phones = cmu[w][0]
            arpabet = " ".join(phones)
            result[w] = arpabet
        else:
            missing.append(w)
    return result, missing


def build_fastlex_input(word_prons, output_path):
    """Write a CSV that FastLex can consume."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Word", "Pron_arpabet"])
        for word, pron in sorted(word_prons.items()):
            writer.writerow([word, pron])
    return len(word_prons)


def run_fastlex(fastlex_dir, input_csv, output_csv, n_jobs=8):
    """Run FastLex on the input CSV."""
    fastlex_py = Path(fastlex_dir) / "fastlex.py"
    if not fastlex_py.exists():
        raise FileNotFoundError(f"fastlex.py not found at {fastlex_py}")

    cmd = [
        sys.executable, str(fastlex_py),
        "--lexicon-path", str(input_csv),
        "--output-path", str(output_csv),
        "--orth-col", "Word",
        "--pron-col", "Pron_arpabet",
        "--delimiter-phono", "space",
        "--delimiter-ortho", "none",
        "--n-jobs", str(n_jobs),
        "--progress",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meg_dir", type=str, required=True)
    p.add_argument("--fastlex_dir", type=str, required=True,
                   help="Path to cloned fastlex repo")
    p.add_argument("--output", type=str, default="data/meg_word_density.csv")
    p.add_argument("--n_jobs", type=int, default=8)
    args = p.parse_args()

    # 1. Extract words
    print("Extracting words from MEG-MASC events...")
    words = extract_meg_words(args.meg_dir)
    print(f"  Found {len(words)} unique words")

    # 2. Get pronunciations
    print("Looking up CMU pronunciations...")
    prons, missing = get_cmu_pronunciations(words)
    print(f"  Found pronunciations for {len(prons)}/{len(words)} words")
    if missing:
        print(f"  Missing: {len(missing)} words (first 20: {missing[:20]})")

    # 3. Write FastLex input
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tmp_input = Path(args.output).with_suffix(".input.csv")
    n = build_fastlex_input(prons, tmp_input)
    print(f"  Wrote {n} entries to {tmp_input}")

    # 4. Run FastLex
    print("Running FastLex...")
    run_fastlex(args.fastlex_dir, tmp_input, args.output, args.n_jobs)
    print(f"  Output saved to {args.output}")

    # Clean up temp
    tmp_input.unlink(missing_ok=True)
    print("Done!")


if __name__ == "__main__":
    main()
