"""
tools/score_wavs.py

Score a directory of WAV files for all 6 forensic signals and write
per-track feature vectors to a CSV for use in train_classifier.py.

Works with any WAV dataset — not tied to FakeMusicCaps. Pass --data-dir,
--output-csv, and --label to point it at any collection of audio files.

For datasets organised into subdirectories (e.g. one folder per generator),
each subdirectory is treated as a named group and its name is written to the
'group' column. For a flat directory, group defaults to the --group-name value
(or "default" if omitted).

Usage:
  # Score FakeMusicCaps AI WAVs (subdirs = generator names)
  .venv/bin/python tools/score_wavs.py \\
      --data-dir "/Volumes/Sound Library/FakeMusicCaps" \\
      --output-csv local/fakemusiccaps_features.csv \\
      --label 1 --n-per-group 300

  # Score a flat folder of human reference WAVs
  .venv/bin/python tools/score_wavs.py \\
      --data-dir /path/to/human_wavs \\
      --output-csv local/human_features.csv \\
      --label 0 --group-name human_reference
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.stats import gmean

# ---------------------------------------------------------------------------
# Signal params — must match CONSTANTS in core/config.py
# ---------------------------------------------------------------------------

BASE_SR = 22_050
RANDOM_SEED = 42

CENTROID_TOP_DB         = 30.0
CENTROID_MIN_INTERVAL_S = 0.5
PLR_WINDOW_SECONDS      = 2
PLR_MIN_WINDOWS         = 5
VOICED_NOISE_FLOOR_HZ_LOW     = 4_000
VOICED_NOISE_FLOOR_HZ_HIGH    = 12_000
VOICED_NOISE_FLOOR_HOP_LENGTH = 512
VOICED_NOISE_FLOOR_N_FFT      = 2048
VOCAL_MIN_VOICED_FRAMES       = 50   # relaxed for short clips (≤ 10 s)
VOCAL_PYIN_FMIN = 65.41
VOCAL_PYIN_FMAX = 2093.0

FEATURES = ["group", "label", "ci", "hnr", "plr", "ibi", "cent", "vnf"]


# ---------------------------------------------------------------------------
# Signal scorers
# ---------------------------------------------------------------------------

# Sentinel value returned when a signal cannot be computed for a given track.
# librosa/numpy failures caught below are: ValueError (bad array shape or params),
# RuntimeError (algorithmic failure), and LinAlgError (degenerate matrix in HPSS/STFT).
_SCORER_ERRORS = (ValueError, RuntimeError, np.linalg.LinAlgError)


def score_centroid_instability(audio: np.ndarray, sr: int) -> float:
    try:
        intervals   = librosa.effects.split(audio, top_db=CENTROID_TOP_DB)
        min_samples = int(CENTROID_MIN_INTERVAL_S * sr)
        cvs: list[float] = []
        for start, end in intervals:
            if (end - start) < min_samples:
                continue
            segment  = audio[start:end]
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
            mean_c   = float(np.mean(centroid))
            if mean_c < 1e-9:
                continue
            cvs.append(float(np.std(centroid) / mean_c))
        return float(np.clip(np.mean(cvs), 0.0, 1.0)) if cvs else -1.0
    except _SCORER_ERRORS:
        return -1.0


def score_harmonic_ratio(audio: np.ndarray, sr: int) -> float:
    try:
        intervals   = librosa.effects.split(audio, top_db=CENTROID_TOP_DB)
        min_samples = int(CENTROID_MIN_INTERVAL_S * sr)
        ratios: list[float] = []
        for start, end in intervals:
            if (end - start) < min_samples:
                continue
            segment      = audio[start:end]
            total_energy = float(np.mean(segment ** 2))
            if total_energy < 1e-9:
                continue
            y_harmonic, _ = librosa.effects.hpss(segment)
            ratios.append(float(np.mean(y_harmonic ** 2)) / total_energy)
        return float(np.clip(np.mean(ratios), 0.0, 1.0)) if ratios else -1.0
    except _SCORER_ERRORS:
        return -1.0


def score_plr(audio: np.ndarray, sr: int) -> float:
    try:
        window_samples = int(PLR_WINDOW_SECONDS * sr)
        n_complete     = len(audio) // window_samples
        if n_complete < PLR_MIN_WINDOWS:
            return -1.0
        plr_values: list[float] = []
        for i in range(n_complete):
            window = audio[i * window_samples : (i + 1) * window_samples]
            peak   = float(np.max(np.abs(window)))
            rms    = float(np.sqrt(np.mean(window ** 2)))
            if rms < 1e-9:
                continue
            plr_values.append(20.0 * np.log10(peak + 1e-9) - 20.0 * np.log10(rms + 1e-9))
        return float(np.std(plr_values)) if len(plr_values) >= PLR_MIN_WINDOWS else -1.0
    except _SCORER_ERRORS:
        return -1.0


def score_ibi_variance(audio: np.ndarray, sr: int) -> float:
    try:
        _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        beat_times_ms  = librosa.frames_to_time(beat_frames, sr=sr) * 1000.0
        return float(np.var(np.diff(beat_times_ms))) if len(beat_times_ms) >= 2 else -1.0
    except _SCORER_ERRORS:
        return -1.0


def score_centroid_mean(audio: np.ndarray, sr: int) -> float:
    try:
        if len(audio) < sr:
            return -1.0
        return float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0]))
    except _SCORER_ERRORS:
        return -1.0


def score_voiced_noise_floor(audio: np.ndarray, sr: int) -> float:
    try:
        _, voiced_flag, _ = librosa.pyin(
            audio, fmin=VOCAL_PYIN_FMIN, fmax=VOCAL_PYIN_FMAX,
            sr=sr, hop_length=VOICED_NOISE_FLOOR_HOP_LENGTH,
        )
        if voiced_flag is None or int(np.sum(voiced_flag)) < VOCAL_MIN_VOICED_FRAMES:
            return -1.0
        freqs     = librosa.fft_frequencies(sr=sr, n_fft=VOICED_NOISE_FLOOR_N_FFT)
        band_mask = (freqs >= VOICED_NOISE_FLOOR_HZ_LOW) & (freqs <= VOICED_NOISE_FLOOR_HZ_HIGH)
        if not np.any(band_mask):
            return -1.0
        stft_mag = np.abs(librosa.stft(audio, n_fft=VOICED_NOISE_FLOOR_N_FFT,
                                        hop_length=VOICED_NOISE_FLOOR_HOP_LENGTH))
        n_frames = min(stft_mag.shape[1], len(voiced_flag))
        vals: list[float] = []
        for t in range(n_frames):
            if not voiced_flag[t]:
                continue
            band  = stft_mag[band_mask, t] + 1e-10
            geo   = float(gmean(band))
            arith = float(np.mean(band))
            vals.append(geo / arith if arith > 0 else 0.0)
        return float(np.mean(vals)) if len(vals) >= VOCAL_MIN_VOICED_FRAMES else -1.0
    except _SCORER_ERRORS:
        return -1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score WAV files for forensic signals and write to CSV."
    )
    parser.add_argument("--data-dir",    required=True,  type=Path,
                        help="Root directory of WAV files. Subdirectories are treated as groups.")
    parser.add_argument("--output-csv",  required=True,  type=Path,
                        help="Destination CSV path.")
    parser.add_argument("--label",       required=True,  type=int, choices=[0, 1],
                        help="Ground-truth label: 1=AI, 0=human.")
    parser.add_argument("--n-per-group", type=int, default=0,
                        help="Max WAVs to sample per group (0 = all).")
    parser.add_argument("--group-name",  type=str, default="default",
                        help="Group name for flat directories with no subdirectories.")
    return parser.parse_args()


def _collect_groups(data_dir: Path, group_name: str) -> dict[str, list[Path]]:
    """Return {group_name: [wav_paths]}. Subdirs → one group each; flat dir → single group."""
    subdirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    if subdirs:
        return {d.name: sorted(d.glob("*.wav")) for d in subdirs}
    return {group_name: sorted(data_dir.glob("*.wav"))}


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger(__name__)

    if not args.data_dir.exists():
        log.error(f"--data-dir not found: {args.data_dir}"); sys.exit(1)

    random.seed(RANDOM_SEED)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    groups = _collect_groups(args.data_dir, args.group_name)
    total  = sum(len(v) for v in groups.values())
    log.info(f"Scoring {args.data_dir.name} — {len(groups)} group(s), {total} WAVs total"
             f"{f', sampling {args.n_per_group} per group' if args.n_per_group else ''}")

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURES)
        writer.writeheader()

        for group, wavs in groups.items():
            sample = (random.sample(wavs, min(args.n_per_group, len(wavs)))
                      if args.n_per_group else wavs)
            log.info(f"  {group}: {len(sample)} WAVs")

            for i, p in enumerate(sample, 1):
                try:
                    audio, sr = librosa.load(str(p), sr=BASE_SR, mono=True)
                except Exception as exc:
                    log.warning(f"    [{i}] LOAD ERROR {p.name}: {exc}")
                    continue

                ci  = score_centroid_instability(audio, sr)
                hnr = score_harmonic_ratio(audio, sr)
                plr = score_plr(audio, sr)
                ibi = score_ibi_variance(audio, sr)
                cm  = score_centroid_mean(audio, sr)
                vnf = score_voiced_noise_floor(audio, sr)

                log.info(
                    f"    [{i}/{len(sample)}] {group}  "
                    f"ci={'%.3f'%ci  if ci >=0 else 'skip'}  "
                    f"hnr={'%.3f'%hnr if hnr>=0 else 'skip'}  "
                    f"plr={'%.2f'%plr if plr>=0 else 'skip'}  "
                    f"ibi={'%.0f'%ibi if ibi>=0 else 'skip'}  "
                    f"cent={'%.0f'%cm  if cm >=0 else 'skip'}  "
                    f"vnf={'%.4f'%vnf if vnf>=0 else 'skip'}  "
                    f"{p.name}"
                )

                writer.writerow({
                    "group": group, "label": args.label,
                    "ci": ci, "hnr": hnr, "plr": plr,
                    "ibi": ibi, "cent": cm, "vnf": vnf,
                })
                f.flush()

    log.info(f"Done — saved to {args.output_csv}")


if __name__ == "__main__":
    main()
