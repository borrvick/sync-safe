"""
tools/train_classifier.py

Train logistic regression classifiers on per-track signal features.

Data sources:
  AI (compressed/YouTube path):
    - SONICS calibration log  — 500 Suno + Udio MP3s
  AI (upload path — native WAV):
    - local/fakemusiccaps_features.csv  — FakeMusicCaps WAVs (5 generators)
  Human:
    - SONICS calibration log  — 27 human WAV masters (labeled separately)
    - iTunes task log         — 312 human vocal MP3s (path via --itunes-log)

Produces two models:
  youtube_model : features [ci, hnr, plr, ibi, cent] — no vnf (codec-unreliable)
  upload_model  : all 6 features including vnf        — file-upload path only

Output:
  models/youtube_classifier.pkl
  models/upload_classifier.pkl

Usage:
  # Run score_wavs.py first to generate local/fakemusiccaps_features.csv, then:
  .venv/bin/python tools/train_classifier.py --itunes-log local/itunes_calibration.log
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SONICS_LOG        = Path("/Volumes/Sound Library/SonicsCal/calibration.log")
FAKEMUSICCAPS_CSV = Path("local/fakemusiccaps_features.csv")

FEATURES          = ["ci", "hnr", "plr", "ibi", "cent", "vnf"]
YOUTUBE_FEATURES  = ["ci", "hnr", "plr", "ibi", "cent"]

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_FEAT_RE = re.compile(
    r"ci=(-?[\d.]+)\s+"
    r"hnr=(-?[\d.]+)\s+"
    r"plr=\s*(-?[\d.]+)\s+"
    r"ibi=\s*(-?[\d.]+)\s+"
    r"cent=\s*(-?[\d.]+)\s+"
    r"vnf=(-?[\d.]+)"
)
# Matches "[human]" at start of a log line
_HUMAN_GROUP_RE = re.compile(r"\[human\]")


def parse_sonics_log(path: Path) -> list[dict]:
    """
    Parse the SONICS calibration log, correctly labeling AI vs human groups.
    Lines containing '[human]' → label=0; all other feature lines → label=1 (AI).
    """
    records = []
    for line in path.read_text(errors="replace").splitlines():
        m = _FEAT_RE.search(line)
        if not m:
            continue
        ci, hnr, plr, ibi, cent, vnf = (float(x) for x in m.groups())
        if any(v < 0 for v in [ci, hnr, plr, ibi, cent]):
            continue
        label = 0 if _HUMAN_GROUP_RE.search(line) else 1
        records.append({
            "ci": ci, "hnr": hnr, "plr": plr, "ibi": ibi,
            "cent": cent, "vnf": vnf, "label": label,
        })
    return records


def parse_itunes_log(path: Path) -> list[dict]:
    """Parse the iTunes calibration task log — all records are human (label=0)."""
    records = []
    for line in path.read_text(errors="replace").splitlines():
        m = _FEAT_RE.search(line)
        if not m:
            continue
        ci, hnr, plr, ibi, cent, vnf = (float(x) for x in m.groups())
        if any(v < 0 for v in [ci, hnr, plr, ibi, cent]):
            continue
        records.append({
            "ci": ci, "hnr": hnr, "plr": plr, "ibi": ibi,
            "cent": cent, "vnf": vnf, "label": 0,
        })
    return records


def parse_fakemusiccaps_csv(path: Path) -> list[dict]:
    """Parse the FakeMusicCaps feature CSV — all records are AI (label=1)."""
    records = []
    with path.open() as f:
        for row in csv.DictReader(f):
            ci   = float(row["ci"])
            hnr  = float(row["hnr"])
            plr  = float(row["plr"])
            ibi  = float(row["ibi"])
            cent = float(row["cent"])
            vnf  = float(row["vnf"])
            if any(v < 0 for v in [ci, hnr, plr, ibi, cent]):
                continue
            records.append({
                "ci": ci, "hnr": hnr, "plr": plr, "ibi": ibi,
                "cent": cent, "vnf": vnf, "label": 1,
            })
    return records


def build_xy(
    records: list[dict],
    feature_names: list[str],
    require_vnf: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for r in records:
        if require_vnf and r["vnf"] < 0:
            continue
        rows.append([r[f] for f in feature_names] + [r["label"]])
    arr = np.array(rows)
    return arr[:, :-1], arr[:, -1].astype(int)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_report(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_name: str,
    out_path: Path,
) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score, f1_score
    import joblib

    n_ai    = int((y == 1).sum())
    n_human = int((y == 0).sum())
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"  Features : {feature_names}")
    print(f"  AI       : {n_ai}  |  Human: {n_human}  |  Total: {len(y)}")
    print(f"{'='*60}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"  CV AUC : {aucs.mean():.3f} ± {aucs.std():.3f}  {aucs.round(3).tolist()}")

    pipe.fit(X, y)
    y_prob = pipe.predict_proba(X)[:, 1]
    print(f"  In-sample AUC : {roc_auc_score(y, y_prob):.3f}")

    # Find threshold that maximises F1 for AI class
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.01):
        f1 = f1_score(y, (y_prob >= t).astype(int), pos_label=1)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    y_best = (y_prob >= best_t).astype(int)
    print(f"\n  Best threshold: {best_t:.2f}  (AI F1={best_f1:.3f})")
    print(classification_report(y, y_best, target_names=["human", "AI"]))

    # Feature importances
    clf = pipe.named_steps["clf"]
    print("  Feature importances (scaled logistic coefs):")
    for name, coef in sorted(zip(feature_names, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:6s}  {coef:+.4f}")

    out_path.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": best_t, "features": feature_names}, out_path)
    print(f"\n  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train logistic regression classifiers on forensic signal features."
    )
    parser.add_argument(
        "--itunes-log",
        required=True,
        type=Path,
        help=(
            "Path to the iTunes calibration log (all-human records). "
            "Copy the raw task output to local/ so it persists across sessions, e.g. "
            "local/itunes_calibration.log"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("Loading data sources...")

    if not SONICS_LOG.exists():
        print(f"ERROR: {SONICS_LOG} not found"); sys.exit(1)
    sonics_records = parse_sonics_log(SONICS_LOG)
    ai_mp3    = [r for r in sonics_records if r["label"] == 1]
    human_wav = [r for r in sonics_records if r["label"] == 0]
    print(f"  SONICS AI MP3       : {len(ai_mp3)}")
    print(f"  SONICS human WAV    : {len(human_wav)}")

    fmc_records: list[dict] = []
    if FAKEMUSICCAPS_CSV.exists():
        fmc_records = parse_fakemusiccaps_csv(FAKEMUSICCAPS_CSV)
        print(f"  FakeMusicCaps WAV   : {len(fmc_records)} AI tracks")
    else:
        print("  FakeMusicCaps CSV not found — run score_wavs.py first")
        print("  Continuing without native AI WAV data...")

    if not args.itunes_log.exists():
        print(f"ERROR: iTunes log not found: {args.itunes_log}"); sys.exit(1)
    human_mp3 = parse_itunes_log(args.itunes_log)
    print(f"  iTunes human MP3    : {len(human_mp3)}")

    all_records = ai_mp3 + fmc_records + human_wav + human_mp3

    # YouTube model: MP3 AI (Suno/Udio) vs all human
    # FakeMusicCaps excluded — those are WAV and would skew the MP3-calibrated model
    youtube_records = ai_mp3 + human_wav + human_mp3
    X_yt, y_yt = build_xy(youtube_records, YOUTUBE_FEATURES)
    train_and_report(X_yt, y_yt, YOUTUBE_FEATURES, "youtube",
                     Path("models/youtube_classifier.pkl"))

    # Upload model: all AI (MP3 + WAV) vs all human, vnf required
    X_up, y_up = build_xy(all_records, FEATURES, require_vnf=True)
    train_and_report(X_up, y_up, FEATURES, "upload",
                     Path("models/upload_classifier.pkl"))


if __name__ == "__main__":
    main()
