"""
services/essentia_similarity.py
Mood and genre feature extraction via Essentia (#74).

Essentia is CPU-only (no @spaces.GPU needed) — a meaningful advantage on the
ZeroGPU free tier where GPU quota is limited.

Design:
- EssentiaSimilarity.extract_features() soft-imports essentia at call time so
  an ImportError (essentia not installed) is caught and returns empty features
  rather than crashing the pipeline.
- All heavy work is isolated to extract_features(); _extract_mood_tags() and
  _extract_genre_tags() are pure helpers for independent unit testing.
- No disk state is kept between calls — the temp wav file is written to a
  TemporaryDirectory and cleaned up in a try/finally block.
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from core.config import CONSTANTS
from core.exceptions import AudioSourceError

_log = logging.getLogger(__name__)


class EssentiaSimilarity:
    """
    Extract mood and genre tags from audio using Essentia's MusicExtractor.

    CPU-only — no GPU decorator required.

    Usage:
        svc = EssentiaSimilarity()
        features = svc.extract_features(audio_bytes)
        mood_tags = features.get("mood_tags", [])
    """

    def extract_features(self, raw: bytes) -> dict[str, Any]:
        """
        Run Essentia MusicExtractor on raw audio bytes and return mood/genre tags.

        Writes a temporary WAV file (required by MusicExtractor's file-path API),
        runs extraction, then cleans up.  The temp directory is always removed in
        the finally block regardless of success or failure.

        Args:
            raw: Raw audio bytes (any format librosa can decode — mp3, wav, etc.)

        Returns:
            Dict with keys:
              mood_tags  — list[str], up to ESSENTIA_MAX_MOOD_TAGS mood labels
              genre_tags — list[str], up to ESSENTIA_MAX_MOOD_TAGS genre labels
              bpm        — float, estimated tempo (0.0 if unavailable)
              key        — str, estimated key (e.g. "C major"; "" if unavailable)
            Returns an all-empty dict on any error so callers can skip gracefully.

        Raises:
            AudioSourceError: if raw is empty or not decodable by librosa.
        """
        if not raw:
            raise AudioSourceError(
                "EssentiaSimilarity.extract_features received empty audio bytes.",
                context={"bytes_length": 0},
            )

        try:
            import essentia                              # noqa: PLC0415
            import essentia.standard as es              # noqa: PLC0415
        except ImportError:
            _log.debug("essentia not installed — mood feature extraction skipped")
            return {}

        try:
            import librosa                              # noqa: PLC0415
            import soundfile as sf                      # noqa: PLC0415
        except ImportError as exc:
            _log.debug("librosa/soundfile not available: %s", exc)
            return {}

        try:
            # Decode to float32 mono at 44100 Hz (MusicExtractor's expected rate)
            y, _ = librosa.load(io.BytesIO(raw), sr=44100, mono=True)
        except Exception as exc:
            raise AudioSourceError(
                "EssentiaSimilarity: could not decode audio bytes.",
                context={"original_error": str(exc)},
            ) from exc

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = str(Path(tmp_dir) / "track.wav")
            try:
                sf.write(wav_path, y.astype(np.float32), 44100, subtype="PCM_16")
                extractor = es.MusicExtractor(
                    lowlevelSilenceRate60dBThreshold=-60,
                    lowlevelSilenceRate90dBThreshold=-90,
                )
                features, _ = extractor(wav_path)
            except Exception as exc:
                _log.debug("Essentia MusicExtractor failed: %s", exc)
                return {}

        mood_tags  = _extract_mood_tags(features, CONSTANTS.ESSENTIA_MAX_MOOD_TAGS)
        genre_tags = _extract_genre_tags(features, CONSTANTS.ESSENTIA_MAX_MOOD_TAGS)

        try:
            bpm = float(features["rhythm.bpm"])
        except (KeyError, TypeError, ValueError):
            bpm = 0.0

        try:
            key  = str(features["tonal.key_edma.key"])
            mode = str(features["tonal.key_edma.scale"])
            key  = f"{key} {mode}" if mode else key
        except (KeyError, TypeError):
            key = ""

        return {
            "mood_tags":  mood_tags,
            "genre_tags": genre_tags,
            "bpm":        round(bpm, 1),
            "key":        key,
        }


# ---------------------------------------------------------------------------
# Pure helpers — independently testable
# ---------------------------------------------------------------------------

def _extract_highlevel_tags(features: Any, prefix: str, max_tags: int) -> list[str]:
    """
    Pull top-N winning labels from Essentia highlevel features for a given prefix.

    Essentia stores classifier outputs as:
      highlevel.<prefix>_<classifier>.value       — winning class label (str)
      highlevel.<prefix>_<classifier>.probability — confidence of winning class (float)

    Collects all matching .value keys, sorts by probability descending, deduplicates
    labels (multiple classifiers can emit the same string), and returns up to max_tags.

    Uses a set for O(1) pool key membership checks — avoids O(n²) list scans on
    MusicExtractor pools that may contain hundreds of descriptors.

    Pure function — no I/O.
    """
    scores: list[tuple[str, float]] = []
    try:
        pool_key_set: set[str] = set(features.descriptorNames())
    except Exception:  # noqa: BLE001
        return []

    full_prefix = f"highlevel.{prefix}_"
    for key in pool_key_set:
        if not (key.startswith(full_prefix) and key.endswith(".value")):
            continue
        try:
            label    = str(features[key])
            prob_key = key.replace(".value", ".probability")
            prob     = float(features[prob_key]) if prob_key in pool_key_set else 0.5
            scores.append((label, prob))
        except Exception:  # noqa: BLE001
            continue

    scores.sort(key=lambda x: x[1], reverse=True)
    seen: set[str] = set()
    result: list[str] = []
    for label, _ in scores:
        if label not in seen:
            seen.add(label)
            result.append(label)
        if len(result) >= max_tags:
            break
    return result


def _extract_mood_tags(features: Any, max_tags: int) -> list[str]:
    """Return top-N mood labels from Essentia highlevel features. Pure function — no I/O."""
    return _extract_highlevel_tags(features, "mood", max_tags)


def _extract_genre_tags(features: Any, max_tags: int) -> list[str]:
    """Return top-N genre labels from Essentia highlevel features. Pure function — no I/O."""
    return _extract_highlevel_tags(features, "genre", max_tags)
