"""
services/forensics/detectors/metadata.py
MetadataAnalyzer — C2PA manifest check + SynthID-style watermark scan.

Both signals operate at the file/header level:
  - C2PA:   pure byte parsing, no audio decoding needed.
  - SynthID: requires 44.1 kHz STFT — loads from raw bytes at that SR.
"""
from __future__ import annotations

import io
import json

import numpy as np
from scipy.ndimage import uniform_filter1d

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError

from .._audio import _AudioData
from .._bundle import _SignalBundle
from .._pure import (
    _SYNTHID_HOP,
    _SYNTHID_LOAD_SR,
    _SYNTHID_MAG_SPIKE_DB,
    _SYNTHID_N_FFT,
    _SYNTHID_PHASE_STD_MAX,
    _classify_c2pa_origin,
)


class MetadataAnalyzer:
    """
    Checks C2PA Content Credentials and SynthID-style phase watermarks.

    C2PA degrades gracefully — a missing manifest is normal.
    SynthID requires a 44.1 kHz load, so it reads from _AudioData.raw
    rather than the pre-decoded standard-SR array.
    """

    def process(self, data: _AudioData, bundle: _SignalBundle) -> None:
        bundle.c2pa_flag, bundle.c2pa_label = self._check_c2pa(data.raw)
        bundle.synthid_bins                  = self._check_synthid(data.raw)

    # ------------------------------------------------------------------
    # C2PA
    # ------------------------------------------------------------------

    def _check_c2pa(self, raw: bytes) -> tuple[bool, str]:
        """
        Read C2PA Content Credentials from raw bytes.

        Returns (born_ai, origin).  Gracefully returns (False, "") on any
        error — a missing manifest is normal, not an exception.
        """
        try:
            import c2pa
        except ImportError:
            return False, ""

        try:
            reader = c2pa.Reader.try_create("audio/mpeg", io.BytesIO(raw))
            if reader is None:
                return False, ""

            data = reader.json()
            if isinstance(data, str):
                data = json.loads(data)

            return _classify_c2pa_origin(data)

        except (OSError, ValueError, RuntimeError) as exc:
            err = str(exc).lower()
            if any(k in err for k in ("no active manifest", "not found", "jumbf")):
                return False, ""
            return False, ""

    # ------------------------------------------------------------------
    # SynthID
    # ------------------------------------------------------------------

    def _check_synthid(self, raw: bytes) -> int:
        """
        Scan the 18–22 kHz band for phase-coherent bins (AI watermark).
        Loads at 44.1 kHz so Nyquist = 22.05 kHz.

        Returns:
            Number of coherent bins found (0 = no signal).

        Raises:
            ModelInferenceError: on audio load or STFT failure.
        """
        try:
            import librosa

            audio, sr = librosa.load(
                io.BytesIO(raw), sr=_SYNTHID_LOAD_SR, mono=True
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "SynthID scan: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        if len(audio) < _SYNTHID_LOAD_SR:
            return 0

        try:
            stft  = librosa.stft(audio, n_fft=_SYNTHID_N_FFT, hop_length=_SYNTHID_HOP)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=_SYNTHID_N_FFT)

            hf_mask = (freqs >= CONSTANTS.SYNTHID_BAND_LOW_HZ) & (freqs <= CONSTANTS.SYNTHID_BAND_HIGH_HZ)
            if not np.any(hf_mask):
                return 0

            hf_stft  = stft[hf_mask]
            hf_phase = np.angle(hf_stft)
            hf_mag   = np.abs(hf_stft)

            cos_mean  = np.mean(np.cos(hf_phase), axis=1)
            sin_mean  = np.mean(np.sin(hf_phase), axis=1)
            R         = np.clip(np.sqrt(cos_mean ** 2 + sin_mean ** 2), 1e-9, 1.0 - 1e-9)
            phase_std = np.sqrt(-2.0 * np.log(R))

            mean_mag_db    = 20.0 * np.log10(np.mean(hf_mag, axis=1) + 1e-9)
            local_floor_db = uniform_filter1d(mean_mag_db, size=50, mode="nearest")

            coherent = (phase_std < _SYNTHID_PHASE_STD_MAX) & (
                (mean_mag_db - local_floor_db) >= _SYNTHID_MAG_SPIKE_DB
            )
            return int(np.sum(coherent))

        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "SynthID scan: STFT analysis failed.",
                context={"original_error": str(exc)},
            ) from exc
