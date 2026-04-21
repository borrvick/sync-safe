"""
modal_worker/app.py
Modal ML worker — receives analysis jobs from Django and POSTs results back.

Deploy:     modal deploy modal_worker/app.py
Test local: modal run modal_worker/app.py::run_analysis --job-id test --source-url https://...
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
from typing import Any

import modal
import requests
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings — all numeric constants and model names live here, not inline
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    DJANGO_BASE_URL:      str   = "http://localhost:8000"
    MODAL_WEBHOOK_SECRET: str   = ""
    WHISPER_MODEL:        str   = "base"
    WHISPER_TEMPERATURE:  float = 0.0
    # Modal function infrastructure — read at deploy time (modal deploy)
    MODAL_TIMEOUT:        int   = 600   # seconds; Whisper Base on T4 takes ~1–3 min
    MODAL_RETRIES:        int   = 1     # one automatic retry on transient failure

    model_config = SettingsConfigDict(extra="ignore")


_settings = Settings()


# ---------------------------------------------------------------------------
# Modal app + GPU image
# ---------------------------------------------------------------------------

app = modal.App("sync-safe-ml-worker")

# ffmpeg is required by yt-dlp (audio extraction) and librosa (decoding).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "requests==2.32.3",
        "pydantic-settings==2.13.1",
        "yt-dlp==2025.3.31",
        "openai-whisper==20231117",
        "torch==2.2.2",
        "torchaudio==2.2.2",
        "librosa==0.10.2",
        "allin1==0.1.5",
        "numpy==1.26.4",
    )
)


# ---------------------------------------------------------------------------
# Analysis entry point
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="any",
    timeout=_settings.MODAL_TIMEOUT,
    retries=_settings.MODAL_RETRIES,
)
def run_analysis(job_id: str, source_url: str, config: dict[str, Any]) -> None:
    """
    Entry point called by ModalMLWorkerProvider.dispatch().

    Downloads audio, runs Whisper transcription + allin1/librosa structure
    analysis, then POSTs the full result back to the Django webhook.

    Args:
        job_id:     UUID string matching Analysis.id in the Django DB.
        source_url: YouTube URL or public audio URL to analyse.
        config:     Metadata dict from the API (title, artist, etc.).
    """
    # Re-instantiate Settings inside the container so Modal env vars are picked up.
    s = Settings()
    webhook_url = f"{s.DJANGO_BASE_URL}/api/webhooks/analysis-complete/"

    try:
        result = _run_analysis_pipeline(
            job_id=job_id,
            source_url=source_url,
            config=config,
            s=s,
        )
        _post_result(
            webhook_url=webhook_url,
            webhook_secret=s.MODAL_WEBHOOK_SECRET,
            job_id=job_id,
            job_status="complete",
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 — top-level boundary; guarantees webhook POST
        logger.exception("Analysis failed for job_id=%s: %s", job_id, exc)
        # Attempt to notify Django of the failure. If the webhook endpoint is also
        # unreachable, log and re-raise so Modal's retry fires.
        try:
            _post_result(
                webhook_url=webhook_url,
                webhook_secret=s.MODAL_WEBHOOK_SECRET,
                job_id=job_id,
                job_status="failed",
                result={"error": str(exc)},
            )
        except Exception as post_exc:  # noqa: BLE001
            logger.exception(
                "Webhook POST also failed for job_id=%s: %s — re-raising for Modal retry",
                job_id,
                post_exc,
            )
            raise


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def _run_analysis_pipeline(
    job_id: str,
    source_url: str,
    config: dict[str, Any],
    s: Settings,
) -> dict[str, Any]:
    """
    Download audio, transcribe, analyse structure, return result dict.

    A single TemporaryDirectory is used for the whole pipeline so all
    sub-functions share the same audio file without extra I/O.
    try/finally guarantees cleanup even if a sub-function raises.
    """
    tmpdir_obj = tempfile.TemporaryDirectory()
    try:
        tmpdir      = tmpdir_obj.name
        audio_path  = _download_audio(source_url=source_url, tmpdir=tmpdir)
        audio_bytes = _load_bytes(audio_path)

        transcription = _transcribe(audio_path=audio_path, s=s)
        structure     = _analyse_structure(audio_path=audio_path, audio_bytes=audio_bytes)

        return {
            "job_id":        job_id,
            "source_url":    source_url,
            "title":         config.get("title", ""),
            "artist":        config.get("artist", ""),
            "duration":      structure["duration"],
            "bpm":           structure["bpm"],
            "beats":         structure["beats"],
            "sections":      structure["sections"],
            "transcription": transcription,
        }
    finally:
        tmpdir_obj.cleanup()


# ---------------------------------------------------------------------------
# Step 1 — download
# ---------------------------------------------------------------------------

def _download_audio(source_url: str, tmpdir: str) -> str:
    """
    Download audio from source_url via yt-dlp, convert to WAV, return path.

    Args:
        source_url: YouTube URL or any yt-dlp-compatible URL.
        tmpdir:     Temporary directory managed by the caller.

    Returns:
        Absolute path to the downloaded WAV file inside tmpdir.

    Raises:
        OSError: if yt-dlp exits non-zero or downloads a zero-byte file.
    """
    out_path = os.path.join(tmpdir, "audio.wav")
    # List form (not shell=True) so source_url is never shell-interpreted.
    try:
        subprocess.run(
            [
                "yt-dlp",
                "--quiet",
                "-x",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", out_path,
                source_url,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise OSError(
            f"yt-dlp failed for source_url={source_url}: {exc.stderr.decode()}"
        ) from exc

    # Guard against geo-blocked or empty downloads before passing to ML models.
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise OSError(
            f"yt-dlp produced an empty or missing file for source_url={source_url}"
        )

    logger.info("Downloaded audio to %s for source_url=%s", out_path, source_url)
    return out_path


def _load_bytes(audio_path: str) -> io.BytesIO:
    """Read audio file into BytesIO for librosa."""
    with open(audio_path, "rb") as f:
        buf = io.BytesIO(f.read())
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Step 2 — transcription (Whisper)
# ---------------------------------------------------------------------------

def _transcribe(audio_path: str, s: Settings) -> list[dict[str, Any]]:
    """
    Transcribe audio with Whisper and return timestamped segments.

    Args:
        audio_path: Path to a WAV file on disk.
        s:          Settings instance (provides WHISPER_MODEL, WHISPER_TEMPERATURE).

    Returns:
        List of segment dicts: [{start, end, text}, ...]
        Empty list if Whisper returns no segments.

    Raises:
        RuntimeError: wraps any Whisper internal error.
    """
    import whisper  # heavy import — kept local so module-level import is fast

    try:
        model  = whisper.load_model(s.WHISPER_MODEL)
        result = model.transcribe(audio_path, fp16=False, temperature=s.WHISPER_TEMPERATURE)
    except Exception as exc:
        raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

    segments = result.get("segments") or []
    return [
        {"start": round(seg["start"], 2), "end": round(seg["end"], 2), "text": seg["text"].strip()}
        for seg in segments
    ]


# ---------------------------------------------------------------------------
# Step 3 — structure analysis (allin1 + librosa)
# ---------------------------------------------------------------------------

def _analyse_structure(
    audio_path: str,
    audio_bytes: io.BytesIO,
) -> dict[str, Any]:
    """
    Run allin1 for BPM/beats/sections and librosa for duration.

    Args:
        audio_path:  Path to WAV file (allin1 requires a file path).
        audio_bytes: BytesIO of the same file (librosa reads from BytesIO).

    Returns:
        Dict with keys: duration (float), bpm (float), beats (list[float]),
        sections (list[{start, end, label}]).

    Raises:
        RuntimeError: wraps any allin1 or librosa internal error.
    """
    import allin1   # heavy imports kept local
    import librosa

    try:
        audio_bytes.seek(0)
        y, sr    = librosa.load(audio_bytes, sr=None, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))
    except Exception as exc:
        raise RuntimeError(f"librosa load failed: {exc}") from exc

    try:
        analysis = allin1.analyze(audio_path)
        bpm      = float(analysis.bpm)
        beats    = [round(float(b), 3) for b in (analysis.beats or [])]
        sections = [
            {
                "start": round(float(seg.start), 2),
                "end":   round(float(seg.end), 2),
                "label": seg.label,
            }
            for seg in (analysis.segments or [])
        ]
    except Exception as exc:
        raise RuntimeError(f"allin1 analysis failed: {exc}") from exc

    return {"duration": round(duration, 2), "bpm": round(bpm, 2), "beats": beats, "sections": sections}


# ---------------------------------------------------------------------------
# Webhook callback
# ---------------------------------------------------------------------------

def _post_result(
    webhook_url: str,
    webhook_secret: str,
    job_id: str,
    job_status: str,
    result: dict[str, Any],
) -> None:
    """
    POST analysis result back to Django webhook endpoint.

    Raises:
        requests.HTTPError: if Django returns a non-2xx response.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if webhook_secret:
        headers["X-Modal-Webhook-Secret"] = webhook_secret

    response = requests.post(
        webhook_url,
        json={"job_id": job_id, "status": job_status, "result": result},
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    logger.info("Webhook POST succeeded for job_id=%s status=%s", job_id, job_status)
