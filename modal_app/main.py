"""
modal_app/main.py
Sync-Safe ML worker — all compute runs here; Django is web-only.

Architecture:
  run_orchestrator  — entry point (keep_warm=1); chains all five stages
  run_ingestion     — yt-dlp → WAV bytes  (CPU, 1 vCPU)
  run_forensics     — AI-origin detection  (CPU, 2 vCPU)
  run_analysis      — allin1 structure     (GPU A10G)
  run_nlp           — Groq Whisper + local fallback  (CPU, 2 vCPU)
  run_compliance    — Gallo-Method checks  (CPU, 2 vCPU)

Deploy:    modal deploy modal_app/main.py
Test:      modal run modal_app/main.py::run_orchestrator --job-id test \
               --source-url https://www.youtube.com/watch?v=dQw4w9WgXcQ \
               --config '{}'
"""
from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from groq import Groq

import modal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DJANGO_BASE_URL: str = "http://localhost:8000"
    MODAL_WEBHOOK_SECRET: str = ""
    WEBHOOK_TIMEOUT_S: int = 10
    GROQ_MAX_BYTES: int = 25_165_824   # 24 MB — 1 MB below Groq's 25 MB hard limit
    GROQ_CHUNK_S: int = 600             # 10-minute chunks when splitting oversized audio

    model_config = SettingsConfigDict(extra="ignore")


_settings = Settings()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App registration
# ---------------------------------------------------------------------------

app = modal.App("sync-safe")

# ---------------------------------------------------------------------------
# Modal secrets
# DJANGO_BASE_URL, MODAL_WEBHOOK_SECRET, GROQ_API_KEY stored here.
# Create via: modal secret create sync-safe-secrets KEY=value ...
# ---------------------------------------------------------------------------

_secrets = [modal.Secret.from_name("sync-safe-secrets")]

# ---------------------------------------------------------------------------
# Model-weight volume — avoids re-downloading allin1 weights on cold start
# ---------------------------------------------------------------------------

_model_volume = modal.Volume.from_name("sync-safe-models", create_if_missing=True)
_MODEL_MOUNT_PATH = "/models"

# ---------------------------------------------------------------------------
# Source-code mount — services/, core/, data/, utils/ from repo root
# Deployed via: modal deploy modal_app/main.py  (run from repo root)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_MOUNT_PREFIXES = ("services/", "core/", "data/", "utils/")

_repo_mount = modal.Mount.from_local_dir(
    str(_REPO_ROOT),
    remote_path="/app",
    condition=lambda p: any(p.startswith(prefix) for prefix in _MOUNT_PREFIXES),
)

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

# CPU image — shared base for ingestion, forensics, NLP, compliance.
# torch CPU build keeps image lean; GPU-only functions use _gpu_image.
_cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        # Infra
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        # Audio download
        "yt-dlp>=2024.11.0",
        # Audio analysis (CPU)
        "librosa>=0.10.2",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        # AI-origin forensics
        "c2pa-python>=0.29.0",
        # Compliance / lyric audit
        "better-profanity>=0.7.0",
        "detoxify>=0.5.2",
        "transformers>=4.40.0",
        # torch CPU-only for Whisper local fallback (smaller than CUDA build)
        "torch>=2.1.0",
        "openai-whisper>=20231117",
        # Groq Whisper API
        "groq>=0.9.0",
    )
    # spaCy + en_core_web_sm (NER for brand/location in compliance)
    .pip_install(
        "spacy>=3.7.0",
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    )
)

# GPU image — allin1 + CUDA torch for structure analysis only.
_gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "librosa>=0.10.2",
        "numpy>=1.26.0",
        # CUDA-enabled torch for allin1 (default index = CUDA build)
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "allin1>=1.0.0",
    )
)

# ---------------------------------------------------------------------------
# Webhook helper (used by orchestrator and each function on failure)
# ---------------------------------------------------------------------------

def _post_webhook(
    job_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error: str = "",
) -> None:
    """POST result back to Django. Raises requests.HTTPError on non-2xx."""
    import requests  # local import — not available at module-load time in tests

    base_url = os.environ.get("DJANGO_BASE_URL", _settings.DJANGO_BASE_URL)
    secret = os.environ.get("MODAL_WEBHOOK_SECRET", _settings.MODAL_WEBHOOK_SECRET)
    url = f"{base_url}/api/webhooks/analysis-complete/"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if secret:
        headers["X-Modal-Webhook-Secret"] = secret

    payload: dict[str, Any] = {"job_id": job_id, "status": status}
    if status == "complete" and result is not None:
        payload["result"] = result
    else:
        payload["error"] = error

    resp = requests.post(url, json=payload, headers=headers, timeout=_settings.WEBHOOK_TIMEOUT_S)
    resp.raise_for_status()
    logger.info("Webhook POST succeeded — job_id=%s status=%s", job_id, status)


# ---------------------------------------------------------------------------
# Step 1 — Ingestion
# ---------------------------------------------------------------------------

@app.function(
    cpu=1,
    memory=2048,
    timeout=120,
    image=_cpu_image,
    secrets=_secrets,
    mounts=[_repo_mount],
)
def run_ingestion(source_url: str) -> tuple[bytes, int]:
    """
    Download audio from source_url via yt-dlp and return raw WAV bytes.

    Returns:
        (raw_bytes, sample_rate) — passed to downstream functions.

    Raises:
        AudioSourceError: download failed, geo-blocked, or empty file.
        ValidationError:  URL is not from a permitted domain.
    """
    sys.path.insert(0, "/app")
    from services.ingestion import Ingestion  # noqa: PLC0415

    audio = Ingestion().load(source_url)
    return audio.raw, audio.sample_rate


# ---------------------------------------------------------------------------
# Step 2 — Forensics (CPU)
# ---------------------------------------------------------------------------

@app.function(
    cpu=2,
    memory=4096,
    timeout=120,
    image=_cpu_image,
    secrets=_secrets,
    mounts=[_repo_mount],
)
def run_forensics(audio_bytes: bytes, sample_rate: int) -> dict[str, Any]:
    """
    Run AI-origin detection on audio bytes.

    Returns:
        ForensicsResult.to_dict() — JSON-serialisable.

    Raises:
        ModelInferenceError: any internal analysis failure.
    """
    sys.path.insert(0, "/app")
    from core.models import AudioBuffer  # noqa: PLC0415
    from services.forensics import Forensics  # noqa: PLC0415

    buf = AudioBuffer(raw=audio_bytes, sample_rate=sample_rate)
    result = Forensics().analyze(buf)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Step 3 — Structure analysis (GPU — sole GPU function)
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    timeout=600,
    memory=8192,
    image=_gpu_image,
    secrets=_secrets,
    mounts=[_repo_mount],
    volumes={_MODEL_MOUNT_PATH: _model_volume},
)
def run_analysis(audio_bytes: bytes, sample_rate: int) -> dict[str, Any]:
    """
    Run allin1 structure analysis (BPM, key, sections, beats).

    allin1 model weights are cached in the Modal Volume to avoid re-download
    on cold starts. GPU A10G gives ~10× speedup over CPU for this workload.

    Returns:
        StructureResult.to_dict() — JSON-serialisable.

    Raises:
        ModelInferenceError: allin1 or librosa failure.
    """
    sys.path.insert(0, "/app")
    # Point allin1 weight cache at the Modal Volume so weights persist.
    os.environ.setdefault("ALLIN1_CACHE_DIR", _MODEL_MOUNT_PATH)

    from core.models import AudioBuffer  # noqa: PLC0415
    from services.analysis import Analysis  # noqa: PLC0415

    buf = AudioBuffer(raw=audio_bytes, sample_rate=sample_rate)
    result = Analysis().analyze(buf)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Step 4 — NLP / transcription (CPU — Groq primary, local Whisper fallback)
# ---------------------------------------------------------------------------

@app.function(
    cpu=2,
    memory=4096,
    timeout=300,
    image=_cpu_image,
    secrets=_secrets,
    mounts=[_repo_mount],
)
def run_nlp(audio_bytes: bytes, sample_rate: int) -> list[dict[str, Any]]:
    """
    Transcribe audio to timestamped segments.

    Primary: Groq whisper-large-v3 (fastest available, called from inside Modal
    so GROQ_API_KEY stays a Modal secret and never passes through Django).
    Fallback: local openai-whisper on CPU if Groq returns 429 or 5xx.

    Returns:
        list of {start, end, text} dicts — same shape as TranscriptSegment.to_dict().
    """
    sys.path.insert(0, "/app")
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if groq_key:
        try:
            return _transcribe_groq(audio_bytes, groq_key)
        except Exception as exc:  # noqa: BLE001 — Groq API boundary; any failure falls back to local Whisper
            logger.warning(
                "Groq transcription failed (%s) — falling back to local Whisper", exc
            )

    return _transcribe_local(audio_bytes, sample_rate)


def _transcribe_groq(audio_bytes: bytes, api_key: str) -> list[dict[str, Any]]:
    """Call Groq whisper-large-v3; chunk input if it exceeds the 25 MB limit."""
    import io

    from groq import Groq  # noqa: PLC0415

    client = Groq(api_key=api_key)

    if len(audio_bytes) <= _settings.GROQ_MAX_BYTES:
        return _groq_single(client, io.BytesIO(audio_bytes))

    # Oversized: split into fixed-duration chunks and concatenate segments.
    # Simple byte-split at constant intervals — accurate timestamps come from
    # Groq's verbose_json which includes segment-level start/end times.
    chunks = _split_audio_chunks(audio_bytes)
    segments: list[dict[str, Any]] = []
    for chunk_bytes, offset_s in chunks:
        chunk_buf = io.BytesIO(chunk_bytes)
        raw_segs = _groq_single(client, chunk_buf)
        for seg in raw_segs:
            segments.append({
                "start": round(seg["start"] + offset_s, 2),
                "end":   round(seg["end"]   + offset_s, 2),
                "text":  seg["text"],
            })
    return segments


def _groq_single(client: "Groq", audio_buf: io.BytesIO) -> list[dict[str, Any]]:
    """Transcribe a single <25 MB buffer via Groq; return normalised segments."""
    audio_buf.name = "audio.wav"
    result = client.audio.transcriptions.create(
        file=audio_buf,
        model="whisper-large-v3",
        response_format="verbose_json",
    )
    return [
        {
            "start": round(float(seg.start), 2),
            "end":   round(float(seg.end),   2),
            "text":  seg.text.strip(),
        }
        for seg in (result.segments or [])
    ]


def _split_audio_chunks(audio_bytes: bytes) -> list[tuple[bytes, float]]:
    """
    Split raw WAV bytes into chunks of at most _GROQ_MAX_BYTES.

    Returns list of (chunk_bytes, offset_seconds) tuples.
    Splitting is frame-aligned to avoid cutting mid-sample.
    """
    import io
    import wave

    try:
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            frames_per_chunk = int(framerate * _settings.GROQ_CHUNK_S)
            chunks: list[tuple[bytes, float]] = []
            for start_frame in range(0, nframes, frames_per_chunk):
                wf.setpos(start_frame)
                chunk_frames = wf.readframes(frames_per_chunk)
                # Re-wrap as a valid WAV file so Groq can parse it.
                buf = io.BytesIO()
                with wave.open(buf, "wb") as out:
                    out.setnchannels(channels)
                    out.setsampwidth(sampwidth)
                    out.setframerate(framerate)
                    out.writeframes(chunk_frames)
                offset_s = start_frame / framerate
                chunks.append((buf.getvalue(), offset_s))
        return chunks
    except Exception as exc:  # noqa: BLE001 — WAV parse failure; safe to fall back to single chunk
        logger.warning("WAV parse failed for chunk split (%s) — returning single chunk", exc)
        return [(audio_bytes, 0.0)]


def _transcribe_local(audio_bytes: bytes, sample_rate: int) -> list[dict[str, Any]]:
    """Fall back to local openai-whisper (CPU). Slower but always available."""
    from core.models import AudioBuffer  # noqa: PLC0415
    from services.transcription import Transcription  # noqa: PLC0415

    buf = AudioBuffer(raw=audio_bytes, sample_rate=sample_rate)
    segments = Transcription().transcribe(buf)
    return [s.to_dict() for s in segments]


# ---------------------------------------------------------------------------
# Step 5 — Compliance (CPU)
# ---------------------------------------------------------------------------

@app.function(
    cpu=2,
    memory=2048,
    timeout=180,
    image=_cpu_image,
    secrets=_secrets,
    mounts=[_repo_mount],
)
def run_compliance(
    audio_bytes: bytes,
    sample_rate: int,
    structure: dict[str, Any],
    transcription: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Run Gallo-Method sync-readiness checks.

    Args:
        audio_bytes:   raw WAV bytes.
        sample_rate:   native sample rate from ingestion.
        structure:     StructureResult.to_dict() from run_analysis.
        transcription: list of TranscriptSegment.to_dict() from run_nlp.

    Returns:
        ComplianceReport.to_dict() — JSON-serialisable.
    """
    sys.path.insert(0, "/app")
    from core.models import AudioBuffer, Section, TranscriptSegment  # noqa: PLC0415
    from services.compliance import Compliance  # noqa: PLC0415

    buf = AudioBuffer(raw=audio_bytes, sample_rate=sample_rate)
    transcript = [TranscriptSegment(**seg) for seg in transcription]
    sections = [Section(**s) for s in structure.get("sections", [])]
    beats: list[float] = structure.get("beats", [])

    result = Compliance().check(
        audio=buf,
        transcript=transcript,
        sections=sections,
        beats=beats,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Orchestrator — entry point Django dispatches to
# ---------------------------------------------------------------------------

@app.function(
    cpu=2,
    memory=4096,
    timeout=900,  # ingestion + all stages; generous ceiling for cold GPU starts
    image=_cpu_image,
    secrets=_secrets,
    keep_warm=1,  # one warm container to reduce cold-start latency for end users
)
def run_orchestrator(job_id: str, source_url: str, config: dict[str, Any]) -> None:
    """
    Chain all five stages and POST the combined result to Django.

    Ingestion runs first (sequential) since all analysis stages need audio bytes.
    Forensics, analysis, and NLP run in parallel (spawned) to minimise wall time.
    Compliance runs last because it depends on structure + transcription outputs.

    On any unhandled exception a failure webhook is POSTed so Django can mark
    the Analysis as failed. The exception is then re-raised so Modal records it
    and fires the configured retry (MODAL_RETRIES in existing app).
    """
    try:
        # Stage 1 — ingest (must complete before analysis can proceed)
        audio_bytes, sample_rate = run_ingestion.remote(source_url)

        # Stages 2-4 — parallel: forensics, structure, transcription
        forensics_future = run_forensics.spawn(
            audio_bytes=audio_bytes, sample_rate=sample_rate
        )
        analysis_future = run_analysis.spawn(
            audio_bytes=audio_bytes, sample_rate=sample_rate
        )
        nlp_future = run_nlp.spawn(
            audio_bytes=audio_bytes, sample_rate=sample_rate
        )

        forensics: dict[str, Any] = forensics_future.get()
        structure: dict[str, Any] = analysis_future.get()
        transcription: list[dict[str, Any]] = nlp_future.get()

        # Stage 5 — compliance (sequential; depends on structure + transcription)
        compliance: dict[str, Any] = run_compliance.remote(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            structure=structure,
            transcription=transcription,
        )

        result: dict[str, Any] = {
            "job_id":        job_id,
            "source_url":    source_url,
            "title":         config.get("title", ""),
            "artist":        config.get("artist", ""),
            "forensics":     forensics,
            "structure":     structure,
            "transcription": transcription,
            "compliance":    compliance,
        }
        _post_webhook(job_id=job_id, status="complete", result=result)

    except Exception as exc:  # noqa: BLE001 — top-level boundary; guarantees webhook POST
        logger.exception("Orchestrator failed for job_id=%s: %s", job_id, exc)
        try:
            _post_webhook(job_id=job_id, status="failed", error=str(exc))
        except Exception as post_exc:  # noqa: BLE001
            # Webhook POST itself failed — log and re-raise for Modal's retry.
            logger.exception(
                "Webhook POST also failed for job_id=%s: %s", job_id, post_exc
            )
        raise
