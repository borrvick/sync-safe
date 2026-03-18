# CLAUDE.md — Sync-Safe Forensic & Compliance Portal

---

## Collaboration Rules

1. **Dry Run First**: Before implementing any service, provide a 3-sentence summary of the proposed class hierarchy and data flow. Wait for **'OK'** before writing code.
2. **Modular Design**: Write one service file at a time.
3. **Output Limits**: Use quiet flags (`yt-dlp -q`, `pytest --quiet`).
4. **Statelessness**: Use `try/finally` to delete any temp files immediately after use.

---

## Hosting & Runtime Constraints

- **Platform**: Hugging Face ZeroGPU (free tier — 25 min/day GPU quota)
- **GPU Decorator**: Wrap `@spaces.GPU` on all Whisper, allin1, and spaCy NER functions. Each decorated function must be fully self-contained (no shared GPU state between calls).
- **No Database**: All state is ephemeral. Recommendations use Live Similarity (Last.fm + yt-dlp).
- **No Disk Writes**: Audio ingestion uses `io.BytesIO` only. Compliance uses `tempfile.TemporaryDirectory` with `try/finally`.
- **Security**: `shlex.quote()` on all strings passed to yt-dlp. API keys via `st.secrets` / `os.environ`.

---

## File Map

| File | Responsibility |
|------|----------------|
| `app.py` | Main Streamlit UI — orchestrates flow, sidebar inputs, metric cards, audio state manager |
| `services/ingestion.py` | YouTube URL (yt-dlp → BytesIO) and file upload handling |
| `services/forensics.py` | `@spaces.GPU` — C2PA manifest check, librosa groove/IBI analysis, spectral fingerprint loop detection |
| `services/analysis.py` | `@spaces.GPU` — allin1 structure (BPM, beats, labels), energy metrics |
| `services/nlp.py` | `@spaces.GPU` — Whisper lyrics transcription → timestamped JSON segments |
| `services/compliance.py` | `@spaces.GPU` — Gallo-Method compliance checks (sting, 4-8 bar, intro timer, lyric audit) |
| `services/discovery.py` | Last.fm `track.getSimilar` → yt-dlp `ytsearch1:` URL lookup |
| `services/legal.py` | ASCAP/BMI/SESAC sync licensing link generator |

---

## Architecture: Interface-First

Define contracts before implementations. All external integrations must be expressed as `Protocol` classes before any concrete class is written. Required interfaces:

```python
from typing import Protocol

class AudioProcessor(Protocol): ...
class TranscriptionProvider(Protocol): ...
class StorageProvider(Protocol): ...
```

**Rule:** The class that runs Whisper must not be the class that saves results to disk. Swapping Whisper for Deepgram, or local storage for S3, must require zero changes to core business logic.

---

## SOLID Compliance

- **Single Responsibility**: Every class does one thing. If you can describe its purpose using the word "and," split it.
- **Open/Closed**: Extend via new classes or injected dependencies — never by modifying existing ones.
- **Dependency Inversion**: High-level modules depend on abstractions (Protocols), not concretions.

---

## Dependency Injection

Use **Constructor Injection** exclusively.

```python
class TranscriptionService:
    def __init__(self, provider: TranscriptionProvider, storage: StorageProvider):
        self.provider = provider
        self.storage = storage
```

- If a class requires **more than 3 dependencies**, propose a `ServiceRegistry` or `Context` object instead.
- **Never** use `global` state or singletons unless strictly required for hardware-level access (e.g., GPU device handle). Any exception must be justified in a comment.

---

## Configuration Hierarchy

Use `pydantic-settings` for all config. No flat `constants.py` files.

| Tier | Examples | Source |
|------|----------|--------|
| **System Constants** | Sample rates, FFT sizes, bar lengths | Hardcoded in `Settings` |
| **Environment Variables** | API keys, file paths | `.env` / `st.secrets` |
| **Model Hyperparameters** | Whisper temperature, Demucs segments, IBI variance threshold | Config file or env |

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SAMPLE_RATE: int = 44100
    INTRO_MAX_SECONDS: int = 15
    LOOP_CORRELATION_THRESHOLD: float = 0.98
    STING_ENERGY_RATIO: float = 0.05
    BAR_ENERGY_DELTA_MIN: float = 0.10

    LASTFM_API_KEY: str
    OPENAI_API_KEY: str

    WHISPER_MODEL: str = "base"
    WHISPER_TEMPERATURE: float = 0.0
    DEMUCS_SEGMENTS: int = 4

    class Config:
        env_file = ".env"
```

**Rule:** No magic numbers in implementation code. Every numeric constant (thresholds, durations, model names) belongs in `Settings`.

---

## Error Handling

Audio pipelines are brittle. Use domain-specific exception types — never bare `except Exception`.

Required custom exceptions:

```python
class AudioSourceError(Exception): ...       # File not found, unreadable format, yt-dlp failure
class ModelInferenceError(Exception): ...    # OOM, timeout, Whisper/allin1 crash
class ValidationError(Exception): ...        # Bad input shape, invalid params, missing segments
```

All I/O and heavy-processing methods must catch low-level errors and re-raise as the appropriate type:

```python
try:
    result = whisper_model.transcribe(audio_path)
except torch.cuda.OutOfMemoryError as e:
    raise ModelInferenceError("Whisper OOM — reduce segment size or lower DEMUCS_SEGMENTS") from e
```

---

## Pure Functions

Move as much logic as possible into pure functions (no side effects, no I/O, deterministic output). This is especially important for:

- Gallo-Method threshold checks
- IBI variance calculations
- Lyric keyword matching
- Compliance flag generation

If a function reads a file, calls a model, or writes output, isolate it from logic that doesn't need to.

---

## Type Hinting

Strict type hints required on every function signature.

```python
from typing import Optional, Union
from pathlib import Path

def run_lyric_audit(
    segments: list[dict[str, str | float]],
    config: Optional[Settings] = None,
) -> list[dict[str, str | int]]:
    ...
```

- Use lowercase `list[T]` and `dict[K, V]` (Python 3.10+)
- Use `Optional[T]` where `None` is valid
- No untyped function signatures

---

## Forensic Detection Logic

- **C2PA**: `c2pa-python` v2.3+ — check for "born-AI" assertions in manifests
- **Groove/IBI**: librosa beat tracking; flag zero-variance IBI as "Perfect Quantization" (AI signal)
- **Spectral Slop**: Flag anomalies in 16kHz+ range
- **Loop Detection**: Cross-correlation of 4/8-bar segments; score > `Settings.LOOP_CORRELATION_THRESHOLD` → "Likely Stock Loop"

---

## Gallo-Method Compliance Logic

- **Sting Check**: librosa onset strength — detect sharp energy drop at track end. Flag if final 2s energy < `Settings.STING_ENERGY_RATIO` of mean OR track ends on root-note hit.
- **4-8 Bar Rule**: allin1 beat grid + librosa spectral contrast — verify energy evolution (≥ `Settings.BAR_ENERGY_DELTA_MIN` delta) across every 4-bar window. Flag stagnant segments.
- **Intro Timer**: allin1 `section_labels` — flag any segment labelled `intro` that exceeds `Settings.INTRO_MAX_SECONDS`.
- **Lyric Audit Pipeline**:
  1. `openai-whisper` (Base model) → timestamped JSON `[{start, end, text}]`
  2. `profanity-check` → flag explicit segments
  3. `spacy en_core_web_sm` NER → flag `ORG` (brand mentions) and `GPE` (locations)
  4. Keyword dict (`VIOLENCE_TERMS`, `DRUG_TERMS`) → "Safety Zone" classification
  5. Output: `[{timestamp_s, issue_type, text, recommendation}]`

---

## Audio State Manager (`app.py`)

- `st.session_state.start_time` — current playhead position in seconds (int)
- `st.session_state.player_key` — increments on every timestamp click to force `st.audio` re-init
- Pattern: `st.audio(bytes, start_time=st.session_state.start_time, key=f"player_{st.session_state.player_key}")`

---

## Lyric Audit Table Schema

| Column | Type | Notes |
|--------|------|-------|
| `timestamp_s` | int | Seconds from track start |
| `issue_type` | str | `EXPLICIT` / `BRAND` / `LOCATION` / `VIOLENCE` / `DRUGS` |
| `text` | str | Flagged transcript excerpt |
| `recommendation` | str | Supervisor action guidance |

---

## Pre-Commit Checklist

Before submitting any code change, verify:

- [ ] Dry Run summary provided and approved before implementation
- [ ] All external integrations have a `Protocol` defined
- [ ] No class has more than one primary responsibility
- [ ] No magic numbers — all numeric constants in `Settings`
- [ ] All I/O and model calls raise domain-specific exceptions
- [ ] Constructor injection used; no `global` state
- [ ] Every function signature has complete type hints
- [ ] Pure functions used wherever side effects aren't needed
- [ ] `try/finally` cleans up any `tempfile.TemporaryDirectory`
- [ ] All yt-dlp strings wrapped in `shlex.quote()`
- [ ] All GPU functions decorated with `@spaces.GPU` and self-contained