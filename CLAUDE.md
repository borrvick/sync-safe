# CLAUDE.md — Sync-Safe Forensic & Compliance Portal

## Collaboration Rules
1. **Token Hygiene**: Before implementing any service, provide a 3-sentence summary. Wait for 'OK'.
2. **Modular Design**: Write one service file at a time.
3. **Output Limits**: Use quiet flags (`yt-dlp -q`, `pytest --quiet`).
4. **Statelessness**: Use `try/finally` to delete any temp files immediately after use.

## Architecture Constraints
- **Hosting**: Hugging Face ZeroGPU (free tier — 25 min/day GPU quota)
- **GPU Decorator**: Wrap `@spaces.GPU` on all Whisper, allin1, and spaCy NER functions. Must be self-contained.
- **No Database**: All state is ephemeral. Recommendations use Live Similarity (Last.fm + yt-dlp).
- **No Disk Writes**: Audio ingestion uses `io.BytesIO` only. Compliance uses `tempfile.TemporaryDirectory` with `try/finally`.
- **Security**: `shlex.quote()` on all strings passed to yt-dlp. API keys via `st.secrets`/`os.environ`.

## File Map
| File | Responsibility |
|------|---------------|
| `app.py` | Main Streamlit UI — orchestrates flow, sidebar inputs, metric cards, audio state manager |
| `services/ingestion.py` | YouTube URL (yt-dlp → BytesIO) and file upload handling |
| `services/forensics.py` | `@spaces.GPU` — C2PA manifest check, librosa groove/IBI analysis, spectral fingerprint loop detection |
| `services/analysis.py` | `@spaces.GPU` — allin1 structure (BPM, beats, labels), energy metrics |
| `services/nlp.py` | `@spaces.GPU` — Whisper lyrics transcription → timestamped JSON segments |
| `services/compliance.py` | `@spaces.GPU` — Gallo-Method compliance checks (sting, 4-8 bar, intro timer, lyric audit) |
| `services/discovery.py` | Last.fm `track.getSimilar` → yt-dlp `ytsearch1:` URL lookup |
| `services/legal.py` | ASCAP/BMI/SESAC sync licensing link generator |

## Forensic Detection Logic
- **C2PA**: `c2pa-python` v2.3+ — check for "born-AI" assertions in manifests
- **Groove/IBI**: librosa beat tracking; flag zero-variance IBI as "Perfect Quantization" (AI signal)
- **Spectral Slop**: Flag anomalies in 16kHz+ range
- **Loop Detection**: Cross-correlation of 4/8-bar segments; score >0.98 → "Likely Stock Loop"

## Gallo-Method Compliance Logic
- **Sting Check**: librosa onset strength — detect sharp energy drop at track end. Flag if final 2s energy < 5% of mean OR track ends on root-note hit (sustained final onset).
- **4-8 Bar Rule**: allin1 beat grid + librosa spectral contrast — verify energy evolution (≥10% delta) across every 4-bar window. Flag stagnant segments.
- **Intro Timer**: allin1 `section_labels` — if any segment labelled `intro` exceeds 15 seconds, flag it.
- **Lyric Audit Pipeline**:
  1. `openai-whisper` (Base model) → timestamped JSON `[{start, end, text}]`
  2. `profanity-check` → flag explicit segments
  3. `spacy en_core_web_sm` NER → flag `ORG` (brand mentions) and `GPE` (locations)
  4. Keyword dict (`VIOLENCE_TERMS`, `DRUG_TERMS`) → "Safety Zone" classification
  5. Output: `[{timestamp_s, issue_type, text, recommendation}]`

## Audio State Manager (app.py)
- `st.session_state.start_time` — current playhead position in seconds (int)
- `st.session_state.player_key` — increments on every timestamp click to force `st.audio` re-init
- Pattern: `st.audio(bytes, start_time=st.session_state.start_time, key=f"player_{st.session_state.player_key}")`

## Lyric Audit Table Schema
| Column | Type | Notes |
|--------|------|-------|
| `timestamp_s` | int | Seconds from track start |
| `issue_type` | str | `EXPLICIT` / `BRAND` / `LOCATION` / `VIOLENCE` / `DRUGS` |
| `text` | str | Flagged transcript excerpt |
| `recommendation` | str | Supervisor action guidance |
