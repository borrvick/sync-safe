#!/bin/bash
# run.sh — launch the Sync-Safe Forensic Portal locally
# Sets DYLD_LIBRARY_PATH so torchcodec can find libavutil.56 (FFmpeg 4.x)
# required by torchaudio 2.9+ → demucs → allin1 on macOS.

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@4/lib:${DYLD_LIBRARY_PATH}"

cd "$(dirname "$0")"
.venv/bin/streamlit run app.py "$@"
