# Sync-Safe

Music sync licensing analysis platform.

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16 (App Router) — deployed on Vercel |
| Backend API | Django 5 + DRF — deployed on Railway |
| ML Worker | Modal (GPU) — Whisper + allin1 + librosa |
| Database | PostgreSQL (Railway) |

## Repo layout

```
backend/        Django API, models, webhooks
frontend/       Next.js frontend
modal_worker/   Modal GPU worker (deploy: modal deploy modal_worker/app.py)
legacy/         Archived Streamlit / HuggingFace app (superseded)
```

## Development

### Backend
```bash
cd backend
pip install -r requirements-dev.txt
python manage.py runserver
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Modal worker
```bash
pip install modal
modal deploy modal_worker/app.py
```

Set `USE_FIXTURE_WORKER=True` in your `.env` to exercise the full
Django → webhook → complete flow locally without spending Modal GPU credits.
