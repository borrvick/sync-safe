'use strict';
// Lightweight mock for the Django API used exclusively by Playwright E2E tests.
// Started as a child process by playwright.config.ts webServer[0].
const http = require('http');

const PORT = parseInt(process.argv[2] ?? '4001', 10);

const ANALYSIS_ID = 'test-abc-123';
const ERROR_LABEL_ID = 'error-label-id';

function makeAnalysis(id) {
  return {
    id,
    title: 'Test Track',
    artist: 'Test Artist',
    source_url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    status: 'complete',
    label: '',
    error: '',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    result_json: {
      structure: {
        bpm: 120.0,
        key: 'C major',
        duration_s: 210.0,
        sections: [
          { label: 'intro', start: 0, end: 14 },
          { label: 'verse', start: 14, end: 60 },
        ],
      },
      forensics: {
        c2pa_verdict: 'CLEAN',
        flags: [],
        loop_detected: false,
        loop_score: 0.1,
        perfect_quantization: false,
        spectral_anomaly: false,
        ibi_variance: 252.0,
      },
      compliance: {
        overall_pass: true,
        sting_pass: true,
        bar_rule_pass: true,
        intro_pass: true,
        flags: [],
        lyric_flags: [
          {
            timestamp_s: 45,
            issue_type: 'EXPLICIT',
            text: 'example flagged line',
            recommendation: 'Review with supervisor',
          },
        ],
      },
      transcription: [
        { start: 0, end: 4, text: 'Hello world' },
        { start: 45, end: 49, text: 'Example flagged line' },
      ],
    },
  };
}

const LABELS = [
  { slug: 'sync-ready', name: 'Sync Ready', description: '', sort_order: 1 },
  { slug: 'needs-review', name: 'Needs Review', description: '', sort_order: 2 },
  { slug: 'rejected', name: 'Rejected', description: '', sort_order: 3 },
];

const ANALYSES_LIST = {
  count: 1,
  next: null,
  previous: null,
  results: [
    {
      id: ANALYSIS_ID,
      title: 'Test Track',
      artist: 'Test Artist',
      source_url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
      status: 'complete',
      label: '',
      error: '',
      created_at: '2026-01-01T00:00:00Z',
    },
  ],
};

const CORS_HEADERS = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PATCH, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

function json(res, status, body) {
  res.writeHead(status, CORS_HEADERS);
  res.end(JSON.stringify(body));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => { data += chunk; });
    req.on('end', () => {
      try { resolve(data ? JSON.parse(data) : {}); }
      catch { reject(new Error('Invalid JSON')); }
    });
  });
}

// Strip trailing slash for uniform matching.
function normPath(url) {
  const [path] = url.split('?');
  return path.length > 1 && path.endsWith('/') ? path.slice(0, -1) : path;
}

const server = http.createServer(async (req, res) => {
  try {
    await handleRequest(req, res);
  } catch {
    json(res, 500, { detail: 'Mock server error.' });
  }
});

async function handleRequest(req, res) {
  const { method } = req;
  const path = normPath(req.url ?? '/');

  if (method === 'OPTIONS') {
    res.writeHead(204, CORS_HEADERS);
    return res.end();
  }

  if (path === '/health' && method === 'GET') {
    return json(res, 200, { ok: true });
  }

  if (path === '/api/analyses') {
    if (method === 'GET') return json(res, 200, ANALYSES_LIST);
    if (method === 'POST') {
      let body;
      try { body = await readBody(req); }
      catch { return json(res, 400, { detail: 'Bad request.' }); }
      const url = typeof body.source_url === 'string' ? body.source_url : '';
      if (!url.includes('youtube')) {
        return json(res, 400, { source_url: ['This URL is not supported. Use a YouTube link.'] });
      }
      return json(res, 201, { id: ANALYSIS_ID });
    }
  }

  if (path === '/api/analyses/labels' && method === 'GET') {
    return json(res, 200, LABELS);
  }

  if (path === `/api/analyses/${ANALYSIS_ID}` && method === 'GET') {
    return json(res, 200, makeAnalysis(ANALYSIS_ID));
  }

  if (path === `/api/analyses/${ERROR_LABEL_ID}` && method === 'GET') {
    return json(res, 200, makeAnalysis(ERROR_LABEL_ID));
  }

  if (path === `/api/analyses/${ANALYSIS_ID}/label` && method === 'PATCH') {
    await readBody(req);
    return json(res, 200, {});
  }

  if (path === `/api/analyses/${ERROR_LABEL_ID}/label` && method === 'PATCH') {
    await readBody(req);
    return json(res, 500, { detail: 'Internal server error.' });
  }

  return json(res, 404, { detail: 'Not found.' });
}

server.listen(PORT, () => {
  process.stdout.write(`Mock API listening on http://localhost:${PORT}\n`);
});

process.on('SIGTERM', () => server.close(() => process.exit(0)));
process.on('SIGINT', () => server.close(() => process.exit(0)));
