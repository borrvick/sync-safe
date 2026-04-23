import Link from "next/link";
import { notFound } from "next/navigation";
import { apiFetch } from "@/app/lib/api";
import type { Analysis, TrackLabel } from "@/app/lib/definitions";
import { Poller } from "./poller";
import { LabelSelector } from "./label-selector";

// ---------------------------------------------------------------------------
// Data fetching
// ---------------------------------------------------------------------------

async function getAnalysis(id: string): Promise<Analysis | null> {
  const res = await apiFetch(`/api/analyses/${id}/`);
  if (res.status === 404) return null;
  if (!res.ok) return null;
  return res.json() as Promise<Analysis>;
}

async function getLabels(): Promise<TrackLabel[]> {
  const res = await apiFetch("/api/analyses/labels/");
  if (!res.ok) return [];
  return res.json() as Promise<TrackLabel[]>;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function Pill({
  children,
  variant = "neutral",
}: {
  children: React.ReactNode;
  variant?: "neutral" | "pass" | "fail" | "warn";
}) {
  const cls = {
    neutral: "bg-gray-100 text-gray-700",
    pass: "bg-green-100 text-green-800",
    fail: "bg-red-100 text-red-800",
    warn: "bg-yellow-100 text-yellow-800",
  }[variant];
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}
    >
      {children}
    </span>
  );
}

function SectionCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="bg-white rounded-xl border border-gray-200 p-5">
      <h2 className="text-sm font-semibold text-gray-900 mb-4">{title}</h2>
      {children}
    </section>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 py-2 border-b border-gray-50 last:border-0">
      <span className="text-xs text-gray-500 shrink-0 w-40">{label}</span>
      <span className="text-sm text-gray-900 text-right">{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Report sections
// ---------------------------------------------------------------------------

type ResultJson = Record<string, unknown>;

function TrackOverview({
  analysis,
  result,
}: {
  analysis: Analysis;
  result: ResultJson;
}) {
  const structure = result.structure as Record<string, unknown> | undefined;
  const bpm = structure?.bpm as number | undefined;
  const key = structure?.key as string | undefined;
  const duration = structure?.duration_s as number | undefined;
  const sections = (structure?.sections as { label: string; start: number; end: number }[]) ?? [];

  function fmt(s: number) {
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  }

  return (
    <SectionCard title="Track Overview">
      <div className="mb-4">
        <Row label="Title" value={analysis.title || "—"} />
        <Row label="Artist" value={analysis.artist || "—"} />
        <Row
          label="Source"
          value={
            <a
              href={analysis.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-600 hover:underline truncate max-w-xs block"
            >
              YouTube ↗
            </a>
          }
        />
        {bpm !== undefined && bpm > 0 && (
          <Row label="BPM" value={bpm.toFixed(1)} />
        )}
        {key && <Row label="Key" value={key} />}
        {duration !== undefined && duration > 0 && (
          <Row label="Duration" value={fmt(duration)} />
        )}
      </div>

      {sections.length > 0 && (
        <>
          <p className="text-xs font-medium text-gray-500 mb-2">Sections</p>
          <div className="flex flex-wrap gap-2">
            {sections.map((s, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-700"
              >
                <span className="font-medium capitalize">{s.label}</span>
                <span className="text-gray-400">
                  {fmt(s.start)}–{fmt(s.end)}
                </span>
              </span>
            ))}
          </div>
        </>
      )}
    </SectionCard>
  );
}

function AuthenticityAudit({ result }: { result: ResultJson }) {
  const f = result.forensics as Record<string, unknown> | undefined;
  if (!f) return null;

  const verdict = f.c2pa_verdict as string;
  const flags = (f.flags as string[]) ?? [];
  const loopDetected = f.loop_detected as boolean;
  const loopScore = f.loop_score as number;
  const perfectQ = f.perfect_quantization as boolean;
  const spectral = f.spectral_anomaly as boolean;
  const ibi = f.ibi_variance as number;

  const verdictVariant =
    verdict === "CLEAN" ? "pass" : verdict === "AI_GENERATED" ? "fail" : "warn";

  return (
    <SectionCard title="Authenticity Audit">
      <Row
        label="C2PA verdict"
        value={<Pill variant={verdictVariant}>{verdict}</Pill>}
      />
      <Row
        label="IBI variance"
        value={ibi !== undefined ? ibi.toFixed(4) : "—"}
      />
      <Row
        label="Perfect quantization"
        value={
          <Pill variant={perfectQ ? "fail" : "pass"}>
            {perfectQ ? "Yes" : "No"}
          </Pill>
        }
      />
      <Row
        label="Loop detected"
        value={
          <Pill variant={loopDetected ? "warn" : "pass"}>
            {loopDetected ? `Yes (score: ${loopScore.toFixed(2)})` : "No"}
          </Pill>
        }
      />
      <Row
        label="Spectral anomaly"
        value={
          <Pill variant={spectral ? "warn" : "pass"}>
            {spectral ? "Yes" : "No"}
          </Pill>
        }
      />
      {flags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {flags.map((flag) => (
            <Pill key={flag} variant="fail">
              {flag}
            </Pill>
          ))}
        </div>
      )}
    </SectionCard>
  );
}

function SyncReadiness({ result }: { result: ResultJson }) {
  const c = result.compliance as Record<string, unknown> | undefined;
  if (!c) return null;

  const overall = c.overall_pass as boolean;
  const sting = c.sting_pass as boolean;
  const bar = c.bar_rule_pass as boolean;
  const intro = c.intro_pass as boolean;
  const flags = (c.flags as string[]) ?? [];

  return (
    <SectionCard title="Sync Readiness">
      <Row
        label="Overall"
        value={
          <Pill variant={overall ? "pass" : "fail"}>
            {overall ? "Pass" : "Fail"}
          </Pill>
        }
      />
      <Row
        label="Sting check"
        value={<Pill variant={sting ? "pass" : "fail"}>{sting ? "Pass" : "Fail"}</Pill>}
      />
      <Row
        label="4–8 bar rule"
        value={<Pill variant={bar ? "pass" : "fail"}>{bar ? "Pass" : "Fail"}</Pill>}
      />
      <Row
        label="Intro length"
        value={<Pill variant={intro ? "pass" : "fail"}>{intro ? "Pass" : "Fail"}</Pill>}
      />
      {flags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {flags.map((flag) => (
            <Pill key={flag} variant="fail">
              {flag}
            </Pill>
          ))}
        </div>
      )}
    </SectionCard>
  );
}

function LyricsAudit({ result }: { result: ResultJson }) {
  const c = result.compliance as Record<string, unknown> | undefined;
  const transcription = (result.transcription as { start: number; end: number; text: string }[]) ?? [];
  const lyricFlags = (c?.lyric_flags as Record<string, unknown>[]) ?? [];

  if (transcription.length === 0 && lyricFlags.length === 0) return null;

  function fmt(s: number) {
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  }

  return (
    <SectionCard title="Lyrics & Content Audit">
      {lyricFlags.length > 0 && (
        <div className="mb-4">
          <p className="text-xs font-medium text-gray-500 mb-2">Flagged segments</p>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-500 border-b border-gray-100">
                  <th className="pb-2 pr-3 font-medium">Time</th>
                  <th className="pb-2 pr-3 font-medium">Issue</th>
                  <th className="pb-2 pr-3 font-medium">Text</th>
                  <th className="pb-2 font-medium">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {lyricFlags.map((flag, i) => (
                  <tr key={i}>
                    <td className="py-2 pr-3 text-xs text-gray-500 whitespace-nowrap">
                      {typeof flag.timestamp_s === "number"
                        ? fmt(flag.timestamp_s)
                        : "—"}
                    </td>
                    <td className="py-2 pr-3">
                      <Pill variant="fail">
                        {String(flag.issue_type ?? "—")}
                      </Pill>
                    </td>
                    <td className="py-2 pr-3 text-gray-700 max-w-xs truncate">
                      {String(flag.text ?? "—")}
                    </td>
                    <td className="py-2 text-xs text-gray-500">
                      {String(flag.recommendation ?? "—")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {transcription.length > 0 && (
        <details className="group">
          <summary className="text-xs font-medium text-gray-500 cursor-pointer select-none hover:text-gray-700">
            Full transcript ({transcription.length} segments)
          </summary>
          <div className="mt-2 space-y-1 max-h-64 overflow-y-auto pr-1">
            {transcription.map((seg, i) => (
              <div key={i} className="flex gap-2 text-sm">
                <span className="text-xs text-gray-400 w-14 shrink-0 pt-0.5">
                  {fmt(seg.start)}
                </span>
                <span className="text-gray-700">{seg.text}</span>
              </div>
            ))}
          </div>
        </details>
      )}
    </SectionCard>
  );
}

function DiscoveryLicensing({ analysis }: { analysis: Analysis }) {
  const proLinks = [
    { name: "ASCAP", url: `https://www.ascap.com/repertory#ace/search/workID/` },
    { name: "BMI", url: `https://repertoire.bmi.com/` },
    { name: "SESAC", url: `https://www.sesac.com/repertory/search` },
  ];

  return (
    <SectionCard title="Discovery & Licensing">
      <p className="text-xs text-gray-500 mb-3">
        Search PRO repertory databases for this track&apos;s licensing status.
      </p>
      <div className="flex flex-wrap gap-2">
        {proLinks.map((pro) => (
          <a
            key={pro.name}
            href={pro.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center rounded-lg border border-gray-200 px-3 py-1.5 text-sm text-gray-700 hover:border-indigo-400 hover:text-indigo-700 transition-colors"
          >
            {pro.name} ↗
          </a>
        ))}
      </div>
      {analysis.source_url && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <a
            href={analysis.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-indigo-600 hover:underline"
          >
            View source on YouTube ↗
          </a>
        </div>
      )}
    </SectionCard>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default async function AnalysisPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const [analysis, labels] = await Promise.all([getAnalysis(id), getLabels()]);

  if (!analysis) notFound();

  const isTerminal =
    analysis.status === "complete" || analysis.status === "failed";
  const result = analysis.result_json ?? {};

  return (
    <>
      <Poller analysisId={id} status={analysis.status} />

      {/* Header */}
      <div className="mb-6">
        <Link
          href="/dashboard"
          className="text-xs text-gray-400 hover:text-gray-600 mb-2 block"
        >
          ← Dashboard
        </Link>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold text-gray-900">
              {analysis.title || analysis.artist || "Analysis"}
            </h1>
            <p className="text-sm text-gray-500 mt-0.5 truncate max-w-lg">
              {analysis.source_url}
            </p>
          </div>
          <LabelSelector
            analysisId={id}
            currentLabel={analysis.label}
            labels={labels}
          />
        </div>
      </div>

      {/* Processing state */}
      {!isTerminal && (
        <div className="bg-white rounded-xl border border-gray-200 p-8 text-center mb-6">
          <div className="inline-flex items-center gap-2 text-sm text-gray-600">
            <span className="animate-spin text-indigo-500 text-lg">⟳</span>
            <span>
              {analysis.status === "pending"
                ? "Queued — waiting for a worker…"
                : "Running analysis… this takes a few minutes."}
            </span>
          </div>
          <p className="mt-2 text-xs text-gray-400">
            This page polls automatically every 2 seconds.
          </p>
        </div>
      )}

      {/* Failed state */}
      {analysis.status === "failed" && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-5 mb-6">
          <p className="text-sm font-medium text-red-800">Analysis failed</p>
          {analysis.error && (
            <p className="mt-1 text-sm text-red-600">{analysis.error}</p>
          )}
        </div>
      )}

      {/* Report sections */}
      {analysis.status === "complete" && (
        <div className="space-y-4">
          <TrackOverview analysis={analysis} result={result} />
          <AuthenticityAudit result={result} />
          <SyncReadiness result={result} />
          <LyricsAudit result={result} />
          <DiscoveryLicensing analysis={analysis} />
        </div>
      )}
    </>
  );
}
