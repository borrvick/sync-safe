import Link from "next/link";
import { apiFetch } from "@/app/lib/api";
import type { Analysis, PaginatedAnalyses } from "@/app/lib/definitions";
import { SubmitForm } from "./submit-form";

const STATUS_BADGE: Record<Analysis["status"], string> = {
  pending: "bg-yellow-100 text-yellow-800",
  running: "bg-blue-100 text-blue-800",
  complete: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
};

function StatusBadge({ status }: { status: Analysis["status"] }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_BADGE[status]}`}
    >
      {status}
    </span>
  );
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

async function getAnalyses(): Promise<PaginatedAnalyses | null> {
  const res = await apiFetch("/api/analyses/");
  if (!res.ok) return null;
  return res.json() as Promise<PaginatedAnalyses>;
}

export default async function DashboardPage() {
  const data = await getAnalyses();

  return (
    <>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold text-gray-900">Analyses</h1>
        <SubmitForm />
      </div>

      {!data || data.results.length === 0 ? (
        <div className="rounded-xl border border-dashed border-gray-300 bg-white px-6 py-12 text-center">
          <p className="text-sm text-gray-500">No analyses yet.</p>
          <p className="mt-1 text-xs text-gray-400">
            Click &ldquo;New analysis&rdquo; to get started.
          </p>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-100">
            <thead>
              <tr className="bg-gray-50">
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">
                  Track
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">
                  Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide hidden sm:table-cell">
                  Label
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide hidden md:table-cell">
                  Submitted
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {data.results.map((a) => (
                <tr key={a.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-3">
                    <Link
                      href={`/analyses/${a.id}`}
                      className="block focus:outline-none focus:underline"
                    >
                      <p className="text-sm font-medium text-gray-900 truncate max-w-xs hover:text-indigo-600">
                        {a.title || a.artist || "Untitled"}
                      </p>
                      <p className="text-xs text-gray-400 truncate max-w-xs">
                        {a.source_url}
                      </p>
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={a.status} />
                    {a.status === "failed" && a.error && (
                      <p className="mt-0.5 text-xs text-red-500 truncate max-w-[12rem]">
                        {a.error}
                      </p>
                    )}
                  </td>
                  <td className="px-4 py-3 hidden sm:table-cell text-sm text-gray-600">
                    {a.label || <span className="text-gray-300">—</span>}
                  </td>
                  <td className="px-4 py-3 hidden md:table-cell text-sm text-gray-500">
                    {formatDate(a.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {data.count > data.results.length && (
            <div className="px-4 py-3 border-t border-gray-100 text-xs text-gray-400">
              Showing {data.results.length} of {data.count}
            </div>
          )}
        </div>
      )}
    </>
  );
}
