"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

interface Props {
  analysisId: string;
  status: string;
}

export function Poller({ analysisId, status }: Props) {
  const router = useRouter();
  const [pollError, setPollError] = useState(false);

  useEffect(() => {
    if (status === "complete" || status === "failed") return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/analyses/${analysisId}`);
        if (!res.ok) {
          // Non-2xx from the proxy route — stop polling and surface the error.
          clearInterval(interval);
          setPollError(true);
          return;
        }
        const data = (await res.json()) as { status: string };
        if (data.status === "complete" || data.status === "failed") {
          clearInterval(interval);
          router.refresh();
        }
      } catch {
        // Network failure — stop polling rather than silently retrying forever.
        clearInterval(interval);
        setPollError(true);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [analysisId, status, router]);

  if (pollError) {
    return (
      <p className="mb-4 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
        Lost connection while waiting for results.{" "}
        <button
          onClick={() => router.refresh()}
          className="underline hover:no-underline"
        >
          Refresh
        </button>
      </p>
    );
  }

  return null;
}
