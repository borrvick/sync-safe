"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

interface Props {
  analysisId: string;
  status: string;
}

const POLL_INITIAL_MS  = 2_000;  // start at 2 s
const POLL_MAX_MS      = 30_000; // cap at 30 s
const POLL_BACKOFF     = 1.5;    // multiply interval by 1.5× on each tick

export function Poller({ analysisId, status }: Props) {
  const router = useRouter();
  const [pollError, setPollError] = useState(false);
  const intervalMs = useRef(POLL_INITIAL_MS);
  const timeoutId  = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (status === "complete" || status === "failed") return;

    let cancelled = false;

    async function tick() {
      try {
        const res = await fetch(`/api/analyses/${analysisId}`);
        if (!res.ok) {
          if (!cancelled) setPollError(true);
          return;
        }
        const data = (await res.json()) as { status: string };
        if (data.status === "complete" || data.status === "failed") {
          router.refresh();
          return;
        }
      } catch {
        if (!cancelled) setPollError(true);
        return;
      }

      if (!cancelled) {
        // Back off on each tick up to the cap.
        intervalMs.current = Math.min(
          Math.round(intervalMs.current * POLL_BACKOFF),
          POLL_MAX_MS,
        );
        timeoutId.current = setTimeout(tick, intervalMs.current);
      }
    }

    timeoutId.current = setTimeout(tick, intervalMs.current);

    return () => {
      cancelled = true;
      if (timeoutId.current !== null) clearTimeout(timeoutId.current);
    };
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
