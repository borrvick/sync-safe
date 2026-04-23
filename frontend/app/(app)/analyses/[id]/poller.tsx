"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

interface Props {
  analysisId: string;
  status: string;
}

export function Poller({ analysisId, status }: Props) {
  const router = useRouter();

  useEffect(() => {
    if (status === "complete" || status === "failed") return;

    const interval = setInterval(async () => {
      const res = await fetch(`/api/analyses/${analysisId}`);
      if (!res.ok) return;
      const data = (await res.json()) as { status: string };
      if (data.status === "complete" || data.status === "failed") {
        clearInterval(interval);
        router.refresh();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [analysisId, status, router]);

  return null;
}
