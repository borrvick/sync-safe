"use client";

import { useTransition, useState } from "react";
import { setLabel } from "@/app/actions/analyses";

interface TrackLabel {
  slug: string;
  name: string;
  description: string;
  sort_order: number;
}

interface Props {
  analysisId: string;
  currentLabel: string;
  labels: TrackLabel[];
}

export function LabelSelector({ analysisId, currentLabel, labels }: Props) {
  const [selected, setSelected] = useState(currentLabel);
  const [isPending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);

  function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const value = e.target.value;
    setSelected(value);
    setError(null);
    startTransition(async () => {
      try {
        await setLabel(analysisId, value);
      } catch {
        setError("Failed to save label.");
      }
    });
  }

  return (
    <div>
      <label
        htmlFor="label-select"
        className="block text-xs font-medium text-gray-500 mb-1"
      >
        Sync category
      </label>
      <select
        id="label-select"
        value={selected}
        onChange={handleChange}
        disabled={isPending}
        className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:opacity-60"
        aria-label="Sync category label"
      >
        <option value="">— None —</option>
        {labels.map((l) => (
          <option key={l.slug} value={l.slug}>
            {l.name}
          </option>
        ))}
      </select>
      {isPending && (
        <span className="ml-2 text-xs text-gray-400">Saving…</span>
      )}
      {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
    </div>
  );
}
