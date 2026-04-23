"use client";

import { useActionState, useState } from "react";
import { submitAnalysis } from "@/app/actions/analyses";
import type { FormState } from "@/app/lib/definitions";

export function SubmitForm() {
  const [open, setOpen] = useState(false);
  const [state, action, pending] = useActionState<FormState, FormData>(
    submitAnalysis,
    undefined
  );

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-lg px-4 py-2 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
      >
        New analysis
      </button>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-900">New analysis</h2>
        <button
          onClick={() => setOpen(false)}
          aria-label="Close"
          className="text-gray-400 hover:text-gray-600 text-lg leading-none"
        >
          ×
        </button>
      </div>

      {state?.message && (
        <p className="mb-3 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
          {state.message}
        </p>
      )}

      <form action={action} className="space-y-3">
        <div>
          <label
            htmlFor="source_url"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            YouTube URL <span className="text-red-500">*</span>
          </label>
          <input
            id="source_url"
            name="source_url"
            type="url"
            required
            placeholder="https://www.youtube.com/watch?v=..."
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
          {state?.errors?.source_url && (
            <p className="mt-1 text-xs text-red-600">
              {state.errors.source_url[0]}
            </p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label
              htmlFor="title"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Title
            </label>
            <input
              id="title"
              name="title"
              type="text"
              placeholder="Optional"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>
          <div>
            <label
              htmlFor="artist"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Artist
            </label>
            <input
              id="artist"
              name="artist"
              type="text"
              placeholder="Optional"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>
        </div>

        <div className="flex gap-2 pt-1">
          <button
            type="submit"
            disabled={pending}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white text-sm font-medium rounded-lg px-4 py-2 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            {pending ? "Submitting…" : "Analyze"}
          </button>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="text-sm text-gray-500 hover:text-gray-700 px-3 py-2"
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}
