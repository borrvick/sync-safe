"use client";

import { useActionState, useState, useRef } from "react";
import { submitAnalysis } from "@/app/actions/analyses";
import type { FormState } from "@/app/lib/definitions";

type Mode = "url" | "file";

const MAX_FILE_MB = 20;
const ALLOWED_EXTS = [".mp3", ".wav", ".flac", ".m4a"];

export function SubmitForm() {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<Mode>("url");
  const [fileError, setFileError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const [state, action, pending] = useActionState<FormState, FormData>(
    submitAnalysis,
    undefined
  );

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    setFileError(null);
    setFileName(null);
    if (!file) return;
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      setFileError(`Unsupported type. Use: ${ALLOWED_EXTS.join(", ")}`);
      e.target.value = "";
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setFileError(`File exceeds the ${MAX_FILE_MB} MB limit.`);
      e.target.value = "";
      return;
    }
    setFileName(file.name);
  }

  function switchMode(next: Mode) {
    setMode(next);
    setFileError(null);
    setFileName(null);
    if (fileRef.current) fileRef.current.value = "";
  }

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

      {/* Mode tabs */}
      <div className="flex gap-1 mb-4 bg-gray-100 rounded-lg p-1 w-fit">
        <button
          type="button"
          onClick={() => switchMode("url")}
          className={`text-xs font-medium px-3 py-1.5 rounded-md transition-colors ${
            mode === "url"
              ? "bg-white text-gray-900 shadow-sm"
              : "text-gray-500 hover:text-gray-700"
          }`}
        >
          YouTube URL
        </button>
        <button
          type="button"
          onClick={() => switchMode("file")}
          className={`text-xs font-medium px-3 py-1.5 rounded-md transition-colors ${
            mode === "file"
              ? "bg-white text-gray-900 shadow-sm"
              : "text-gray-500 hover:text-gray-700"
          }`}
        >
          Upload file
        </button>
      </div>

      {state?.message && (
        <p className="mb-3 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
          {state.message}
        </p>
      )}

      <form action={action} className="space-y-3">
        <input type="hidden" name="mode" value={mode} />

        {mode === "url" ? (
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
        ) : (
          <div>
            <label
              htmlFor="audio_file"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Audio file <span className="text-red-500">*</span>
              <span className="ml-1 font-normal text-gray-400">
                ({ALLOWED_EXTS.join(", ")}, max {MAX_FILE_MB} MB)
              </span>
            </label>
            <input
              id="audio_file"
              name="audio_file"
              type="file"
              accept={ALLOWED_EXTS.join(",")}
              ref={fileRef}
              onChange={handleFileChange}
              className="w-full text-sm text-gray-700 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
            />
            {fileError && (
              <p className="mt-1 text-xs text-red-600">{fileError}</p>
            )}
            {fileName && !fileError && (
              <p className="mt-1 text-xs text-gray-500">{fileName}</p>
            )}
          </div>
        )}

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

        <div className="flex items-center gap-2 pt-1">
          <input
            id="force_rerun"
            name="force_rerun"
            type="checkbox"
            className="h-3.5 w-3.5 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
          />
          <label htmlFor="force_rerun" className="text-xs text-gray-500 select-none">
            Force re-run (ignore cached result)
          </label>
        </div>

        <div className="flex gap-2 pt-1">
          <button
            type="submit"
            disabled={pending || (mode === "file" && !!fileError)}
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
