"use client";

export function TimestampButton({
  seconds,
  children,
}: {
  seconds: number;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={() => window.__syncSafeSeek?.(seconds)}
      className="text-indigo-500 hover:text-indigo-700 hover:underline focus:outline-none focus:underline cursor-pointer"
      aria-label={`Seek to ${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, "0")}`}
    >
      {children}
    </button>
  );
}
