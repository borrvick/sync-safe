"use server";

import { redirect } from "next/navigation";
import { z } from "zod";
import { apiFetch } from "@/app/lib/api";
import type { FormState } from "@/app/lib/definitions";

const ALLOWED_AUDIO_TYPES = new Set([".mp3", ".wav", ".flac", ".m4a"]);
const MAX_FILE_BYTES = 20 * 1024 * 1024; // 20 MB

const SubmitUrlSchema = z.object({
  source_url:  z.url({ error: "Enter a valid YouTube URL." }),
  title:       z.string().trim().optional(),
  artist:      z.string().trim().optional(),
  force_rerun: z.boolean().optional().default(false),
});

export async function submitAnalysis(
  _state: FormState,
  formData: FormData
): Promise<FormState> {
  const mode = formData.get("mode") as string;

  // ── File upload path ───────────────────────────────────────────────────────
  if (mode === "file") {
    const audioFile = formData.get("audio_file");
    if (!(audioFile instanceof File) || audioFile.size === 0) {
      return { message: "Please select an audio file." };
    }
    const ext = audioFile.name.slice(audioFile.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED_AUDIO_TYPES.has(ext)) {
      return { message: `Unsupported file type. Use: ${[...ALLOWED_AUDIO_TYPES].join(", ")}` };
    }
    if (audioFile.size > MAX_FILE_BYTES) {
      return { message: "File exceeds the 20 MB limit." };
    }

    const body = new FormData();
    body.append("audio_file", audioFile);
    body.append("title",  (formData.get("title")  as string) || "");
    body.append("artist", (formData.get("artist") as string) || "");
    body.append("force_rerun", formData.get("force_rerun") === "on" ? "true" : "false");

    let res: Response;
    try {
      res = await apiFetch("/api/analyses/", { method: "POST", body });
    } catch {
      return { message: "Could not reach the server. Please try again." };
    }

    if (!res.ok) {
      const json = await res.json().catch(() => ({}));
      return {
        message:
          (json as { detail?: string }).detail ??
          "Failed to submit. Please try again.",
      };
    }

    const { id } = (await res.json()) as { id: string };
    redirect(`/analyses/${id}`);
  }

  // ── URL path ───────────────────────────────────────────────────────────────
  const validated = SubmitUrlSchema.safeParse({
    source_url:  formData.get("source_url"),
    title:       formData.get("title") || undefined,
    artist:      formData.get("artist") || undefined,
    force_rerun: formData.get("force_rerun") === "on",
  });

  if (!validated.success) {
    return { errors: validated.error.flatten().fieldErrors };
  }

  let res: Response;
  try {
    res = await apiFetch("/api/analyses/", {
      method: "POST",
      body: JSON.stringify(validated.data),
    });
  } catch {
    return { message: "Could not reach the server. Please try again." };
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail =
      (body as { source_url?: string[] }).source_url?.[0] ??
      (body as { detail?: string }).detail ??
      "Failed to submit. Please try again.";
    return { message: detail };
  }

  const { id } = (await res.json()) as { id: string };
  redirect(`/analyses/${id}`);
}

export async function setLabel(analysisId: string, label: string) {
  let res: Response;
  try {
    res = await apiFetch(`/api/analyses/${analysisId}/label/`, {
      method: "PATCH",
      body: JSON.stringify({ label }),
    });
  } catch {
    throw new Error("Could not reach the server.");
  }

  if (!res.ok) {
    throw new Error("Failed to update label.");
  }
}
