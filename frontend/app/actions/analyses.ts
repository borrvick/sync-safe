"use server";

import { redirect } from "next/navigation";
import { z } from "zod";
import { apiFetch } from "@/app/lib/api";
import type { FormState } from "@/app/lib/definitions";

const SubmitSchema = z.object({
  source_url: z.url({ error: "Enter a valid YouTube URL." }),
  title: z.string().trim().optional(),
  artist: z.string().trim().optional(),
});

export async function submitAnalysis(
  _state: FormState,
  formData: FormData
): Promise<FormState> {
  const validated = SubmitSchema.safeParse({
    source_url: formData.get("source_url"),
    title: formData.get("title") || undefined,
    artist: formData.get("artist") || undefined,
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
