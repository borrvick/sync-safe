import { apiFetch } from "@/app/lib/api";
import { NextResponse } from "next/server";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const res = await apiFetch(`/api/analyses/${id}/`);
  if (!res.ok) {
    return NextResponse.json({ error: "not found" }, { status: res.status });
  }
  return NextResponse.json(await res.json());
}
