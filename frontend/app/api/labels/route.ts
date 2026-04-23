import { apiFetch } from "@/app/lib/api";
import { NextResponse } from "next/server";

export async function GET() {
  const res = await apiFetch("/api/analyses/labels/");
  if (!res.ok) return NextResponse.json([], { status: res.status });
  return NextResponse.json(await res.json());
}
