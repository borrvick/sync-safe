"use server";

import { redirect } from "next/navigation";
import {
  ForgotPasswordSchema,
  FormState,
  LoginSchema,
  RegisterSchema,
} from "@/app/lib/definitions";
import { clearSession, setSession } from "@/app/lib/session";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const SERVER_ERROR: FormState = {
  message: "Could not reach the server. Please try again.",
};

export async function login(
  _state: FormState,
  formData: FormData
): Promise<FormState> {
  const validated = LoginSchema.safeParse({
    email: formData.get("email"),
    password: formData.get("password"),
  });

  if (!validated.success) {
    return { errors: validated.error.flatten().fieldErrors };
  }

  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/api/auth/login/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(validated.data),
    });
  } catch {
    return SERVER_ERROR;
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    return {
      message:
        (body as { detail?: string }).detail ??
        "Invalid credentials. Please try again.",
    };
  }

  const { access, refresh } = (await res.json()) as {
    access: string;
    refresh: string;
  };
  await setSession(access, refresh);
  redirect("/dashboard");
}

export async function register(
  _state: FormState,
  formData: FormData
): Promise<FormState> {
  const validated = RegisterSchema.safeParse({
    email: formData.get("email"),
    password: formData.get("password"),
  });

  if (!validated.success) {
    return { errors: validated.error.flatten().fieldErrors };
  }

  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/api/auth/register/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(validated.data),
    });
  } catch {
    return SERVER_ERROR;
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail =
      (body as { email?: string[] }).email?.[0] ??
      "Registration failed. Please try again.";
    return { message: detail };
  }

  redirect("/login?registered=1");
}

export async function forgotPassword(
  _state: FormState,
  formData: FormData
): Promise<FormState> {
  const validated = ForgotPasswordSchema.safeParse({
    email: formData.get("email"),
  });

  if (!validated.success) {
    return { errors: validated.error.flatten().fieldErrors };
  }

  try {
    await fetch(`${BASE_URL}/api/auth/password/reset/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(validated.data),
    });
  } catch {
    // Network failure: still return success — enumeration-proof behaviour must
    // be preserved even when the backend is unreachable.
  }

  return { message: "ok" };
}

export async function logout() {
  await clearSession();
  redirect("/login");
}
