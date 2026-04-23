import { z } from "zod";

export const LoginSchema = z.object({
  email: z.email({ error: "Enter a valid email address." }).trim(),
  password: z.string().min(1, { error: "Password is required." }),
});

export const RegisterSchema = z.object({
  email: z.email({ error: "Enter a valid email address." }).trim(),
  password: z
    .string()
    .min(8, { error: "Password must be at least 8 characters." })
    .regex(/[a-zA-Z]/, { error: "Password must contain at least one letter." })
    .regex(/[0-9]/, { error: "Password must contain at least one number." }),
});

export const ForgotPasswordSchema = z.object({
  email: z.email({ error: "Enter a valid email address." }).trim(),
});

export type FormState =
  | {
      errors?: Record<string, string[]>;
      message?: string;
    }
  | undefined;

export interface Analysis {
  id: string;
  source_url: string;
  title: string;
  artist: string;
  status: "pending" | "running" | "complete" | "failed";
  label: string;
  created_at: string;
  updated_at: string;
  result_json: Record<string, unknown> | null;
  error: string;
}

export interface PaginatedAnalyses {
  count: number;
  next: string | null;
  previous: string | null;
  results: Analysis[];
}
