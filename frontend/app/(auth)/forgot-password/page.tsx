"use client";

import { useActionState } from "react";
import Link from "next/link";
import { forgotPassword } from "@/app/actions/auth";
import type { FormState } from "@/app/lib/definitions";

export default function ForgotPasswordPage() {
  const [state, action, pending] = useActionState<FormState, FormData>(
    forgotPassword,
    undefined
  );

  const sent = state?.message === "ok";

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8">
      <h1 className="text-2xl font-semibold text-gray-900 mb-1">
        Reset password
      </h1>
      <p className="text-sm text-gray-500 mb-6">
        Enter your email and we'll send reset instructions.
      </p>

      {sent ? (
        <div className="text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg px-3 py-3">
          If that email is registered you'll receive a reset link shortly.
        </div>
      ) : (
        <form action={action} className="space-y-4">
          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Email
            </label>
            <input
              id="email"
              name="email"
              type="email"
              autoComplete="email"
              required
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              placeholder="you@example.com"
            />
            {state?.errors?.email && (
              <p className="mt-1 text-xs text-red-600">
                {state.errors.email[0]}
              </p>
            )}
          </div>

          <button
            type="submit"
            disabled={pending}
            className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white text-sm font-medium rounded-lg px-4 py-2.5 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            {pending ? "Sending…" : "Send reset link"}
          </button>
        </form>
      )}

      <p className="mt-5 text-center text-sm text-gray-500">
        <Link href="/login" className="text-indigo-600 hover:underline">
          Back to sign in
        </Link>
      </p>
    </div>
  );
}
