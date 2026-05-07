import { getAccessToken } from "./session";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface FetchOptions extends Omit<RequestInit, "headers"> {
  headers?: Record<string, string>;
}

export async function apiFetch(
  path: string,
  options: FetchOptions = {}
): Promise<Response> {
  const token = await getAccessToken();
  // Omit Content-Type for FormData — fetch sets it automatically with the correct boundary.
  const isFormData = options.body instanceof FormData;
  const headers: Record<string, string> = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...options.headers,
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  return fetch(`${BASE_URL}${path}`, { ...options, headers });
}
