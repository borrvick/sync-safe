import Link from "next/link";
import { logout } from "@/app/actions/auth";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200">
        <div className="mx-auto max-w-5xl px-4 flex items-center justify-between h-14">
          <Link
            href="/dashboard"
            className="text-sm font-semibold text-gray-900"
          >
            Sync-Safe
          </Link>
          <form action={logout}>
            <button
              type="submit"
              className="text-sm text-gray-500 hover:text-gray-900 transition-colors focus:outline-none focus:underline"
            >
              Sign out
            </button>
          </form>
        </div>
      </nav>
      <main className="mx-auto max-w-5xl px-4 py-8">{children}</main>
    </div>
  );
}
