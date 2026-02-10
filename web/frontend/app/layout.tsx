import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Spectra",
  description: "Scenario-first evaluation engine for machine learning models",
};

/**
 * Conditionally imports Clerk components only when NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY is set.
 * This preserves Phase 4 local-dev behavior (no auth, no Clerk dependency at runtime).
 */
const clerkEnabled = !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Conditionally load Clerk and AuthTokenProvider
  let ClerkProvider: React.ComponentType<{ children: React.ReactNode }> | null = null;
  let UserButton: React.ComponentType | null = null;
  let SignedIn: React.ComponentType<{ children: React.ReactNode }> | null = null;
  let AuthTokenProvider: React.ComponentType<{ children: React.ReactNode }> | null = null;

  if (clerkEnabled) {
    const clerk = await import("@clerk/nextjs");
    ClerkProvider = clerk.ClerkProvider;
    UserButton = clerk.UserButton;
    SignedIn = clerk.SignedIn;
    const authMod = await import("@/lib/auth");
    AuthTokenProvider = authMod.AuthTokenProvider;
  }

  const nav = (
    <header style={{ borderBottom: "1px solid #e0e0e0", padding: "1rem 2rem" }}>
      <nav style={{ display: "flex", gap: "2rem", alignItems: "center" }}>
        <Link href="/" style={{ fontSize: "1.25rem", fontWeight: "bold", textDecoration: "none", color: "inherit" }}>
          Spectra
        </Link>
        <Link href="/" style={{ textDecoration: "none", color: "inherit" }}>
          Home
        </Link>
        <Link href="/new" style={{ textDecoration: "none", color: "inherit" }}>
          New
        </Link>
        <Link href="/compare" style={{ textDecoration: "none", color: "inherit" }}>
          Compare
        </Link>
        <Link href="/assistant" style={{ textDecoration: "none", color: "inherit" }}>
          Assistant
        </Link>
        {SignedIn && UserButton && (
          <div style={{ marginLeft: "auto" }}>
            <SignedIn>
              <UserButton />
            </SignedIn>
          </div>
        )}
      </nav>
    </header>
  );

  const body = (
    <html lang="en">
      <body>
        {nav}
        <main style={{ padding: "2rem" }}>
          {AuthTokenProvider ? <AuthTokenProvider>{children}</AuthTokenProvider> : children}
        </main>
      </body>
    </html>
  );

  if (ClerkProvider) {
    return <ClerkProvider>{body}</ClerkProvider>;
  }

  return body;
}
