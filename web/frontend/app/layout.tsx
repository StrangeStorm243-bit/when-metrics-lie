import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Spectra",
  description: "Scenario-first evaluation engine for machine learning models",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <header style={{ borderBottom: "1px solid #e0e0e0", padding: "1rem 2rem" }}>
          <nav style={{ display: "flex", gap: "2rem", alignItems: "center" }}>
            <Link href="/" style={{ fontSize: "1.25rem", fontWeight: "bold", textDecoration: "none", color: "inherit" }}>
              Spectra
            </Link>
            <Link href="/" style={{ textDecoration: "none", color: "inherit" }}>
              Home
            </Link>
            <Link href="/new" style={{ textDecoration: "none", color: "inherit" }}>
              New Experiment
            </Link>
          </nav>
        </header>
        <main style={{ padding: "2rem" }}>
          {children}
        </main>
      </body>
    </html>
  );
}

