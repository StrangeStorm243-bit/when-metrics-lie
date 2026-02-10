"use client";

/**
 * Auth token provider for Spectra frontend.
 *
 * When Clerk is configured, this component registers a global token provider
 * that the API client uses to attach Bearer tokens to requests.
 *
 * When Clerk is NOT configured, the token provider returns null and the
 * API client proceeds without auth headers (local dev mode).
 */

import { useAuth } from "@clerk/nextjs";
import { useEffect } from "react";
import { setTokenProvider } from "./api";

export function AuthTokenProvider({ children }: { children: React.ReactNode }) {
  let getToken: (() => Promise<string | null>) | null = null;

  try {
    // useAuth() will return default values when Clerk is not configured.
    const auth = useAuth();
    getToken = auth.getToken;
  } catch {
    // Clerk not available; proceed without auth.
  }

  useEffect(() => {
    if (getToken) {
      setTokenProvider(() => getToken());
    }
    return () => {
      setTokenProvider(null);
    };
  }, [getToken]);

  return <>{children}</>;
}
