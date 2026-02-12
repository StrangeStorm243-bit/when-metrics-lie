import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Clerk middleware for Spectra.
 *
 * When NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY is set, protects all routes
 * except /sign-in and /sign-up.
 *
 * When NOT set, this middleware is a no-op, preserving Phase 4 local-dev behavior.
 */

const clerkEnabled = !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

export default async function middleware(request: NextRequest) {
  if (!clerkEnabled) {
    return NextResponse.next();
  }

  // Dynamically import Clerk middleware only when configured
  const { clerkMiddleware, createRouteMatcher } = await import("@clerk/nextjs/server");

  const isPublicRoute = createRouteMatcher([
    "/sign-in(.*)",
    "/sign-up(.*)",
    "/runs(.*)",
  ]);

  // Create and invoke the Clerk middleware
  const handler = clerkMiddleware(async (auth, req) => {
    if (!isPublicRoute(req)) {
      await auth.protect();
    }
  });

  return handler(request, {} as any);
}

export const config = {
  matcher: [
    // Skip Next.js internals and static files
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    // Always run for API routes
    "/(api|trpc)(.*)",
  ],
};
