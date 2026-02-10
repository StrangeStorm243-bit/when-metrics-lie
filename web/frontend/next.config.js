/** @type {import('next').NextConfig} */
const nextConfig = {
  // Clerk sign-in/sign-up routes
  env: {
    NEXT_PUBLIC_CLERK_SIGN_IN_URL: "/sign-in",
    NEXT_PUBLIC_CLERK_SIGN_UP_URL: "/sign-up",
  },
}

module.exports = nextConfig
