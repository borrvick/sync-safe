import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    // Default server action body limit is 1 MB — raise it to handle audio file uploads up to 20 MB.
    serverActions: {
      bodySizeLimit: "21mb",
    },
  },
};

export default nextConfig;
