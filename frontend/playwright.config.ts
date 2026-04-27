import { defineConfig, devices } from "@playwright/test";

const MOCK_PORT = 4001;
const APP_PORT = 3001;

export default defineConfig({
  testDir: "tests/e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: `http://localhost:${APP_PORT}`,
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: `node tests/e2e/fixtures/mock-api.cjs ${MOCK_PORT}`,
      url: `http://localhost:${MOCK_PORT}/health`,
      timeout: 10_000,
      reuseExistingServer: !process.env.CI,
    },
    {
      command: "npm run dev",
      url: `http://localhost:${APP_PORT}`,
      timeout: 120_000,
      reuseExistingServer: !process.env.CI,
      env: {
        PORT: String(APP_PORT),
        NEXT_PUBLIC_API_URL: `http://localhost:${MOCK_PORT}`,
      },
    },
  ],
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
});
