import { test, expect } from "@playwright/test";

const ANALYSIS_URL = "/analyses/test-abc-123";
// Lyric flag at timestamp_s=45, formatted as "0:45" by TimestampButton aria-label.
const SEEK_SECONDS = 45;

test.describe("Timestamp seek", () => {
  test.beforeEach(async ({ page }) => {
    // Abort YouTube IFrame API so it doesn't delay networkidle.
    await page.route("https://www.youtube.com/**", (route) => route.abort());
    await page.context().addCookies([
      { name: "access_token", value: "test-token", domain: "localhost", path: "/" },
    ]);
    await page.goto(ANALYSIS_URL);
    await page.waitForLoadState("networkidle");
  });

  test("timestamp buttons are rendered with correct aria-label", async ({
    page,
  }) => {
    const btn = page.getByRole("button", { name: /Seek to 0:45/ });
    await expect(btn).toBeVisible();
  });

  test("clicking a timestamp button invokes __syncSafeSeek with correct seconds", async ({
    page,
  }) => {
    // Wait for YouTubePlayer useEffect to register window.__syncSafeSeek.
    await page.waitForFunction(
      () => typeof window.__syncSafeSeek === "function"
    );

    // Wrap the seek bridge to capture the argument.
    await page.evaluate(() => {
      const orig = window.__syncSafeSeek;
      (window as unknown as Record<string, unknown>).__seekResult = undefined;
      window.__syncSafeSeek = (s: number) => {
        (window as unknown as Record<string, unknown>).__seekResult = s;
        orig?.(s);
      };
    });

    await page.getByRole("button", { name: /Seek to 0:45/ }).click();

    const seekResult = await page.evaluate(
      () => (window as unknown as Record<string, unknown>).__seekResult
    );
    expect(seekResult).toBe(SEEK_SECONDS);
  });
});
