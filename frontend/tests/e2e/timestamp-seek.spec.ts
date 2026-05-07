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

  test("clicking a timestamp button dispatches seek-to-seconds event with correct seconds", async ({
    page,
  }) => {
    // Register listener before clicking so the event is never missed.
    const eventPromise = page.evaluate(() => {
      return new Promise<number>((resolve) => {
        window.addEventListener(
          "seek-to-seconds",
          (e) => resolve((e as CustomEvent<{ seconds: number }>).detail.seconds),
          { once: true }
        );
      });
    });

    await page.getByRole("button", { name: /Seek to 0:45/ }).click();

    const seconds = await eventPromise;
    expect(seconds).toBe(SEEK_SECONDS);
  });
});
