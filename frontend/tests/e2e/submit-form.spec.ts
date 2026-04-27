import { test, expect } from "@playwright/test";

const ANALYSIS_ID = "test-abc-123";

test.describe("Submit form", () => {
  test.beforeEach(async ({ page }) => {
    await page.context().addCookies([
      { name: "access_token", value: "test-token", domain: "localhost", path: "/" },
    ]);
    await page.goto("/dashboard");
    await page.waitForLoadState("networkidle");
  });

  test("opens when 'New analysis' is clicked", async ({ page }) => {
    await page.getByRole("button", { name: "New analysis" }).click();
    await expect(page.getByRole("heading", { name: "New analysis" })).toBeVisible();
    await expect(page.locator("#source_url")).toBeVisible();
  });

  test("closes when Cancel is clicked", async ({ page }) => {
    await page.getByRole("button", { name: "New analysis" }).click();
    await expect(page.locator("#source_url")).toBeVisible();
    await page.getByRole("button", { name: "Cancel" }).click();
    await expect(page.locator("#source_url")).not.toBeVisible();
  });

  test("navigates to report page after successful submission", async ({ page }) => {
    await page.getByRole("button", { name: "New analysis" }).click();
    await page.locator("#source_url").fill(
      "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    );
    await page.getByRole("button", { name: "Analyze" }).click();
    await page.waitForURL(
      (url) => url.pathname === `/analyses/${ANALYSIS_ID}`,
      { timeout: 15_000 }
    );
  });

  test("shows server error when URL is not a YouTube link", async ({ page }) => {
    await page.getByRole("button", { name: "New analysis" }).click();
    // A valid URL that isn't YouTube — passes browser type="url" validation
    // but the mock server returns 400 for non-YouTube domains.
    await page.locator("#source_url").fill("https://vimeo.com/123456789");
    await page.getByRole("button", { name: "Analyze" }).click();
    await expect(
      page.getByText("This URL is not supported. Use a YouTube link.")
    ).toBeVisible({ timeout: 10_000 });
  });
});
