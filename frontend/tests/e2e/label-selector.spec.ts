import { test, expect } from "@playwright/test";

const ANALYSIS_URL = "/analyses/test-abc-123";
const ERROR_ANALYSIS_URL = "/analyses/error-label-id";

test.describe("Label selector", () => {
  test.beforeEach(async ({ page }) => {
    await page.context().addCookies([
      { name: "access_token", value: "test-token", domain: "localhost", path: "/" },
    ]);
  });

  test("renders with the current label value", async ({ page }) => {
    await page.goto(ANALYSIS_URL);
    await page.waitForLoadState("networkidle");
    const select = page.locator("#label-select");
    await expect(select).toBeVisible();
    await expect(select).toHaveValue("");
  });

  test("shows Saving… while the Server Action is in flight", async ({
    page,
  }) => {
    await page.goto(ANALYSIS_URL);
    await page.waitForLoadState("networkidle");

    // Slow down the Server Action response so the pending state is observable.
    await page.route("/_next/action*", async (route) => {
      await new Promise((r) => setTimeout(r, 400));
      await route.continue();
    });

    await page.locator("#label-select").selectOption("sync-ready");
    await expect(page.getByText("Saving…")).toBeVisible();
    await expect(page.getByText("Saving…")).not.toBeVisible({ timeout: 5_000 });
  });

  test("persists label change without error when server succeeds", async ({
    page,
  }) => {
    await page.goto(ANALYSIS_URL);
    await page.waitForLoadState("networkidle");

    await page.locator("#label-select").selectOption("sync-ready");

    // Error message must never appear.
    await expect(
      page.getByText("Failed to save label.")
    ).not.toBeVisible({ timeout: 5_000 });
    // Optimistic update stays selected.
    await expect(page.locator("#label-select")).toHaveValue("sync-ready");
  });

  test("shows error message when the server returns an error", async ({
    page,
  }) => {
    await page.goto(ERROR_ANALYSIS_URL);
    await page.waitForLoadState("networkidle");

    await page.locator("#label-select").selectOption("sync-ready");

    await expect(
      page.getByText("Failed to save label.")
    ).toBeVisible({ timeout: 10_000 });
  });
});
