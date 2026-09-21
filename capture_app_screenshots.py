from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
SCREENSHOTS = ROOT / "screenshots"
SCREENSHOTS.mkdir(exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1200})
    page.goto("http://localhost:8501", timeout=120000)
    page.wait_for_timeout(15000)

    page.screenshot(path=SCREENSHOTS / "01_app_home.png", full_page=True)

    # Set sample input values to generate a real prediction result.
    number_inputs = page.locator("input[type='number']")
    number_inputs.nth(0).fill("2500")
    number_inputs.nth(1).fill("4")
    number_inputs.nth(2).fill("3")

    page.locator("button:has-text('Predict Price')").click()
    page.wait_for_timeout(20000)
    page.screenshot(path=SCREENSHOTS / "02_prediction_result.png", full_page=True)

    print(f"Saved screenshots to: {SCREENSHOTS}")
    browser.close()
