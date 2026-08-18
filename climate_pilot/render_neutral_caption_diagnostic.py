"""
Renders climate_pilot/diagnostic_neutral_caption/posts/*/html/*.html into PNGs, same pattern as
render_climate_html_to_png.py -- run on the machine with a working Chromium binary, not the GPU
server. Baseline-only diagnostic (no engagement scales), so this is the trimmed-down equivalent
of that script for just this folder.

Usage:
    python3 climate_pilot/render_neutral_caption_diagnostic.py

Safe to re-run / resume: skips any PNG that already exists.
"""
import os
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

OUT_DIR = Path(__file__).resolve().parent
POSTS_DIR = OUT_DIR / "diagnostic_neutral_caption" / "posts"


def capture_screenshot(html_path: str, output_image: str):
    if Path(output_image).exists():
        print(f"  [SKIP] Already exists: {Path(output_image).name}")
        return

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=800,1200")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.binary_location = "/usr/bin/chromium"

    driver = webdriver.Chrome(options=chrome_options)
    try:
        file_url = "file://" + os.path.abspath(html_path)
        driver.get(file_url)
        time.sleep(2)
        driver.save_screenshot(output_image)
        print(f"  Screenshot saved as {output_image}")
    finally:
        driver.quit()


def main():
    for variant in ("correct", "incorrect"):
        html_dir = POSTS_DIR / variant / "html"
        png_dir = POSTS_DIR / variant / "PNGs"
        png_dir.mkdir(parents=True, exist_ok=True)
        html_files = sorted(html_dir.glob("*.html"))
        print(f"\n[{variant.upper()}] Processing {len(html_files)} file(s) from {html_dir}")
        for idx, html_file in enumerate(html_files):
            output_png = png_dir / (html_file.stem + ".png")
            print(f"  -> {html_file.name}")
            capture_screenshot(str(html_file), str(output_png))
            if idx % 10 == 0:
                time.sleep(3)
    print("\nDone.")


if __name__ == "__main__":
    main()
