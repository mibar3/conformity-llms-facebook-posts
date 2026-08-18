"""
Renders the HTML files produced by generate_climate_stimuli.py into PNG screenshots via a
headless Chromium + Selenium, following the exact same pattern already used for the main study
(utils/html_to_png.ipynb) -- run this wherever you normally run that notebook (a machine with a
working Chromium binary; per that notebook's own note, the GPU server does not have one).

Usage:
    python3 climate_pilot/render_climate_html_to_png.py

Safe to re-run / resume: skips any PNG that already exists, same as the main study's renderer.
"""
import os
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

OUT_DIR = Path(__file__).resolve().parent
POSTS_DIR = OUT_DIR / "posts"
SCALE_VALUES = [10, 100, 1000, 10000, 100000, 1000000]


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


def process_folder(label: str, html_dir: Path, png_dir: Path):
    if not html_dir.exists():
        print(f"[SKIP] Directory not found: {html_dir}")
        return
    html_files = sorted(html_dir.glob("*.html"))
    if not html_files:
        print(f"[SKIP] No HTML files found in: {html_dir}")
        return

    png_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{label.upper()}] Processing {len(html_files)} file(s) from {html_dir}")
    for idx, html_file in enumerate(html_files):
        output_png = png_dir / (html_file.stem + ".png")
        print(f"  -> {html_file.name}")
        capture_screenshot(str(html_file), str(output_png))
        if idx % 10 == 0:
            time.sleep(3)  # brief pause every 10 files, same as the main study's renderer


def main():
    for variant in ("correct", "incorrect"):
        # Baseline (0 engagement)
        process_folder(
            label=f"{variant}/baseline",
            html_dir=POSTS_DIR / variant / "html",
            png_dir=POSTS_DIR / variant / "PNGs",
        )
        # 6 engagement scales, metrics/realistic condition
        for scale in SCALE_VALUES:
            process_folder(
                label=f"{variant}/metrics/realistic/{scale}",
                html_dir=POSTS_DIR / variant / "html" / "metrics" / "realistic" / str(scale),
                png_dir=POSTS_DIR / variant / "PNGs" / "metrics" / "realistic" / str(scale),
            )
    print("\nDone.")


if __name__ == "__main__":
    main()
