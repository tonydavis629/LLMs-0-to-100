"""Screenshot the three states of page.html for the computer-use slide.

Run from the repo root:
    uv run python slides/module_11_applications/source/computer_use_demo/capture.py

Each screenshot gets a red ring where the agent's next action lands. The ring's
position is read from the page, so the coordinates printed here are the real
pixel coordinates quoted on the slide.
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent          # this folder
OUT = HERE.parent.parent / "images"             # the module's images folder
# (state, CSS selector of the element the agent acts on next, or None)
STEPS = [("1", "#search"), ("2", "#order"), ("3", None)]

with sync_playwright() as p:
    browser = p.chromium.launch()
    # 800x500 CSS pixels, saved at 2x so the text stays sharp on the slide.
    page = browser.new_page(viewport={"width": 800, "height": 500}, device_scale_factor=2)
    for state, selector in STEPS:
        page.goto(f"file://{HERE / 'page.html'}?state={state}")
        if selector:
            box = page.locator(selector).bounding_box()
            x = round(box["x"] + box["width"] / 2)   # center of the target element
            y = round(box["y"] + box["height"] / 2)
            page.evaluate(
                "([x, y]) => document.body.insertAdjacentHTML('beforeend',"
                " `<div class='marker' style='left:${x}px;top:${y}px'></div>`)",
                [x, y],
            )
            print(f"state {state}: {selector} at ({x}, {y})")
        page.screenshot(path=str(OUT / f"computer_use_{state}.png"))
    browser.close()
