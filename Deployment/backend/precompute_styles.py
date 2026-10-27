"""
precompute_styles.py

Precomputes VGG-19 AdaIN mean and std statistics for all default styles across
all supported resolutions (128, 512, 1280).

Saves results to styles/style_stats.pt.
NOTE: Re-run this script whenever styles are added, removed, or modified.
"""

import json
import sys
import time
from pathlib import Path
import torch

# Add backend directory to sys.path so we can import inference
ROOT_DIR = Path(__file__).resolve().parent.parent


sys.path.append(str(ROOT_DIR / "backend"))

from inference import get_style_stats

STYLES_DIR = ROOT_DIR / "styles"
STYLES_JSON_PATH = STYLES_DIR / "styles.json"
STATS_OUTPUT_PATH = STYLES_DIR / "style_stats.pt"

TARGET_SIZES = (128, 512, 1280)
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".jfif")


def resolve_style_path(style_id, category):
    """Finds image path for a style under styles/<category>/<id>.<ext>."""
    cat_dir = STYLES_DIR / category
    if not cat_dir.is_dir():
        return None
    for ext in VALID_EXTS:
        candidate = cat_dir / f"{style_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def main():
    if not STYLES_JSON_PATH.is_file():
        sys.exit(f"[ERROR] styles.json not found at {STYLES_JSON_PATH}. Run build_styles.py first.")

    with open(STYLES_JSON_PATH, "r", encoding="utf-8") as f:
        styles = json.load(f)

    if not styles:
        sys.exit("[ERROR] styles.json is empty. No styles to precompute.")

    total_styles = len(styles)
    stats_dict = {}

    print(f"Starting precomputation for {total_styles} styles across sizes {TARGET_SIZES}...")
    start_total_time = time.time()

    for idx, item in enumerate(styles, start=1):
        style_id = item["id"]
        category = item.get("category", "")
        style_path = resolve_style_path(style_id, category)

        if not style_path:
            print(f"[{idx}/{total_styles}] [WARNING] File for '{style_id}' in '{category}' not found. Skipping.")
            continue

        style_start = time.time()
        for size in TARGET_SIZES:
            # Compute stats on active device (GPU or CPU)
            mean, std = get_style_stats(str(style_path), size=size)

            # Store on CPU to keep the .pt file device-agnostic and lightweight
            key = f"{style_id}|{size}"
            stats_dict[key] = (mean.cpu(), std.cpu())

        style_elapsed = time.time() - style_start
        print(f"[{idx}/{total_styles}] Precomputed '{style_id}' ({category}) in {style_elapsed:.2f}s")

    # Save to disk
    torch.save(stats_dict, STATS_OUTPUT_PATH)
    total_elapsed = time.time() - start_total_time

    print("\n" + "=" * 50)
    print(f"Precomputed {len(stats_dict)} stat combinations.")
    print(f"Saved to: {STATS_OUTPUT_PATH.resolve()}")
    print(f"Total time elapsed: {total_elapsed:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    main()