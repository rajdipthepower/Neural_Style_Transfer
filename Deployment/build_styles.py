"""
build_styles.py

Prepares style metadata and thumbnails for the AdaIN Style Transfer web app:
1. Scans styles/portrait, styles/landscape, and styles/abstract for image files.
2. Validates style IDs (lowercase, alphanumeric, underscores only; no duplicates).
3. Generates 200x200 center-cropped thumbnails in styles/thumbs/.
4. Generates styles/styles.json sorted by category and name.
5. Prints summary counts and warns about images below 1280px resolution.
"""

import json
import re
import sys
from pathlib import Path
from PIL import Image, ImageOps

# Project directory structure
ROOT_DIR = Path(__file__).resolve().parent
STYLES_DIR = ROOT_DIR / "styles"
THUMBS_DIR = STYLES_DIR / "thumbs"
JSON_OUTPUT_PATH = STYLES_DIR / "styles.json"

# Target categories to scan
CATEGORIES = ["portrait", "landscape", "abstract"]

# Supported image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp",".jfif"}

# ID format rule: lowercase letters, numbers, and underscores only
ID_PATTERN = re.compile(r"^[a-z0-9_]+$")


def build():
    # Ensure thumbnail output folder exists
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    seen_ids = {}  # Tracks {style_id: filepath} to detect duplicates
    styles_list = []
    category_counts = {cat: 0 for cat in CATEGORIES}
    warnings = []

    # Get Pillow resampling filter safely (compatible with both old & new Pillow)
    resample_filter = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)

    print("Scanning style directories...\n")

    for category in CATEGORIES:
        category_dir = STYLES_DIR / category
        if not category_dir.is_dir():
            print(f"[INFO] Skipping '{category}': folder does not exist.")
            continue

        for file_path in category_dir.iterdir():
            # Check for supported image extensions
            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            style_id = file_path.stem.lower()

            # 1. Validate ID format (only a-z, 0-9, and _)
            if not ID_PATTERN.match(style_id):
                sys.exit(
                    f"\n[ERROR] Invalid style ID '{style_id}' in {file_path}.\n"
                    f"IDs must only contain lowercase letters (a-z), numbers (0-9), and underscores (_)."
                )

            # 2. Check for duplicate IDs across all folders
            if style_id in seen_ids:
                sys.exit(
                    f"\n[ERROR] Duplicate style ID '{style_id}' found!\n"
                    f"File 1: {seen_ids[style_id]}\n"
                    f"File 2: {file_path}\n"
                    f"Each style must have a completely unique filename stem."
                )
            seen_ids[style_id] = file_path

            # 3. Create clean Title Case name (e.g., 'starry_night' -> 'Starry Night')
            style_name = style_id.replace("_", " ").title()

            # 4. Open and process the source image
            try:
                with Image.open(file_path) as img:
                    # Fix orientation based on EXIF tags
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")

                    width, height = img.size

                    # Check if the short edge is below 1280px for the 'broad' preset
                    short_edge = min(width, height)
                    if short_edge < 1280:
                        warnings.append(
                            f"WARNING: '{style_id}' ({category}) short edge is {short_edge}px "
                            f"(recommended: at least 1280px for high-res 'broad' scale)."
                        )

                    # Create 200x200 center-cropped thumbnail
                    thumb = ImageOps.fit(
                        img,
                        (200, 200),
                        method=resample_filter,
                        centering=(0.5, 0.5)
                    )

                    thumb_path = THUMBS_DIR / f"{style_id}.jpg"
                    thumb.save(thumb_path, format="JPEG", quality=85)

            except Exception as e:
                sys.exit(f"\n[ERROR] Failed to process image '{file_path}': {e}")

            # 5. Record style metadata
            styles_list.append({
                "id": style_id,
                "name": style_name,
                "category": category,
                "thumb": f"/styles/thumbs/{style_id}.jpg"
            })
            category_counts[category] += 1

    # 6. Sort styles: first by category, then alphabetically by name
    styles_list.sort(key=lambda s: (s["category"], s["name"]))

    # 7. Write styles/styles.json
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(styles_list, f, indent=2)

    # 8. Print summary
    print("=" * 45)
    print("STYLE BUILD SUMMARY")
    print("=" * 45)
    for cat in CATEGORIES:
        print(f"  - {cat.capitalize():<12}: {category_counts[cat]} styles")
    print(f"  - Total Styles: {len(styles_list)}")
    print(f"\nMetadata written to: {JSON_OUTPUT_PATH}")
    print(f"Thumbnails saved to: {THUMBS_DIR}")

    if warnings:
        print("\n" + "!" * 45)
        print("RESOLUTION WARNINGS:")
        print("!" * 45)
        for w in warnings:
            print(f"  {w}")
    else:
        print("\nAll image resolutions meet the 1280px threshold.")
    print("=" * 45)


if __name__ == "__main__":
    build()