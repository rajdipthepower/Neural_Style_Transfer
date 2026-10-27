import json
import sys
import time
from pathlib import Path
import psutil
import streamlit as st
import torch

# Add backend directory to sys.path so inference and models load once at import
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "backend"))
import inference

# Cache precomputed style statistics in memory
@st.cache_resource
def load_all_style_stats():
    stats_path = ROOT_DIR / "styles" / "style_stats.pt"
    return torch.load(stats_path, map_location=inference.device)


def main():
    st.title("DeepArt - Smoke Test")

    # 1. Content image uploader
    content_file = st.file_uploader(
        "Upload Content Image", 
        type=["jpg", "jpeg", "jfif", "png", "webp"]
    )

    # 2. Read available style IDs from styles/styles.json
    styles_json_path = ROOT_DIR / "styles" / "styles.json"
    if not styles_json_path.is_file():
        st.error(f"Missing {styles_json_path}. Run build_styles.py first.")
        return

    with open(styles_json_path, "r", encoding="utf-8") as f:
        styles_data = json.load(f)

    style_ids = [item["id"] for item in styles_data]
    selected_style_id = st.selectbox("Choose a Style ID", style_ids)

    # 3. Stylize Button
    if st.button("Stylize"):
        if content_file is None:
            st.warning("Please upload a content image first.")
            return

        stats_dict = load_all_style_stats()
        key = f"{selected_style_id}|512"

        if key not in stats_dict:
            st.error(f"Key '{key}' not found in styles/style_stats.pt. Run precompute_styles.py.")
            return

        # Prepare style stats on the active device
        mean, std = stats_dict[key]
        style_stats = (mean.to(inference.device), std.to(inference.device))

        # Run inference and measure time
        start_time = time.time()
        result_image = inference.stylize(content_file, style_stats, alpha=0.8)
        elapsed_sec = time.time() - start_time

        # Measure current process resident memory (RAM) in MB
        process = psutil.Process()
        memory_mb = process.memory_info().rss // (1024 * 1024)

        # 4. Display result and benchmark caption
        caption = f"Completed in {elapsed_sec:.2f}s | Process Memory: {memory_mb} MB"
        st.image(result_image, caption=caption)


if __name__ == "__main__":
    main()