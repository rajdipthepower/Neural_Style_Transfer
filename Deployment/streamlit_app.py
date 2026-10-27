import base64
import gc
import io
import json
import sys
import threading
import time
from pathlib import Path
from PIL import Image, ImageOps
import streamlit as st
import torch

# -----------------------------------------------------------------------------
# Configuration & Backend Setup
# -----------------------------------------------------------------------------

st.set_page_config(page_title="DeepArt", page_icon="🎨", layout="centered")

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
STYLES_DIR = ROOT_DIR / "styles"
STYLES_JSON_PATH = STYLES_DIR / "styles.json"
STATS_PT_PATH = STYLES_DIR / "style_stats.pt"
BG_IMAGE_PATH = ROOT_DIR / "frontend" / "static" / "bg.jpg"

sys.path.append(str(BACKEND_DIR))
import inference
from inference import device, get_style_stats, stylize, stylize_blend

SCALE_MAP = {
    "Fine": 128,
    "Medium": 512,
    "Broad": 1280
}

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# -----------------------------------------------------------------------------
# Cached Resources
# -----------------------------------------------------------------------------

@st.cache_resource
def get_stylize_lock():
    return threading.Lock()


@st.cache_resource
def load_precomputed_stats():
    if STATS_PT_PATH.is_file():
        try:
            return torch.load(STATS_PT_PATH, map_location=device)
        except Exception as e:
            print(f"[WARNING] Could not load style_stats.pt: {e}")
    return {}


@st.cache_data
def get_base64_bg():
    if BG_IMAGE_PATH.is_file():
        with open(BG_IMAGE_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


@st.cache_data
def load_styles_metadata():
    if STYLES_JSON_PATH.is_file():
        with open(STYLES_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# -----------------------------------------------------------------------------
# Custom Styling (CSS Injection)
# -----------------------------------------------------------------------------

def inject_custom_css():
    bg_base64 = get_base64_bg()
    bg_style = ""
    if bg_base64:
        bg_style = f"""
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)),
                        url("data:image/jpeg;base64,{bg_base64}") no-repeat center center fixed !important;
            background-size: cover !important;
        }}
        """

    custom_css = f"""
    <style>
    /* Hide Streamlit default chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    {bg_style}

    /* 1. SOLID FROSTED GLASS MAIN CARD */
    .block-container {{
        background: rgba(15, 23, 42, 0.88) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7) !important;
        padding: 2.5rem 2rem !important;
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
        max-width: 800px !important;
    }}

    /* 2. UNIVERSAL WHITE TEXT (For headings and general text on the dark card) */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    [data-testid="stMarkdownContainer"] p,
    div[data-testid="stWidgetLabel"] p {{
        color: #ffffff !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.8) !important;
    }}

    /* Title & Tagline */
    .app-title {{
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 0.2rem;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.9);
    }}
    .app-title span {{
        color: #818cf8 !important;
    }}
    .app-tagline {{
        font-size: 1.05rem;
        color: #cbd5e1 !important;
        text-align: center;
        margin-bottom: 2rem;
    }}

    /* 3. SECTION HEADINGS (Bright Gold) */
    h3, .stApp h3 {{
        color: #fbbf24 !important;
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        border-bottom: 2px solid rgba(251, 191, 36, 0.3) !important;
        padding-bottom: 6px !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.9) !important;
    }}

    /* 4. SOLID BLACK TEXT ON ALL FILE UPLOADERS (Fixes "200MB per file" and button) */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] *,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p {{
        color: #0f172a !important;
        text-shadow: none !important;
        font-weight: 700 !important;
    }}
    [data-testid="stFileUploader"] button {{
        background-color: #f1f5f9 !important;
        border: 1px solid #94a3b8 !important;
    }}
    [data-testid="stFileUploader"] button * {{
        color: #0f172a !important;
        font-weight: 800 !important;
    }}

    /* 5. DISABLE FULLSCREEN EXPAND ON IMAGES */
    button[title="View fullscreen"],
    [data-testid="StyledFullScreenButton"],
    [data-testid="stFullScreenFrame"] button[aria-label="Fullscreen"],
    [data-testid="stFullScreenFrame"] [data-testid="stBaseButton-elementToolbar"] {{
        display: none !important;
    }}

    /* 6. SLIDER & CONTROLS */
    div[data-testid="stSlider"] * {{
        color: #ffffff !important;
        font-weight: 700 !important;
    }}

    /* 7. RADIO BUTTONS */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] label p {{
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }}

    /* 8. TABS */
    div[data-testid="stTabs"] button {{
        color: #cbd5e1 !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: #818cf8 !important;
        border-bottom-color: #818cf8 !important;
    }}

    /* 9. PRESET BUTTONS UNDER THUMBNAILS */
    [data-testid="stColumn"] .stButton > button {{
        width: 100% !important;
        background-color: #1e293b !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 8px !important;
        margin-top: -6px !important;
    }}
    [data-testid="stColumn"] .stButton > button:hover {{
        border-color: #818cf8 !important;
        color: #818cf8 !important;
    }}

    /* 10. ACTIVE SELECTION BADGE */
    .selected-badge {{
        display: inline-block;
        background-color: #312e81;
        color: #e0e7ff !important;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 800 !important;
        font-size: 1rem;
        margin-top: 12px;
        border: 1px solid #818cf8;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }}

    /* 11. TOGGLE & SELECTBOXES */
    div[data-testid="stToggle"] * {{
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }}
    div[data-testid="stSelectbox"] * {{
        color: #0f172a !important;
        font-weight: 600 !important;
    }}

    /* FOOTER */
    .app-footer {{
        text-align: center;
        color: #ffffff !important;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 2.5rem;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.9);
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Image Validation Logic
# -----------------------------------------------------------------------------

def validate_uploaded_image(uploaded_file):
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File is too large. Maximum allowed size is 10 MB.")
        return None

    raw_bytes = uploaded_file.getvalue()
    stream = io.BytesIO(raw_bytes)

    try:
        with Image.open(stream) as img:
            if img.format not in ALLOWED_FORMATS:
                st.error(f"Unsupported format '{img.format}'. Only JPEG, PNG, and WEBP are allowed.")
                return None

            width, height = img.size
            if min(width, height) < 64:
                st.error(f"Image dimension too small ({width}x{height}). Minimum dimension must be >= 64px.")
                return None

            if (width * height) > 40_000_000:
                st.error("Image pixel count exceeds limit of 40 million pixels.")
                return None

            img.verify()

    except Exception:
        st.error("Corrupt or unreadable image file.")
        return None

    stream.seek(0)
    return stream

# -----------------------------------------------------------------------------
# Helper: Resolve Stats for a Style Selection
# -----------------------------------------------------------------------------

def resolve_slot_stats(style_choice, custom_stream, scale_size, all_styles, precomputed_dict):
    """Returns (mean, std) for a chosen preset or custom uploaded style."""
    if style_choice == "Upload your own":
        custom_stream.seek(0)
        c_size = min(scale_size, 512)
        return get_style_stats(custom_stream, size=c_size)
    else:
        selected_item = next(
            (s for s in all_styles if f"{s['category'].title()} - {s['name']}" == style_choice),
            None
        )
        if not selected_item:
            return None
        style_id = selected_item["id"]
        cache_key = f"{style_id}|{scale_size}"
        if cache_key in precomputed_dict:
            mean, std = precomputed_dict[cache_key]
            return (mean.to(device), std.to(device))
        else:
            cat = selected_item.get("category", "")
            candidate_path = ROOT_DIR / "styles" / cat / f"{style_id}.jpg"
            return get_style_stats(str(candidate_path), size=scale_size)

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------

def main():
    inject_custom_css()

    # Session State Setup
    if "selected_style_id" not in st.session_state:
        st.session_state["selected_style_id"] = None
    if "selected_style_name" not in st.session_state:
        st.session_state["selected_style_name"] = None
    if "custom_style_stream" not in st.session_state:
        st.session_state["custom_style_stream"] = None
    if "result_image" not in st.session_state:
        st.session_state["result_image"] = None
    if "result_bytes" not in st.session_state:
        st.session_state["result_bytes"] = None
    if "content_pil" not in st.session_state:
        st.session_state["content_pil"] = None

    # Header
    st.markdown('<div class="app-title">Deep<span>Art</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-tagline">Turn your photos into masterwork paintings in seconds with AdaIN.</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 1. CONTENT IMAGE SECTION
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("### 1. Choose Content Photo")

        content_file = st.file_uploader(
            "Upload your content photo (JPEG, PNG, WEBP):",
            type=["jpg", "jpeg", "jfif", "png", "webp"]
        )

        valid_content_stream = None
        if content_file is not None:
            valid_content_stream = validate_uploaded_image(content_file)
            if valid_content_stream:
                content_img = Image.open(valid_content_stream)
                content_img = ImageOps.exif_transpose(content_img).convert("RGB")
                st.session_state["content_pil"] = content_img

                _, center_col, _ = st.columns([1, 2, 1])
                with center_col:
                    st.image(content_img, caption="Content Preview", width=300)

                valid_content_stream.seek(0)

    # -------------------------------------------------------------------------
    # 2. STYLE SELECTION SECTION (Single Mode vs. Blend Mode)
    # -------------------------------------------------------------------------
    all_styles = load_styles_metadata()

    with st.container(border=True):
        st.markdown("### 2. Choose Painting Style")

        mix_styles = st.toggle("Mix two styles", value=False)

        # --- BRANCH A: MIX TWO STYLES ---
        if mix_styles:
            style_options = [f"{s['category'].title()} - {s['name']}" for s in all_styles] + ["Upload your own"]
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### Style A")
                choice_a = st.selectbox("Select Style A", options=style_options, key="select_a")
                custom_stream_a = None

                if choice_a == "Upload your own":
                    file_a = st.file_uploader("Upload painting for Style A", type=["jpg", "jpeg", "jfif", "png", "webp"], key="upload_a")
                    if file_a:
                        custom_stream_a = validate_uploaded_image(file_a)
                        if custom_stream_a:
                            img_a = Image.open(custom_stream_a)
                            st.image(img_a, caption="Style A Preview", width=140)
                            custom_stream_a.seek(0)
                else:
                    item_a = next((s for s in all_styles if f"{s['category'].title()} - {s['name']}" == choice_a), None)
                    if item_a:
                        thumb_path_a = ROOT_DIR / "styles" / "thumbs" / f"{item_a['id']}.jpg"
                        if thumb_path_a.is_file():
                            st.image(str(thumb_path_a), width=140)

                scale_label_a = st.radio("Brushstroke size A", ["Fine", "Medium", "Broad"], index=1, horizontal=True, key="scale_a")
                target_scale_a = SCALE_MAP[scale_label_a]

            with col_b:
                st.markdown("#### Style B")
                default_b_idx = 1 if len(style_options) > 1 else 0
                choice_b = st.selectbox("Select Style B", options=style_options, index=default_b_idx, key="select_b")
                custom_stream_b = None

                if choice_b == "Upload your own":
                    file_b = st.file_uploader("Upload painting for Style B", type=["jpg", "jpeg", "jfif", "png", "webp"], key="upload_b")
                    if file_b:
                        custom_stream_b = validate_uploaded_image(file_b)
                        if custom_stream_b:
                            img_b = Image.open(custom_stream_b)
                            st.image(img_b, caption="Style B Preview", width=140)
                            custom_stream_b.seek(0)
                else:
                    item_b = next((s for s in all_styles if f"{s['category'].title()} - {s['name']}" == choice_b), None)
                    if item_b:
                        thumb_path_b = ROOT_DIR / "styles" / "thumbs" / f"{item_b['id']}.jpg"
                        if thumb_path_b.is_file():
                            st.image(str(thumb_path_b), width=140)

                scale_label_b = st.radio("Brushstroke size B", ["Fine", "Medium", "Broad"], index=1, horizontal=True, key="scale_b")
                target_scale_b = SCALE_MAP[scale_label_b]

            st.markdown("<br>", unsafe_allow_html=True)
            blend_ratio = st.slider("Blend (Style A <-> Style B)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            weight_a = 1.0 - blend_ratio
            weight_b = blend_ratio
            st.caption(f"**Blend balance:** Style A = {int(weight_a * 100)}% | Style B = {int(weight_b * 100)}%")

            if choice_a != "Upload your own" and choice_a == choice_b and target_scale_a == target_scale_b:
                st.warning("Both Style A and Style B are the same painting at the same brushstroke size. Pick different styles to see a blend effect!")

        # --- BRANCH B: SINGLE STYLE ---
        else:
            tab_portrait, tab_landscape, tab_abstract, tab_custom = st.tabs(
                ["Portrait", "Landscape", "Abstract", "Upload your own"]
            )

            def render_preset_tab(category_name):
                styles_subset = [s for s in all_styles if s.get("category") == category_name]
                cols = st.columns(4)
                for idx, s in enumerate(styles_subset):
                    with cols[idx % 4]:
                        thumb_path = ROOT_DIR / "styles" / "thumbs" / f"{s['id']}.jpg"
                        is_selected = (st.session_state["selected_style_id"] == s['id'])

                        if thumb_path.is_file():
                            st.image(str(thumb_path), use_container_width=True)

                        button_label = f"✓ {s['name']}" if is_selected else s['name']
                        if st.button(
                            button_label,
                            key=f"btn_{s['id']}",
                            type="primary" if is_selected else "secondary",
                            use_container_width=True
                        ):
                            st.session_state["selected_style_id"] = s["id"]
                            st.session_state["selected_style_name"] = s["name"]
                            st.session_state["custom_style_stream"] = None
                            st.rerun()

            with tab_portrait:
                render_preset_tab("portrait")
            with tab_landscape:
                render_preset_tab("landscape")
            with tab_abstract:
                render_preset_tab("abstract")
            with tab_custom:
                custom_style_file = st.file_uploader(
                    "Upload custom style painting:",
                    type=["jpg", "jpeg", "jfif", "png", "webp"],
                    key="custom_style_uploader"
                )
                if custom_style_file is not None:
                    valid_custom_stream = validate_uploaded_image(custom_style_file)
                    if valid_custom_stream:
                        st.session_state["custom_style_stream"] = valid_custom_stream
                        st.session_state["selected_style_id"] = None
                        st.session_state["selected_style_name"] = None

                        style_preview_img = Image.open(valid_custom_stream)
                        _, custom_col, _ = st.columns([1, 2, 1])
                        with custom_col:
                            st.image(style_preview_img, caption="Custom Style Preview", width=260)
                        valid_custom_stream.seek(0)

            if st.session_state["selected_style_name"]:
                st.markdown(
                    f'<div class="selected-badge">✓ Selected: {st.session_state["selected_style_name"]}</div>',
                    unsafe_allow_html=True
                )
            elif st.session_state["custom_style_stream"]:
                st.markdown(
                    '<div class="selected-badge">✓ Custom Style Uploaded</div>',
                    unsafe_allow_html=True
                )

    # -------------------------------------------------------------------------
    # 3. CONTROLS SECTION
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("### 3. Adjust Controls")

        alpha_val = st.slider("Style strength", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

        if not mix_styles:
            scale_label = st.radio("Brushstroke size", ["Fine", "Medium", "Broad"], index=1, horizontal=True)
            target_scale_size = SCALE_MAP[scale_label]

        has_content = valid_content_stream is not None

        if mix_styles:
            has_style_a = (choice_a != "Upload your own") or (custom_stream_a is not None)
            has_style_b = (choice_b != "Upload your own") or (custom_stream_b is not None)
            ready_to_stylize = has_content and has_style_a and has_style_b
        else:
            has_single_style = (st.session_state["selected_style_id"] is not None) or (st.session_state["custom_style_stream"] is not None)
            ready_to_stylize = has_content and has_single_style

        stylize_clicked = st.button("Stylize Photo", type="primary", disabled=not ready_to_stylize, use_container_width=True)

    # -------------------------------------------------------------------------
    # 4. INFERENCE EXECUTION WITH ANIMATED STATUS
    # -------------------------------------------------------------------------
    lock = get_stylize_lock()

    if stylize_clicked:
        if not lock.acquire(blocking=False):
            st.error("Server is busy, try again in a few seconds.")
            return

        try:
            status_title = "🎨 Blending two styles onto your photo..." if mix_styles else "🎨 Transforming your photo..."
            with st.status(status_title, expanded=True) as status:
                st.write("1/3 Loading and preprocessing content...")
                precomputed_dict = load_precomputed_stats()

                # --- BRANCH 1: MIX TWO STYLES ---
                if mix_styles:
                    st.write("2/3 Extracting style representations for Style A & Style B...")
                    stats_a = resolve_slot_stats(choice_a, custom_stream_a, target_scale_a, all_styles, precomputed_dict)
                    stats_b = resolve_slot_stats(choice_b, custom_stream_b, target_scale_b, all_styles, precomputed_dict)

                    if not stats_a or not stats_b:
                        st.error("Failed to load statistics for one of the selected styles.")
                        return

                    st.write(f"3/3 Blending features (Style A: {int(weight_a * 100)}% + Style B: {int(weight_b * 100)}%)...")
                    valid_content_stream.seek(0)
                    out_img = stylize_blend(
                        valid_content_stream,
                        [stats_a, stats_b],
                        [weight_a, weight_b],
                        alpha=alpha_val
                    )

                # --- BRANCH 2: SINGLE STYLE ---
                else:
                    st.write("2/3 Extracting neural style representation...")
                    if st.session_state["selected_style_id"]:
                        style_id = st.session_state["selected_style_id"]
                        cache_key = f"{style_id}|{target_scale_size}"
                        if cache_key in precomputed_dict:
                            mean, std = precomputed_dict[cache_key]
                            current_stats = (mean.to(device), std.to(device))
                        else:
                            style_item = next((item for item in all_styles if item["id"] == style_id), None)
                            cat = style_item.get("category", "") if style_item else ""
                            candidate_path = ROOT_DIR / "styles" / cat / f"{style_id}.jpg"
                            current_stats = get_style_stats(str(candidate_path), size=target_scale_size)
                    else:
                        style_stream = st.session_state["custom_style_stream"]
                        style_stream.seek(0)
                        custom_size = min(target_scale_size, 512)
                        current_stats = get_style_stats(style_stream, size=custom_size)

                    st.write("3/3 Applying AdaIN neural brushstrokes...")
                    valid_content_stream.seek(0)
                    out_img = stylize(valid_content_stream, current_stats, alpha=alpha_val)

                # Store output image & bytes in session state
                st.session_state["result_image"] = out_img
                buf = io.BytesIO()
                out_img.save(buf, format="JPEG", quality=90)
                st.session_state["result_bytes"] = buf.getvalue()

                status.update(label="✨ Stylization complete!", state="complete", expanded=False)

        except Exception as err:
            print(f"[ERROR] Inference failed: {err}")
            st.error("Something went wrong while processing your image.")

        finally:
            lock.release()
            gc.collect()

    # -------------------------------------------------------------------------
    # 5. RESULT DISPLAY (Compact & Centered)
    # -------------------------------------------------------------------------
    if st.session_state["result_image"] is not None:
        with st.container(border=True):
            st.markdown("### Result")

            show_original = st.toggle("Show original")

            _, res_col, _ = st.columns([0.5, 3, 0.5])
            with res_col:
                if show_original and st.session_state["content_pil"] is not None:
                    st.image(st.session_state["content_pil"], caption="Original Content Image", width=420)
                else:
                    st.image(st.session_state["result_image"], caption="Stylized Result", width=420)

            st.download_button(
                label="Download Image",
                data=st.session_state["result_bytes"],
                file_name="deepart_stylized.jpg",
                mime="image/jpeg",
                use_container_width=True
            )

    # Footer
    st.markdown('<div class="app-footer">Made with Love by Rajdip ❤️❤️</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
