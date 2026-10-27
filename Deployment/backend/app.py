import io
import json
import gc
import threading
from pathlib import Path
from PIL import Image
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory

# Import inference functions and device from inference.py (do not modify inference.py)
from inference import device, get_style_stats, stylize

# Directory configuration
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STYLES_DIR = ROOT_DIR / "styles"
STYLES_JSON_PATH = STYLES_DIR / "styles.json"
STYLE_STATS_PT_PATH = STYLES_DIR / "style_stats.pt"

# Scale name mapping to pixel resolution
SCALE_MAP = {
    "fine": 128,
    "medium": 512,
    "broad": 1280
}

# Supported image formats (identified by Pillow format descriptor)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# Initialize Flask application
app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR / "static"),
    static_url_path="/static"
)

# 10 MB max request payload size
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Lock to ensure only one stylization job runs at a time
stylize_lock = threading.Lock()

# Cache for style statistics: keyed by (style_id, scale_size)
style_stats_cache = {}


def load_precomputed_stats():
    """
    Loads precomputed style stats from styles/style_stats.pt into style_stats_cache.
    Converts '<id>|<size>' keys into (id, int(size)) tuples on current device.
    """
    if not STYLE_STATS_PT_PATH.is_file():
        print("[INFO] styles/style_stats.pt not found. Default styles will compute lazily.")
        return

    try:
        saved_stats = torch.load(STYLE_STATS_PT_PATH, map_location=device)
        for key_str, (mean, std) in saved_stats.items():
            style_id, size_str = key_str.split("|")
            style_stats_cache[(style_id, int(size_str))] = (mean.to(device), std.to(device))
        print(f"[INFO] Loaded {len(style_stats_cache)} precomputed style stats into memory.")
    except Exception as err:
        print(f"[WARNING] Failed to load precomputed style stats: {err}. Falling back to lazy compute.")


# Load precomputed style stats at startup
load_precomputed_stats()


def read_styles_metadata():
    """Reads styles metadata from styles/styles.json. Returns [] if not found."""
    if not STYLES_JSON_PATH.is_file():
        return []
    try:
        with open(STYLES_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def resolve_style_image_path(style_id):
    """
    Finds the image file path on disk for a given preset style_id.
    Looks in styles/<category>/<style_id>.<ext>.
    """
    styles = read_styles_metadata()
    style_info = next((item for item in styles if item.get("id") == style_id), None)
    if not style_info:
        return None

    category = style_info.get("category", "")
    target_dir = STYLES_DIR / category

    if not target_dir.is_dir():
        return None

    # Search common image extensions matching the style ID stem
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".jfif"):
        candidate = target_dir / f"{style_id}{ext}"
        if candidate.is_file():
            return candidate

    return None


def validate_image_stream(file_storage):
    """
    Reads an uploaded file from memory, validates format, integrity, and dimensions.
    Returns a seeked BytesIO stream if valid, or raises ValueError with a descriptive message.
    """
    if not file_storage or not file_storage.filename:
        raise ValueError("Missing file upload.")

    file_bytes = file_storage.read()
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    image_stream = io.BytesIO(file_bytes)

    try:
        with Image.open(image_stream) as img:
            if img.format not in ALLOWED_FORMATS:
                raise ValueError(
                    f"Unsupported format '{img.format}'. Only JPEG, PNG, and WEBP are allowed."
                )

            width, height = img.size
            if min(width, height) < 64:
                raise ValueError(
                    f"Image dimension too small ({width}x{height}). Both width and height must be >= 64px."
                )
            if (width * height) > 40_000_000:
                raise ValueError(
                    f"Image pixel count exceeds limit ({width * height} > 40,000,000 pixels)."
                )

            img.verify()

    except (ValueError, Exception) as err:
        if isinstance(err, ValueError):
            raise err
        raise ValueError(f"Corrupt or unreadable image file.")

    image_stream.seek(0)
    return image_stream


# -----------------------------------------------------------------------------
# ERROR HANDLERS
# -----------------------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle files exceeding 10MB."""
    return jsonify({"error": "Request entity too large. Maximum allowed size is 10 MB."}), 413


# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Serve the single-page frontend application."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/styles/<path:filename>", methods=["GET"])
def serve_styles_assets(filename):
    """Serve thumbs and assets from styles/."""
    return send_from_directory(STYLES_DIR, filename)


@app.route("/api/styles", methods=["GET"])
def get_styles():
    """Returns the list of presets from styles/styles.json, or [] if missing."""
    styles = read_styles_metadata()
    return jsonify(styles), 200


@app.route("/api/stylize", methods=["POST"])
def run_stylize():
    """
    Executes style transfer:
    - content: required image file
    - style_id OR style: mutually exclusive, at least one required
    - alpha: float [0.0, 1.0], default 0.8
    - scale: 'fine'|'medium'|'broad', default 'medium'
    """
    # 1. Non-blocking lock check: return 503 if currently running inference
    if not stylize_lock.acquire(blocking=False):
        return jsonify({"error": "Server is busy processing another request. Please try again."}), 503

    try:
        # 2. Validate Alpha parameter
        raw_alpha = request.form.get("alpha", "0.8")
        try:
            alpha = float(raw_alpha)
            if not (0.0 <= alpha <= 1.0):
                return jsonify({"error": "Parameter 'alpha' must be a float between 0.0 and 1.0."}), 400
        except ValueError:
            return jsonify({"error": "Invalid 'alpha' parameter. Must be a numeric value."}), 400

        # 3. Validate Scale parameter
        scale_name = request.form.get("scale", "medium").lower()
        if scale_name not in SCALE_MAP:
            return jsonify({
                "error": f"Invalid 'scale' parameter '{scale_name}'. Allowed: 'fine', 'medium', 'broad'."
            }), 400
        style_size = SCALE_MAP[scale_name]

        # 4. Validate Content image upload
        if "content" not in request.files:
            return jsonify({"error": "Field 'content' is required as a file upload."}), 400

        try:
            content_stream = validate_image_stream(request.files["content"])
        except ValueError as err:
            return jsonify({"error": f"Content image error: {str(err)}"}), 400

        # 5. Validate Style input (style_id OR style file, never both, at least one)
        has_style_id = bool(request.form.get("style_id"))
        has_style_file = "style" in request.files and request.files["style"].filename != ""

        if has_style_id and has_style_file:
            return jsonify({"error": "Provide either 'style_id' or 'style' file, not both."}), 400

        if not has_style_id and not has_style_file:
            return jsonify({"error": "Either 'style_id' or 'style' file must be provided."}), 400

        # 6. Extract / compute style statistics
        if has_style_id:
            style_id = request.form.get("style_id").strip()
            cache_key = (style_id, style_size)

            # Check cache (populated at startup from style_stats.pt)
            if cache_key in style_stats_cache:
                current_style_stats = style_stats_cache[cache_key]
            else:
                # Fallback: compute lazily if key was not precomputed
                style_file_path = resolve_style_image_path(style_id)
                if not style_file_path:
                    return jsonify({"error": f"Style ID '{style_id}' not found in styles directory."}), 400

                current_style_stats = get_style_stats(style_file_path, size=style_size)
                style_stats_cache[cache_key] = current_style_stats

        else:
            # Custom style upload: capped at 512px to reduce memory use
            try:
                style_stream = validate_image_stream(request.files["style"])
            except ValueError as err:
                return jsonify({"error": f"Style image error: {str(err)}"}), 400

            custom_style_size = min(style_size, 512)
            current_style_stats = get_style_stats(style_stream, size=custom_style_size)

        # 7. Perform style transfer
        output_pil_img = stylize(content_stream, current_style_stats, alpha=alpha)

        # 8. Encode resulting PIL Image into JPEG bytes
        output_buffer = io.BytesIO()
        output_pil_img.save(output_buffer, format="JPEG", quality=90)
        output_buffer.seek(0)

        return send_file(
            output_buffer,
            mimetype="image/jpeg",
            as_attachment=False
        )

    except Exception as err:
        print("Inference error:", err)
        return jsonify({"error": "Something went wrong while processing your image."})

    finally:
        stylize_lock.release()
        gc.collect()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)