import os
import base64
import io
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

app = Flask(__name__, static_folder="static")
CORS(app)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def np_img(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.float32)

def from_np(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def resize_for_processing(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def enhance_base(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    img = ImageEnhance.Color(img).enhance(1.1)
    return img

def style_cartoon(img: Image.Image) -> Image.Image:
    img = resize_for_processing(img, 1024)
    smooth = img.filter(ImageFilter.GaussianBlur(radius=3))
    smooth = smooth.filter(ImageFilter.SMOOTH_MORE)
    smooth = smooth.filter(ImageFilter.SMOOTH_MORE)
    smooth = ImageEnhance.Color(smooth).enhance(2.2)
    smooth = ImageEnhance.Contrast(smooth).enhance(1.3)
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageEnhance.Contrast(edges).enhance(3.0)
    edges = edges.point(lambda x: 0 if x < 30 else 255)
    edges_inv = edges.point(lambda x: 255 - x)
    edge_rgb = edges_inv.convert("RGB")
    edge_arr = np.array(edge_rgb, dtype=np.float32) / 255.0
    smooth_arr = np.array(smooth, dtype=np.float32)
    result_arr = smooth_arr * edge_arr
    result = from_np(result_arr)
    result = ImageEnhance.Sharpness(result).enhance(1.6)
    return result

def style_anime(img: Image.Image) -> Image.Image:
    img = resize_for_processing(img, 1024)
    smooth = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    smooth = smooth.filter(ImageFilter.SMOOTH_MORE)
    smooth = ImageEnhance.Color(smooth).enhance(1.9)
    smooth = ImageEnhance.Contrast(smooth).enhance(1.25)
    smooth = ImageEnhance.Brightness(smooth).enhance(1.05)
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageEnhance.Contrast(edges).enhance(2.5)
    edges = edges.point(lambda x: 0 if x < 25 else min(x * 2, 255))
    smooth_arr = np.array(smooth, dtype=np.float32)
    edge_arr = np.array(edges.convert("RGB"), dtype=np.float32) / 255.0
    result_arr = smooth_arr * (1.0 - edge_arr * 0.65)
    result = from_np(result_arr)
    result = ImageEnhance.Sharpness(result).enhance(1.5)
    return result

def style_pencil_sketch(img: Image.Image) -> Image.Image:
    img = resize_for_processing(img, 1024)
    gray = img.convert("L")
    inverted = gray.point(lambda x: 255 - x)
    blurred = inverted.filter(ImageFilter.GaussianBlur(radius=12))
    gray_arr = np.array(gray, dtype=np.float32)
    blur_arr = np.array(blurred, dtype=np.float32)
    denom = 255.0 - blur_arr
    denom = np.where(denom < 1, 1, denom)
    sketch_arr = np.clip((gray_arr * 255.0) / denom, 0, 255)
    sketch = from_np(sketch_arr).convert("L")
    sketch = ImageEnhance.Contrast(sketch).enhance(1.6)
    sketch = ImageEnhance.Brightness(sketch).enhance(1.1)
    paper = Image.new("RGB", img.size, (248, 243, 230))
    sketch_rgb = sketch.convert("RGB")
    sketch_arr2 = np.array(sketch_rgb, dtype=np.float32) / 255.0
    paper_arr = np.array(paper, dtype=np.float32)
    result_arr = paper_arr * sketch_arr2
    result = from_np(result_arr)
    return result

def style_digital_painting(img: Image.Image) -> Image.Image:
    img = resize_for_processing(img, 1024)
    layer1 = img.filter(ImageFilter.GaussianBlur(radius=1))
    layer2 = img.filter(ImageFilter.GaussianBlur(radius=3))
    layer3 = img.filter(ImageFilter.GaussianBlur(radius=6))
    arr1 = np.array(layer1, dtype=np.float32)
    arr2 = np.array(layer2, dtype=np.float32)
    arr3 = np.array(layer3, dtype=np.float32)
    blended = arr1 * 0.55 + arr2 * 0.30 + arr3 * 0.15
    painted = from_np(blended)
    painted = ImageEnhance.Color(painted).enhance(1.5)
    painted = ImageEnhance.Contrast(painted).enhance(1.35)
    painted = ImageEnhance.Brightness(painted).enhance(0.95)
    painted = painted.filter(ImageFilter.EDGE_ENHANCE)
    painted = painted.filter(ImageFilter.SMOOTH)
    painted = ImageEnhance.Sharpness(painted).enhance(2.0)
    return painted

def style_watercolour(img: Image.Image) -> Image.Image:
    img = resize_for_processing(img, 1024)
    wet = img.filter(ImageFilter.GaussianBlur(radius=2))
    wet = wet.filter(ImageFilter.SMOOTH_MORE)
    wet = wet.filter(ImageFilter.SMOOTH)
    wet = ImageEnhance.Color(wet).enhance(0.85)
    wet = ImageEnhance.Brightness(wet).enhance(1.15)
    wet = ImageEnhance.Contrast(wet).enhance(0.9)
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = edges.point(lambda x: min(x * 1.5, 80))
    wet_arr = np.array(wet, dtype=np.float32)
    edge_arr = np.array(edges.convert("RGB"), dtype=np.float32) / 255.0
    result_arr = wet_arr * (1.0 - edge_arr * 0.4)
    result = from_np(result_arr)
    return result

def style_oil_painting(img: Image.Image) -> Image.Image:
    img = resize_for_processing(img, 1024)
    oil = img.filter(ImageFilter.SMOOTH_MORE)
    oil = oil.filter(ImageFilter.EDGE_ENHANCE)
    oil = oil.filter(ImageFilter.SMOOTH)
    oil = ImageEnhance.Color(oil).enhance(1.3)
    oil = ImageEnhance.Contrast(oil).enhance(1.4)
    oil = ImageEnhance.Brightness(oil).enhance(0.92)
    oil = ImageEnhance.Sharpness(oil).enhance(1.8)
    return oil

def style_van_gogh(img: Image.Image) -> Image.Image:
    img = resize_for_processing(img, 1024)
    vg = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    vg = vg.filter(ImageFilter.SMOOTH)
    vg = ImageEnhance.Color(vg).enhance(1.8)
    vg = ImageEnhance.Contrast(vg).enhance(1.3)
    vg = vg.filter(ImageFilter.EDGE_ENHANCE)
    vg = ImageEnhance.Sharpness(vg).enhance(1.4)
    return vg

def style_cyberpunk(img: Image.Image) -> Image.Image:
    img = resize_for_processing(img, 1024)
    r, g, b = img.split()
    b_arr = np.array(b, dtype=np.float32) * 1.4
    r_arr = np.array(r, dtype=np.float32) * 0.85
    g_arr = np.array(g, dtype=np.float32) * 1.1
    cy = Image.merge("RGB", (
        from_np(r_arr).convert("L"),
        from_np(g_arr).convert("L"),
        from_np(b_arr).convert("L")
    ))
    cy = ImageEnhance.Contrast(cy).enhance(1.6)
    cy = ImageEnhance.Color(cy).enhance(1.7)
    cy = ImageEnhance.Sharpness(cy).enhance(1.5)
    return cy

STYLE_FUNCTIONS = {
    "cartoon": style_cartoon,
    "anime": style_anime,
    "pencil_sketch": style_pencil_sketch,
    "digital_painting": style_digital_painting,
    "watercolour": style_watercolour,
    "oil_painting": style_oil_painting,
    "van_gogh": style_van_gogh,
    "cyberpunk": style_cyberpunk,
}

def image_to_base64(img: Image.Image) -> str:
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=94)
    return base64.b64encode(buf.getvalue()).decode()

def upscale_if_small(img: Image.Image, min_dim: int = 600) -> Image.Image:
    w, h = img.size
    if min(w, h) < min_dim:
        scale = min_dim / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def analyze_subject_local(img: Image.Image) -> dict:
    inputs = blip_processor(img, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    desc = caption.lower()

    if any(x in desc for x in ["man","woman","person","boy","girl"]):
        subject_type = "human"
    elif any(x in desc for x in ["dog","cat","animal","bird"]):
        subject_type = "animal"
    else:
        subject_type = "object"

    lighting = "Bright" if "bright" in desc or "sunny" in desc else "Normal"
    mood = "Vibrant" if "colorful" in desc else "Calm"

    if subject_type == "human":
        gender = "Female" if "woman" in desc or "girl" in desc else "Male"
        age = "Young Adult (18–25)" if "young" in desc else "Adult"
        expression = "Happy 😊" if "smiling" in desc or "happy" in desc else "Neutral 😐"

        return {
            "subject_type": subject_type,
            "gender": gender,
            "age_category": age,
            "expression": expression,
            "expression_confidence": "Medium",
            "lighting": lighting,
            "scene": caption,
            "image_quality": "Good",
            "recommended_style": "anime",
            "style_tips": "Anime works well for portraits",
            "description": caption
        }

    elif subject_type == "animal":
        return {
            "subject_type": subject_type,
            "species": caption,
            "expression": "Natural",
            "lighting": lighting,
            "scene": caption,
            "image_quality": "Good",
            "recommended_style": "cartoon",
            "style_tips": "Cartoon enhances animals",
            "description": caption
        }

    else:
        return {
            "subject_type": subject_type,
            "category": "Scene",
            "mood": mood,
            "dominant_colors": ["auto"],
            "lighting": lighting,
            "scene": caption,
            "image_quality": "Good",
            "recommended_style": "digital_painting",
            "style_tips": "Painting suits landscapes",
            "description": caption
        }

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    img = upscale_if_small(img)
    img = enhance_base(img)
    img_b64 = image_to_base64(img)
    try:
        analysis = analyze_subject_local(img)
        analysis["preview_b64"] = img_b64
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stylize", methods=["POST"])
def stylize():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    style = request.form.get("style", "cartoon")
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    img = upscale_if_small(img)
    img = enhance_base(img)
    fn = STYLE_FUNCTIONS.get(style)
    if not fn:
        return jsonify({"error": f"Unknown style: {style}"}), 400
    try:
        result = fn(img)
        result = ImageEnhance.Sharpness(result).enhance(1.2)
        result_b64 = image_to_base64(result)
        orig_b64 = image_to_base64(img)
        return jsonify({
            "result_b64": result_b64,
            "original_b64": orig_b64,
            "style": style
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)