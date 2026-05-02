# 🎨 Artify v2 — AI-Powered Image Stylization & Analysis Engine

<div align="center">

*Transform any photo into a work of art using real computer vision — no generative AI, no GPU required.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red?logo=pytorch&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-10.x-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Version](https://img.shields.io/badge/Version-0.0.17-informational)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Real-World Problem](#-real-world-problem)
- [Project Objectives](#-project-objectives)
- [How It Works — System Architecture](#-how-it-works--system-architecture)
- [ML Models Used](#-ml-models-used)
- [Style Algorithms](#-style-algorithms-8-styles)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Installation & Setup](#-installation--setup)
- [Running the App](#-running-the-app)
- [Results & Key Insights](#-results--key-insights)
- [Visualizations](#-visualizations)
- [Challenges & Learnings](#-challenges--learnings)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## 🧠 Overview

**Artify v2** is a full-stack, ML-powered web application that does two things:

1. **Analyzes** an uploaded image using real transformer-based vision models (BLIP for captioning, CLIP for understanding) to extract subject type, expression, lighting, mood, and a recommended art style.
2. **Stylizes** that image into one of 8 distinct artistic styles — cartoon, anime, pencil sketch, digital painting, watercolour, oil painting, Van Gogh, or cyberpunk — using deterministic pixel-level image processing algorithms.

The core philosophy: **no generative AI hallucinations, no GPU, no cloud dependency.** Every stylized output is a mathematical transformation of the *actual* input image. The subject, face, and composition are always preserved.

This project demonstrates the full ML engineering stack: model integration, REST API design, image processing pipelines, graceful fallback engineering, and a browser-based frontend — all running locally on CPU.

---

## 🚩 Real-World Problem

The creative tools market (Prisma, Lensa, Adobe Firefly) is dominated by:

- **Generative AI models** that replace your subject rather than stylize it (a person's face gets "reimagined" rather than transformed)
- **Cloud-dependent APIs** with rate limits, privacy risks, and costs
- **GPU requirements** that exclude most consumer hardware

This creates a gap: users who want fast, private, CPU-friendly, subject-preserving artistic stylization with intelligent style recommendations.

**Artify v2** addresses this by combining pre-trained vision transformers (for understanding) with hand-crafted algorithmic pipelines (for transformation) — giving users intelligent guidance and deterministic, reproducible results.

---

## 🎯 Project Objectives

| Objective | Approach |
|---|---|
| Analyze image content intelligently | BLIP image captioning + subject classification logic |
| Recommend the best art style for the content | Rule-based inference from BLIP caption keywords |
| Apply 8 distinct, high-quality artistic styles | Dedicated PIL/NumPy algorithm per style |
| Preserve the original subject in every output | Pixel-level transforms only — no generation |
| Run fully offline on CPU | CPU-mode PyTorch + PIL (no CUDA, no API calls) |
| Expose a clean REST API | Flask with `/analyze` and `/stylize` endpoints |
| Work without a framework-heavy frontend | Vanilla HTML/JS served as a static file |

---

## 🏗️ How It Works — System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Browser (index.html)                  │
│           Upload image → Call /analyze or /stylize       │
└────────────────────┬─────────────────────────────────────┘
                     │ multipart/form-data POST
┌────────────────────▼─────────────────────────────────────┐
│                  Flask REST API (app.py)                  │
│                                                          │
│  ┌──────────────────┐      ┌────────────────────────┐    │
│  │  /analyze        │      │  /stylize              │    │
│  │                  │      │                        │    │
│  │  upscale_if_small│      │  upscale_if_small      │    │
│  │  enhance_base    │      │  enhance_base          │    │
│  │  BLIP caption →  │      │  STYLE_FUNCTIONS[name] │    │
│  │  subject classify│      │  → PIL + NumPy ops     │    │
│  │  → JSON response │      │  → base64 JPEG         │    │
│  └──────────────────┘      └────────────────────────┘    │
│                                                          │
│  Shared preprocessing: upscale → enhance → transform     │
└──────────────────────────────────────────────────────────┘
                     │ loads once at startup
┌────────────────────▼─────────────────────────────────────┐
│              HuggingFace Models (CPU)                    │
│   CLIP  — openai/clip-vit-base-patch32                   │
│   BLIP  — Salesforce/blip-image-captioning-large         │
└──────────────────────────────────────────────────────────┘
```

### Request Lifecycle

```
User uploads image
       ↓
Upscale if smallest dimension < 600px (LANCZOS)
       ↓
Base enhancement: Sharpness ×1.4, Color ×1.1
       ↓
    ┌──────┐       ┌──────────┐
    │Analyze│       │ Stylize  │
    └───┬──┘       └────┬─────┘
        │               │
   BLIP caption    Look up style fn
        │               │
  Classify subject  Apply PIL/NumPy
  (human/animal     transform pipeline
   /object)              │
        │          Final sharpen ×1.2
  Build JSON             │
  + recommend       Encode to base64
  style                  │
        │          Return orig + result
   Return JSON      as JSON
```

---

## 🤖 ML Models Used

### 1. BLIP — Image Captioning (`Salesforce/blip-image-captioning-large`)

**Purpose:** Generate a natural-language description of the uploaded image.

BLIP (Bootstrapped Language-Image Pretraining) is a vision-language model that produces free-text captions for images. In Artify, the caption drives all downstream analysis:

```python
inputs = blip_processor(img, return_tensors="pt")
out = blip_model.generate(**inputs)
caption = blip_processor.decode(out[0], skip_special_tokens=True)
# e.g. "a young woman smiling outdoors in bright sunlight"
```

The caption text is then parsed with keyword matching to extract:
- **Subject type:** human / animal / object-scene
- **Gender & age:** from "woman", "man", "young", "girl", "boy"
- **Expression:** from "smiling", "happy"
- **Lighting:** from "bright", "sunny"
- **Mood:** from "colorful"
- **Recommended style:** mapped from subject type

### 2. CLIP — Visual Embeddings (`openai/clip-vit-base-patch32`)

**Purpose:** Loaded at startup as part of the vision model suite (architecture ready for zero-shot classification and text-image similarity tasks).

CLIP (Contrastive Language-Image Pretraining) learns joint embeddings of images and text. It is initialized alongside BLIP, positioning Artify for future zero-shot style scoring — e.g. asking "does this image look more like a portrait or a landscape?" without any additional training.

### Why These Models?

| Criterion | Choice |
|---|---|
| No fine-tuning required | Both are pre-trained and inference-ready |
| CPU compatibility | PyTorch CPU mode; no CUDA dependency |
| Open weights | Downloaded once from HuggingFace Hub; no API key |
| State-of-the-art captions | BLIP-large outperforms smaller captioning models on natural scenes |

---

## 🎨 Style Algorithms (8 Styles)

Each style is a **deterministic, pixel-level pipeline** — not a neural style transfer. This means outputs are fast, reproducible, and always derived from the actual input.

| Style | Core Technique | Key PIL / NumPy Operations |
|---|---|---|
| **Cartoon** | Quantized colour + black outline overlay | GaussianBlur → Color ×2.2 → edge binarize → multiply overlay |
| **Anime** | Soft smooth + vivid palette + soft edge darkening | Blur ×1.5 → Color ×1.9 → edge darken at 65% opacity |
| **Pencil Sketch** | Colour dodge formula on greyscale + cream paper | `gray × 255 / (255 − inverted_blur)` → paper composite |
| **Digital Painting** | Multi-layer blur blend + edge enhance | 3-layer weighted blend (55/30/15%) → EDGE_ENHANCE → Sharpness ×2.0 |
| **Watercolour** | Wet-look softening + light edge bleeding | SMOOTH ×2 → Color ×0.85 → Brightness ×1.15 → edge ×0.4 overlay |
| **Oil Painting** | Smooth + edge enhance + rich contrast | SMOOTH_MORE → EDGE_ENHANCE → Color ×1.3 → Contrast ×1.4 |
| **Van Gogh** | Dramatic edge enhancement + swirling colour | EDGE_ENHANCE_MORE → Color ×1.8 → Contrast ×1.3 |
| **Cyberpunk** | RGB channel manipulation (blue boost) + high contrast | Split channels → B ×1.4, R ×0.85 → Contrast ×1.6 → Color ×1.7 |

### Pencil Sketch — The Colour Dodge Formula

The most mathematically interesting style uses the classic colour-dodge technique from Photoshop's blend modes:

```
sketch = clip( gray × 255 / (255 − gaussian_blur(inverted_gray)) , 0, 255 )
```

This creates the illusion of pencil strokes by amplifying areas where the original grey values exceed the blurred-inverted version — exactly where pencil marks would lie heaviest. The result is composited onto a warm cream (`RGB 248, 243, 230`) paper background.

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Web Framework** | Flask 3.x | REST API (`/analyze`, `/stylize`), static file serving |
| **CORS** | flask-cors | Enables cross-origin requests from the browser frontend |
| **ML — Captioning** | BLIP (HuggingFace Transformers) | Image-to-text captioning for scene understanding |
| **ML — Embeddings** | CLIP (HuggingFace Transformers) | Visual-language joint embedding model |
| **Deep Learning Runtime** | PyTorch (CPU) | Inference backend for both transformer models |
| **Image Processing** | Pillow (PIL) | All artistic style transformations and enhancements |
| **Numerical Computing** | NumPy | Pixel-level array operations (blend, divide, clip, channel split) |
| **Frontend** | HTML / CSS / JavaScript (Vanilla) | Single-page upload UI served as a static file |
| **Python Version** | 3.10.13 (via pyenv) | Language runtime |

---

## 📁 Project Structure

```
artify_v2/
│
├── app.py                  # Main Flask application — all routes, ML models, style functions
│
├── static/
│   └── index.html          # Frontend UI — upload form, style selector, preview display
│
├── requirements.txt        # Python dependencies
├── SETUP.md                # Developer setup guide (Mac, CPU, no GPU)
├── package_info.json       # Version metadata (v0.0.17)
├── pyvenv.cfg              # Virtual environment config (Python 3.10.13)
│
├── uploads/                # Temp storage for incoming images (auto-created)
└── outputs/                # Temp storage for stylized outputs (auto-created)
```

### Key File Roles

**`app.py`** is the entire backend in one file. It has four logical sections:

1. **Model loading** (at startup, once) — CLIP and BLIP loaded into memory
2. **Preprocessing utilities** — `resize_for_processing`, `enhance_base`, `upscale_if_small`, `np_img`, `from_np`
3. **Style function library** — 8 named functions registered in `STYLE_FUNCTIONS` dict
4. **Flask routes** — `/` (serve UI), `/analyze` (POST), `/stylize` (POST)

**`static/index.html`** is a self-contained single-page frontend. No build step, no npm, no React — just HTML served directly by Flask via `send_from_directory`.

**`SETUP.md`** documents the v1 → v2 improvements (the bug where wrong images were being shown, weak analysis, poor style quality) and the algorithm breakdown per style — valuable context for code reviewers.

---

## 🌐 API Reference

### `POST /analyze`

Accepts a multipart image upload. Returns subject analysis and a base64 preview.

**Request:**
```
Content-Type: multipart/form-data
Body: image=<file>
```

**Response (human subject):**
```json
{
  "subject_type": "human",
  "gender": "Female",
  "age_category": "Young Adult (18–25)",
  "expression": "Happy 😊",
  "expression_confidence": "Medium",
  "lighting": "Bright",
  "scene": "a young woman smiling outdoors in bright sunlight",
  "image_quality": "Good",
  "recommended_style": "anime",
  "style_tips": "Anime works well for portraits",
  "description": "a young woman smiling outdoors in bright sunlight",
  "preview_b64": "<base64 JPEG string>"
}
```

**Response (animal subject):**
```json
{
  "subject_type": "animal",
  "species": "a golden retriever sitting on grass",
  "expression": "Natural",
  "lighting": "Normal",
  "scene": "a golden retriever sitting on grass",
  "image_quality": "Good",
  "recommended_style": "cartoon",
  "style_tips": "Cartoon enhances animals",
  "description": "a golden retriever sitting on grass"
}
```

### `POST /stylize`

Accepts a multipart image and a style name. Returns original and stylized images as base64.

**Request:**
```
Content-Type: multipart/form-data
Body:
  image=<file>
  style=<one of: cartoon | anime | pencil_sketch | digital_painting |
                 watercolour | oil_painting | van_gogh | cyberpunk>
```

**Response:**
```json
{
  "result_b64": "<base64 JPEG — stylized image>",
  "original_b64": "<base64 JPEG — preprocessed original>",
  "style": "cartoon"
}
```

**Error response:**
```json
{ "error": "Unknown style: impressionist" }
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or higher (`python3 --version`)
- `pip` (bundled with Python 3.10+)
- ~4 GB disk space (for BLIP-large and CLIP weights, downloaded once on first run)
- No GPU required — runs entirely on CPU

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/artify-v2.git
cd artify-v2
```

### Step 2 — Create a Virtual Environment

```bash
python3 -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Full dependency set:

```
flask>=3.0.0
flask-cors>=4.0.0
Pillow>=10.0.0
numpy>=1.24.0
torch>=2.0.0
transformers>=4.35.0
```

> **Model weights:** On first run, HuggingFace will automatically download `openai/clip-vit-base-patch32` (~600 MB) and `Salesforce/blip-image-captioning-large` (~1.9 GB) to `~/.cache/huggingface/`. Subsequent runs load from cache.

> **Apple Silicon (M1/M2/M3):** PyTorch CPU runs natively. MPS acceleration is supported with `torch>=2.0.0` but not required.

---

## ▶️ Running the App

```bash
# Activate virtual environment first
source venv/bin/activate

python app.py
```

Expected output:

```
 * Running on http://127.0.0.1:5001
 * Debug mode: on
```

Open your browser at **[http://localhost:5001](http://localhost:5001)**

> **First run:** Allow 30–90 seconds for model weights to load into memory. All subsequent requests are fast.

### Quick API test with curl

```bash
# Analyze
curl -X POST http://localhost:5001/analyze \
  -F "image=@photo.jpg" | python3 -m json.tool

# Stylize
curl -X POST http://localhost:5001/stylize \
  -F "image=@photo.jpg" \
  -F "style=pencil_sketch" | python3 -m json.tool
```

---

## 📊 Results & Key Insights

### v1 → v2 Improvements

| Issue in v1 | Root Cause | v2 Fix |
|---|---|---|
| Wrong image in output | Base64 passthrough bug | Original always derived directly from uploaded bytes |
| Subject face replaced | Attempted generative approach | Pure pixel transforms — no generation at any stage |
| Weak analysis output | No vision model | BLIP captioning provides grounded scene description |
| Poor style differentiation | Generic parameter tweaks | Dedicated algorithm per style with tuned constants |

### Processing Performance (CPU, M1 MacBook, 1024px image)

| Stage | Approximate Time |
|---|---|
| Model load (first run only) | 30–90 seconds |
| Preprocessing (`upscale` + `enhance`) | < 0.1 seconds |
| BLIP caption generation | 3–8 seconds |
| Any style transform | 0.5–2 seconds |
| Base64 encode + JSON response | < 0.1 seconds |

### Subject Classification Reliability

BLIP-large produces descriptive, grounded captions for typical user photos. Keyword parsing achieves reliable human / animal / scene classification for the vast majority of portrait, pet, and landscape photos. Known edge cases: group photos (classified as human if any person keyword appears), abstract or macro images (default to object/scene).

---

## 🖼️ Visualizations

Artify is an image-in / image-out application; the primary visualizations are the before/after style previews in the browser UI.

**Cartoon** — Strong black outlines over flat, high-saturation colour regions. Works best on faces and animals where sharp natural edges already exist.

**Anime** — Softer than cartoon; vivid but not flat. Skin tones and hair take on characteristic anime look via soft edge darkening rather than hard outlines.

**Pencil Sketch** — High-contrast greyscale strokes on warm cream paper. Fine detail areas (hair, fabric texture) produce the densest stroke patterns; the colour-dodge formula naturally creates lighter strokes in bright areas.

**Digital Painting** — The 3-layer Gaussian blend (55% fine / 30% medium / 15% coarse) creates depth-of-field-like softening; a final `EDGE_ENHANCE` + `Sharpness ×2.0` pass restores structure, mimicking brush-on-canvas texture.

**Watercolour** — The most subtle style. Colours wash slightly (Color ×0.85, Brightness ×1.15) with soft edge bleeding that creates the wet-paper diffusion of real watercolour.

**Cyberpunk** — Immediately recognizable: blues ×1.4, reds ×0.85, extreme contrast (×1.6) and saturation (×1.7). Night scenes and urban photos respond most dramatically.

---

## 🧩 Challenges & Learnings

### 1. Generative vs. Transformative — A Design Philosophy Decision
**Challenge:** v1 attempted a generative approach that "re-drew" images in a given style. This caused the model to hallucinate new subjects, especially replacing faces.

**Learning:** For subject-preserving stylization, deterministic pixel transforms are the correct tool. Generative AI is for *creation*; it is the wrong tool for *transformation with preservation*. This distinction is central to responsible ML product design and came up repeatedly during the v1 → v2 rethink.

### 2. Model Weight Size vs. Per-Request Latency
**Challenge:** BLIP-large (~1.9 GB) has significant load time. Loading it inside the request handler would make every request take 30–90 seconds.

**Solution:** Both models are loaded **once at module level** before Flask starts serving. This is the standard production pattern — initialize expensive resources at startup, not per-request. The tradeoff is higher idle RAM (~3 GB) for near-instant inference latency.

### 3. Image Size Normalization
**Challenge:** User uploads range from 200×200 thumbnails to 4000×3000 camera shots. Too small and style effects have no texture to work with; too large and transforms become slow.

**Solution:** A two-stage normalization: `upscale_if_small` (min dimension → 600px, LANCZOS) before processing; `resize_for_processing` (max dimension → 1024px) inside each style function. This ensures consistent quality without memory pressure.

### 4. NumPy Float32 Precision in Blend Operations
**Challenge:** Pixel blending with integer arithmetic causes clipping artifacts — values going negative or above 255 wrap silently, creating strange bright/dark patches in early cartoon and sketch outputs.

**Solution:** All intermediate arrays use `dtype=np.float32`. Clipping to `[0, 255]` only happens at the final `from_np()` step, preserving the full dynamic range through multi-step operations.

### 5. JPEG vs. PNG for Base64 API Transfer
**Challenge:** Full-resolution PNG images as base64 JSON created 5–15 MB payloads, making the API noticeably slow even over localhost.

**Solution:** `image_to_base64()` always encodes as JPEG at `quality=94` — visually indistinguishable from PNG for photographic content but 60–80% smaller. RGBA images are first converted to RGB (JPEG does not support transparency).

---

## 🚀 Future Improvements

- [ ] **Zero-shot style scoring with CLIP** — Use CLIP embeddings to score style-content fit (e.g. "does this landscape suit watercolour more than cyberpunk?") and re-rank recommendations dynamically
- [ ] **Style intensity slider** — Expose key constants (Color multiplier, edge opacity, blur radius) as UI controls so users can tune the strength of each effect
- [ ] **Batch processing** — Accept a zip of images and return stylized outputs as a downloadable zip
- [ ] **Neural style transfer mode** — An optional slow/high-quality path using VGG-based neural style transfer for painterly texture synthesis
- [ ] **Face landmark preservation** — Integrate MediaPipe face detection to apply style transforms with face-region masking, preventing facial distortion
- [ ] **Docker container** — Package the app, models, and dependencies in a single `Dockerfile` for one-command deployment
- [ ] **Streamlit UI alternative** — Interactive sliders and live side-by-side previews without JavaScript
- [ ] **Async processing** — Offload style transforms to a Celery task queue for large images, preventing HTTP timeout
- [ ] **Persistent gallery** — Store processed images with metadata (style, subject type, timestamp) in SQLite for a user-facing gallery view

---

## 👤 Author

**[Nimisha]**
*Aspiring ML Engineer & Data Scientist*

Artify v2 was built to go beyond the Jupyter notebook and demonstrate applied machine learning in a production-style context — integrating pre-trained transformer models (BLIP, CLIP) into a REST API, designing deterministic computer vision pipelines, and shipping a usable end-to-end application. It reflects my interest in the intersection of deep learning, image processing, and software engineering.

- 🔗 [LinkedIn](https://www.linkedin.com/in/nimisha-r-23a888292/)
- 💻 [GitHub](https://www.linkedin.com/in/nimisha-r-23a888292/)
- 📧 [your.email@example.com](nimisharadhakrishnan9487@gmail.com)
- 🌐 [Portfolio](https://your-portfolio.com)

---

## 📄 License

This project is licensed under the **MIT License**.

Pre-trained model weights are subject to their respective licenses:
- **CLIP** — [OpenAI CLIP License](https://github.com/openai/CLIP/blob/main/LICENSE)
- **BLIP** — [Salesforce BSD-3 License](https://github.com/salesforce/BLIP/blob/main/LICENSE.txt)

---

<div align="center">

Built with 🧠 HuggingFace Transformers · 🎨 PIL · 🔢 NumPy · 🐍 Flask

*CPU-only · Offline-first · No generative AI · Subject always preserved*

**[Report a Bug](https://github.com/your-username/artify-v2/issues) · [Request a Feature](https://github.com/your-username/artify-v2/issues) · [View Portfolio](https://your-portfolio.com)**

</div>
