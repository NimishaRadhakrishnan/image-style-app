# Artify v2 — Setup Guide (Mac, No GPU)

## Project Structure
```
artify_v2/
├── app.py
├── requirements.txt
├── static/
│   └── index.html
├── uploads/      ← auto-created
└── outputs/      ← auto-created
```

---

## Step 1 — Python check
```bash
python3 --version   # needs 3.10+
```

## Step 2 — Create virtual environment
```bash
cd artify_v2
python3 -m venv venv
source venv/bin/activate
```

## Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

## Step 4 — Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxx"
```
Or add permanently to ~/.zshrc:
```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-xxxxxxxx"' >> ~/.zshrc && source ~/.zshrc
```
Get a key at: https://console.anthropic.com

## Step 5 — Run
```bash
python app.py
```
Open: http://localhost:5000

---

## What's fixed vs v1

| Issue | v1 | v2 |
|---|---|---|
| Horse/wrong image output | Claude was generating images | PIL transforms the ACTUAL input — no generation |
| Poor style quality | Random parameter tweaks | Dedicated algorithm per style |
| No face preservation | N/A | Styles are CSS filters on pixels — no face replacement |
| Weak analysis | Basic mood | Full subject analysis: gender, age, expression, lighting, quality |
| Wrong image shown | Bug in b64 passthrough | Original always passed directly; result is always derived from input |

## Style algorithms

| Style | Technique |
|---|---|
| Cartoon | Edge detect → binarize → black outline overlay on quantized colour |
| Anime | Gaussian smooth + vivid palette + soft edge darkening |
| Pencil Sketch | Colour dodge (gray ÷ inverted blur) → cream paper composite |
| Digital Painting | Multi-layer blur blend + edge enhance + sharp final pass |
| Watercolour | Soft smoothing + light edges + desaturated palette |
| Oil Painting | Smooth + edge enhance + contrast boost |
| Van Gogh | Edge enhance + swirl colour boost |
| Cyberpunk | RGB channel split (blue+) + high contrast |

All styles: **no generative AI**, **no Stable Diffusion**, **no GPU needed**.
The input image pixels are mathematically transformed — the subject is always preserved.

## VS Code tips
- `Ctrl+`` ` opens integrated terminal
- Install Python extension (ms-python.python)
- Runs on localhost:5000 with auto-reload
