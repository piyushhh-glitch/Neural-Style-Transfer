---
title: NeuroBrush
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# 🎨 NeuroBrush — Neural Style Transfer

**NeuroBrush** is a deep learning web application that transforms images by combining the content of one image with the artistic style of another.

Built using Python, Flask, and PyTorch, NeuroBrush provides a web interface for experimenting with Neural Style Transfer.

## ✨ Features

- **Content Image:** Upload an image whose structure you want to preserve.
- **Style Image:** Upload an image whose artistic style you want to apply.
- **Style Control:** Adjust the available style-strength parameter.
- **Deep Learning:** Uses pretrained neural network weights for image stylization.
- **Web Interface:** Interact with the application through a Flask-powered interface.

## 🛠️ Tech Stack

- Python
- Flask
- PyTorch
- Torchvision
- NumPy
- Pillow
- Flask-WTF
- Gunicorn
- Docker

## 🧠 How It Works

Neural Style Transfer combines two inputs:

1. **Content Image** — provides the objects, shapes, and overall structure.
2. **Style Image** — provides the artistic patterns, textures, and visual appearance.

A neural network extracts content and style features to generate a stylized image that combines both.

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://huggingface.co/spaces/piyushh-glitch/NeuroBrush
cd NeuroBrush
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
python app.py
```

Open the local URL displayed in your terminal.

## ☁️ Deployment

NeuroBrush is configured for deployment on Hugging Face Spaces using Docker.

The application runs on port `7860`, with Gunicorn serving the Flask application.

The deployment requires the application source code, templates, static assets, and pretrained model weights.

## 📁 Main Project Components

- `app.py` — Flask application.
- `templates/` — HTML templates.
- `static/` — Static assets and frontend resources.
- `utils/` — Model and image-processing utilities.
- `vgg_normalised.pth` — VGG model weights.
- `decoder_final.pth` — Decoder model weights.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — Container configuration.

## ⚙️ Notes

- Image processing speed depends on the available CPU and memory.
- Pretrained model weights must be present at the paths expected by the application.
- Large model files may require Git LFS.
- The application should be tested with the target hosting environment's resource limits.

## 📄 License

No license has been specified yet.

---

**NeuroBrush — Where deep learning meets digital art.**