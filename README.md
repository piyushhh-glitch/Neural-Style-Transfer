# NeuroBrush 🎨
### Neural Style Transfer using Deep Learning

NeuroBrush is a web-based Neural Style Transfer application that combines the content of one image with the artistic style of another using a deep learning model.

It uses **Adaptive Instance Normalization (AdaIN)** with a pretrained VGG encoder and decoder to generate stylized images.

## Features

- Upload a content image.
- Upload a style reference image.
- Generate stylized images using Neural Style Transfer.
- Adjust style intensity using alpha blending.
- View and download the generated output.
- Web-based interface built using Flask.

## Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Model Architecture | VGG Encoder and Decoder |
| Style Transfer Technique | Adaptive Instance Normalization (AdaIN) |
| Backend Framework | Flask |
| Frontend | HTML, CSS |
| Image Processing | Pillow (PIL) |
| Web Server | Gunicorn |
| Reverse Proxy & HTTPS | Caddy |
| Cloud Deployment | AWS EC2 |
| Domain & DNS | DuckDNS |

## How It Works

1. The content image and style image are uploaded.
2. The VGG encoder extracts feature representations from both images.
3. AdaIN aligns the channel-wise mean and variance of content features with style features.
4. The decoder reconstructs the stylized image from the transformed features.
5. Alpha blending controls the intensity of the applied style.

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/piyushhh-glitch/Neural-Style-Transfer.git
cd Neural-Style-Transfer
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the application**

```bash
python app.py
```

Open your browser at:

`http://127.0.0.1:5000`

## Live Demo

**[NeuroBrush – Try it here](https://neurobrush.duckdns.org)**

## Repository

[GitHub – Neural Style Transfer](https://github.com/piyushhh-glitch/Neural-Style-Transfer)

---

**Built with Python, PyTorch and Flask.**