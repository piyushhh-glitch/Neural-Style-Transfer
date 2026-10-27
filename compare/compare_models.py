
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Project root ko Python import path mein add karo
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.model import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256

# Model weights
VGG_PATH = PROJECT_ROOT / "vgg_normalised.pth"
GITHUB_WEIGHTS = PROJECT_ROOT / "decoder_final.pth"
TRAINED_WEIGHTS = (
    PROJECT_ROOT / "experiments" / "adain_2000" / "decoder_10.pth"
)

# Input images
CONTENT_IMAGE = (
    PROJECT_ROOT / "cleaned_data" / "content" / "content_00005.jpg"
)
STYLE_IMAGE = (
    PROJECT_ROOT / "cleaned_data" / "style" / "style_02003.jpg"
)

# Output compare folder mein save hoga
OUTPUT_PATH = Path(__file__).resolve().parent / "comparison_4.png"


def load_image(path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    with Image.open(path) as image:
        image = image.convert("RGB")
        return transform(image).unsqueeze(0).to(DEVICE)


def load_decoder(path, rename_github=False):
    model = Decoder().to(DEVICE)

    weights = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    # GitHub weights mein net.* ko decoder.* mein rename karo
    if rename_github:
        weights = {
            key.replace("net.", "decoder.", 1)
            if key.startswith("net.") else key: value
            for key, value in weights.items()
        }

    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


def main():
    required_files = [
        VGG_PATH,
        GITHUB_WEIGHTS,
        CONTENT_IMAGE,
        STYLE_IMAGE,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    print(f"Device: {DEVICE}")
    print("Loading VGG encoder...")
    encoder = VGGEncoder(str(VGG_PATH)).to(DEVICE).eval()

    print("Loading GitHub pretrained decoder...")
    github_decoder = load_decoder(
        GITHUB_WEIGHTS,
        rename_github=True
    )

    # Trained checkpoint optional hai
    trained_decoder = None

    if TRAINED_WEIGHTS.is_file():
        print("Loading your trained decoder...")
        trained_decoder = load_decoder(TRAINED_WEIGHTS)
    else:
        print("Trained checkpoint not found.")
        print("Generating pretrained output only.")

    content = load_image(CONTENT_IMAGE)
    style = load_image(STYLE_IMAGE)

    with torch.no_grad():
        content_features = encoder(content)[-1]
        style_features = encoder(style)[-1]

        target = adaptive_instance_normalization(
            content_features,
            style_features
        )

        github_output = github_decoder(target).clamp(0, 1)

        images = [content, style, github_output]
        labels = ["Content", "Style", "GitHub pretrained"]

        if trained_decoder is not None:
            trained_output = trained_decoder(target).clamp(0, 1)
            images.append(trained_output)
            labels.append("Your trained model")

        save_image(
            torch.cat(images, dim=0),
            OUTPUT_PATH,
            nrow=len(images)
        )

    print(f"\nComparison saved to: {OUTPUT_PATH}")
    print("Order: " + " | ".join(labels))


if __name__ == "__main__":
    main()
