# Neural Style Transfer using AdaIN

A PyTorch implementation of **Neural Style Transfer using Adaptive Instance Normalization (AdaIN)**. The project transfers the artistic style of a reference image onto a content image while preserving the content image's structure.

## Overview

Neural Style Transfer combines:

- **Content Image:** The image whose structure and objects should be preserved.
- **Style Image:** The image whose artistic appearance, textures, and colors should be transferred.
- **AdaIN:** Aligns the channel-wise mean and standard deviation of content features with those of style features.
- **VGG Encoder:** Extracts features from images using pretrained VGG weights.
- **Decoder:** Reconstructs the stylized RGB image from the AdaIN-transformed features.

## How It Works

1. Load the content and style images.
2. Pass both images through the pretrained VGG encoder.
3. Apply AdaIN to align the content feature statistics with the style feature statistics.
4. Pass the transformed features through the decoder to generate a stylized image.
5. Encode the generated image again to calculate content and style losses.
6. Train the decoder using backpropagation while keeping the VGG encoder frozen.

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow (PIL)
- tqdm
