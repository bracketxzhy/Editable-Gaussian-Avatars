---
license: other
license_name: nvidia-segformer
license_link: https://github.com/NVlabs/SegFormer/blob/master/LICENSE
library_name: transformers
pipeline_tag: image-segmentation
tags:
  - segformer
  - human-parsing
  - semantic-segmentation
  - fashion
  - virtual-try-on
language:
  - en
---

# FASHN Human Parser

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/fashn-ai/fashn-human-parser)

A SegFormer-B4 model fine-tuned for human parsing with 18 semantic classes, optimized for fashion and virtual try-on applications.

<p align="center">
  <img src="https://static.fashn.ai/repositories/fashn-human-parser/example.webp" alt="Human Parsing Example" width="800">
</p>

## Model Description

This model segments human images into 18 semantic categories including body parts (face, hair, arms, hands, legs, feet, torso), clothing items (top, dress, skirt, pants, belt, scarf), and accessories (bag, hat, glasses, jewelry).

- **Architecture**: SegFormer-B4 (MIT-B4 encoder + MLP decoder)
- **Input Size**: 384 x 576 (width x height)
- **Output**: 18-class semantic segmentation mask
- **Base Model**: [nvidia/mit-b4](https://huggingface.co/nvidia/mit-b4)

## Usage

### Quick Start with Pipeline

```python
from transformers import pipeline

pipe = pipeline("image-segmentation", model="fashn-ai/fashn-human-parser")
result = pipe("image.jpg")
# result is a list of dicts with 'label', 'score', 'mask' for each detected class
```

The pipeline automatically manages GPU/CPU and returns per-class masks at the original image resolution.

### Explicit Usage

```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import torch

# Load model and processor
processor = SegformerImageProcessor.from_pretrained("fashn-ai/fashn-human-parser")
model = SegformerForSemanticSegmentation.from_pretrained("fashn-ai/fashn-human-parser")

# Load and preprocess image
image = Image.open("path/to/image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # (1, 18, H/4, W/4)

# Upsample to original size and get predictions
upsampled = torch.nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False
)
predictions = upsampled.argmax(dim=1).squeeze().numpy()
```

### Production Usage (Recommended)

For maximum accuracy, use our Python package which implements the exact preprocessing used during training:

```bash
pip install fashn-human-parser
```

```python
from fashn_human_parser import FashnHumanParser

parser = FashnHumanParser()  # auto-detects GPU
segmentation = parser.predict("image.jpg")
# segmentation is a numpy array of shape (H, W) with class IDs 0-17
```

The package uses `cv2.INTER_AREA` for resizing (matching training), while the HuggingFace pipeline uses PIL LANCZOS.

## Label Definitions

| ID | Label |
|----|-------|
| 0 | background |
| 1 | face |
| 2 | hair |
| 3 | top |
| 4 | dress |
| 5 | skirt |
| 6 | pants |
| 7 | belt |
| 8 | bag |
| 9 | hat |
| 10 | scarf |
| 11 | glasses |
| 12 | arms |
| 13 | hands |
| 14 | legs |
| 15 | feet |
| 16 | torso |
| 17 | jewelry |

### Category Mappings

For virtual try-on applications:

| Category | Body Coverage | Relevant Labels |
|----------|--------------|-----------------|
| Tops | Upper body | top, dress, scarf |
| Bottoms | Lower body | skirt, pants, belt |
| One-pieces | Full body | top, dress, scarf, skirt, pants, belt |

### Identity Labels

Labels typically preserved during virtual try-on: `face`, `hair`, `jewelry`, `bag`, `glasses`, `hat`

## Training

This model was fine-tuned on a proprietary dataset curated and annotated by FASHN AI, specifically designed for virtual try-on applications. The 18-class label schema was developed to capture the semantic regions most relevant for clothing transfer and human body understanding in fashion contexts.

## Limitations

- Optimized for single-person images with clear visibility
- Best results on fashion/e-commerce style photography
- Input images are resized to 384x576; very small subjects may lose detail

## Citation

```bibtex
@misc{fashn-human-parser,
  author = {FASHN AI},
  title = {FASHN Human Parser: SegFormer for Fashion Human Parsing},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/fashn-ai/fashn-human-parser}
}
```

## License

This model inherits the [NVIDIA Source Code License for SegFormer](https://github.com/NVlabs/SegFormer/blob/master/LICENSE). Please review the license terms before use.

## Links

- [FASHN AI](https://fashn.ai/)
- [Interactive Demo](https://huggingface.co/spaces/fashn-ai/fashn-human-parser)
- [GitHub Repository](https://github.com/fashn-AI/fashn-human-parser)
- [PyPI Package](https://pypi.org/project/fashn-human-parser/)
