"""FASHN Human Parser - SegFormer-based human parsing model."""

from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import SegformerForSemanticSegmentation

from .labels import IDS_TO_LABELS

# ImageNet normalization constants (same as training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model input size (matches training)
INPUT_HEIGHT = 576
INPUT_WIDTH = 384


class FashnHumanParser:
    """Human parsing model that segments images into 18 semantic classes.

    This class provides the exact preprocessing used during training for
    maximum accuracy. For quick experimentation, you can also use the
    HuggingFace pipeline API directly.

    Args:
        model_id: HuggingFace model ID. Default: "fashn-ai/fashn-human-parser"
        device: Device to run inference on. Default: "cuda" if available, else "cpu"

    Example:
        >>> parser = FashnHumanParser(device="cuda")
        >>> segmentation = parser.predict(image)
        >>> # segmentation is a numpy array of shape (H, W) with values 0-17
    """

    def __init__(
        self,
        model_id: str = "fashn-ai/fashn-human-parser",
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    def _preprocess_single(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image for model input.

        Matches the exact preprocessing used during training:
        1. Resize to 384x576 with cv2.INTER_AREA
        2. Normalize with ImageNet mean/std
        3. Convert to CHW tensor

        Args:
            image: RGB image as numpy array (H, W, 3), uint8

        Returns:
            Preprocessed tensor of shape (3, 576, 384)
        """
        # Resize with INTER_AREA (matches training validation transform)
        resized = cv2.resize(
            image,
            (INPUT_WIDTH, INPUT_HEIGHT),  # cv2 uses (width, height)
            interpolation=cv2.INTER_AREA
        )

        # Convert to float32 and normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

        # Convert HWC to CHW
        transposed = normalized.transpose(2, 0, 1)

        return torch.from_numpy(transposed)

    def _to_numpy(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        """Convert various image formats to RGB numpy array.

        Args:
            image: Input image. Must be RGB format (not BGR from cv2.imread).

        Returns:
            RGB numpy array with dtype uint8.

        Note:
            If using cv2.imread, convert to RGB first: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        """
        if isinstance(image, str):
            # Load from file path with proper resource management and EXIF handling
            with Image.open(image) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                image = np.array(pil_img.convert("RGB"))
        elif isinstance(image, Image.Image):
            # User-provided PIL Image - they handle EXIF themselves
            image = np.array(image.convert("RGB"))

        if isinstance(image, np.ndarray):
            # Validate array shape
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] != 3:
                    raise ValueError(
                        f"Expected RGB image with 3 channels, got {image.shape[2]} channels"
                    )
            else:
                raise ValueError(
                    f"Expected 2D or 3D image array, got {image.ndim}D array"
                )

            # Handle float arrays (assume [0, 1] range)
            if image.dtype in (np.float32, np.float64):
                image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            raise TypeError(
                f"Unsupported image type: {type(image).__name__}. "
                "Expected PIL Image, numpy array, or file path string."
            )

        return image

    @torch.inference_mode()
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str, List],
        return_logits: bool = False,
    ) -> Union[np.ndarray, torch.Tensor, List]:
        """Run human parsing on one or more images.

        Args:
            image: Input image(s) - can be:
                   - Single: PIL Image, numpy array (RGB), or file path
                   - Batch: List of any of the above

                   Note: NumPy arrays must be RGB (not BGR from cv2.imread).
                   Use cv2.cvtColor(img, cv2.COLOR_BGR2RGB) to convert.

            return_logits: If True, return raw logits instead of class predictions

        Returns:
            If single image:
                - return_logits=False: numpy array of shape (H, W) with class IDs (0-17)
                - return_logits=True: torch tensor of shape (1, 18, H, W) with logits
            If batch:
                - List of the above
        """
        # Handle batch vs single
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]

        # Validate inputs
        if len(images) == 0:
            return []
        for i, img in enumerate(images):
            if img is None:
                raise ValueError(f"Image at index {i} is None")

        # Convert all to numpy RGB
        images_np = [self._to_numpy(img) for img in images]
        original_sizes = [(img.shape[0], img.shape[1]) for img in images_np]  # (H, W)

        # Preprocess all images
        batch_tensors = [self._preprocess_single(img) for img in images_np]

        # Stack into batch and move to device
        pixel_values = torch.stack(batch_tensors).to(self.device)

        # Match model dtype
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=model_dtype)

        # Single forward pass for entire batch
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits

        # Process results for each image
        results = []
        for i, size in enumerate(original_sizes):
            img_logits = logits[i:i+1]

            # Upsample to original size
            upsampled = torch.nn.functional.interpolate(
                img_logits,
                size=size,
                mode="bilinear",
                align_corners=False,
            )

            if return_logits:
                results.append(upsampled)
            else:
                # Get class predictions
                pred_seg = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
                results.append(pred_seg)

        return results if is_batch else results[0]

    @staticmethod
    def get_label_name(label_id: int) -> str:
        """Get the label name for a given ID."""
        return IDS_TO_LABELS.get(label_id, "unknown")

    @staticmethod
    def get_labels() -> dict:
        """Get the full ID to label mapping."""
        return IDS_TO_LABELS.copy()
