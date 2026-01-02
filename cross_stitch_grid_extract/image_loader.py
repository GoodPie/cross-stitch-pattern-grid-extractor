"""
Image loading utilities for URLs and local files.
"""

from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image


def load_image_from_url(url: str, timeout: int = 30) -> np.ndarray:
    """
    Load an image from a URL and convert to BGR numpy array for OpenCV.

    Args:
        url: URL of the image to load
        timeout: Request timeout in seconds

    Returns:
        BGR image array suitable for OpenCV processing

    Raises:
        requests.RequestException: If the image cannot be fetched
        ValueError: If the image cannot be decoded
    """
    # Fetch the image
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    # Convert to PIL Image
    try:
        pil_image = Image.open(BytesIO(response.content))
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Could not decode image from {url}: {e}")

    # Convert PIL Image to numpy array (RGB -> BGR for OpenCV)
    rgb_array = np.array(pil_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array
