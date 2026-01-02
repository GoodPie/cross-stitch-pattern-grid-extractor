"""
PDF to image conversion utilities.
"""

from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path


def pdf_page_to_image(pdf_path: Path, page_number: int, dpi: int = 200) -> np.ndarray:
    """
    Convert a single PDF page to a numpy array (BGR format for OpenCV).

    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to convert (1-indexed)
        dpi: Resolution for rendering

    Returns:
        BGR image array suitable for OpenCV processing

    Raises:
        ValueError: If the page cannot be extracted
    """
    images = convert_from_path(
        pdf_path,
        first_page=page_number,
        last_page=page_number,
        dpi=dpi
    )
    if not images:
        raise ValueError(f"Could not extract page {page_number} from {pdf_path}")

    # Convert PIL Image to numpy array (RGB -> BGR for OpenCV)
    pil_image = images[0]
    rgb_array = np.array(pil_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array
