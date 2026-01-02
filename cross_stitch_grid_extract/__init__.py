"""
Cross-stitch grid extraction package.

Extract grid data from cross-stitch pattern PDFs and images using OpenCV.
"""

from .models import GridCell, CompactGridCell, GridPage, ExtractionResult, rgb_to_hex, quantize_color
from .grid_processor import process_image, process_page, save_debug_image
from .image_loader import load_image_from_url

__version__ = "0.0.1"

__all__ = [
    "GridCell",
    "CompactGridCell",
    "GridPage",
    "ExtractionResult",
    "rgb_to_hex",
    "quantize_color",
    "process_image",
    "process_page",
    "save_debug_image",
    "load_image_from_url",
]
