"""
Color extraction and sampling from grid cells.
"""

import numpy as np

from models import rgb_to_hex


def sample_cell_color(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        margin: float = 0.2
) -> tuple[int, int, int]:
    """
    Sample the dominant color from a cell, avoiding edges.

    Args:
        image: BGR image array
        x, y: Top-left corner of cell
        width, height: Cell dimensions
        margin: Fraction of cell to exclude from edges (0-0.5)

    Returns:
        RGB tuple of the dominant color
    """
    # Calculate sample region (avoiding edges)
    margin_x = int(width * margin)
    margin_y = int(height * margin)

    x1 = max(0, x + margin_x)
    y1 = max(0, y + margin_y)
    x2 = min(image.shape[1], x + width - margin_x)
    y2 = min(image.shape[0], y + height - margin_y)

    if x2 <= x1 or y2 <= y1:
        # Fall back to center pixel
        cx = min(max(0, x + width // 2), image.shape[1] - 1)
        cy = min(max(0, y + height // 2), image.shape[0] - 1)
        bgr = image[cy, cx]
    else:
        # Sample region and get median color
        region = image[y1:y2, x1:x2]
        bgr = np.median(region.reshape(-1, 3), axis=0).astype(int)

    # Convert BGR to RGB
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))
