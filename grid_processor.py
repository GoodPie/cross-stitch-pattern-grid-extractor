"""
High-level grid processing and extraction.
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from color_extraction import sample_cell_color
from grid_detection import detect_grid, detect_pattern_boundary
from models import GridCell, GridPage, rgb_to_hex
from pdf_converter import pdf_page_to_image


def extract_grid_cells(
        image: np.ndarray,
        x_positions: list[int],
        y_positions: list[int],
        debug: bool = False
) -> list[GridCell]:
    """
    Extract color data from each grid cell.

    Args:
        image: BGR image array
        x_positions: List of x-coordinates for vertical grid lines
        y_positions: List of y-coordinates for horizontal grid lines
        debug: Enable debug output

    Returns:
        List of GridCell objects with position and color data
    """
    cells = []

    for row, (y1, y2) in enumerate(zip(y_positions[:-1], y_positions[1:])):
        for col, (x1, x2) in enumerate(zip(x_positions[:-1], x_positions[1:])):
            width = x2 - x1
            height = y2 - y1

            color_rgb = sample_cell_color(image, x1, y1, width, height)
            color_hex = rgb_to_hex(color_rgb)

            cell = GridCell(
                row=row,
                col=col,
                x=x1,
                y=y1,
                width=width,
                height=height,
                color_rgb=color_rgb,
                color_hex=color_hex
            )
            cells.append(cell)

    if debug:
        print(f"  Extracted {len(cells)} cells")

    return cells


def process_image(
        image: np.ndarray,
        source_name: str,
        page_number: int = 1,
        debug: bool = False
) -> Optional[GridPage]:
    """
    Process an image and extract grid data.

    Args:
        image: BGR image array
        source_name: Name/identifier for the source (for debug messages)
        page_number: Page number to assign (default: 1)
        debug: Enable debug output

    Returns:
        GridPage object with extracted data, or None if processing fails
    """
    if debug:
        print(f"Processing {source_name}...")
        print(f"  Image size: {image.shape[1]} x {image.shape[0]}")

    # Detect grid structure
    x_positions, y_positions, cell_width, cell_height = detect_grid(image, debug=debug)

    if len(x_positions) < 2 or len(y_positions) < 2:
        print(f"Warning: Could not detect grid in {source_name}", file=sys.stderr)
        return None

    # Extract cell data
    cells = extract_grid_cells(image, x_positions, y_positions, debug)

    rows = len(y_positions) - 1
    cols = len(x_positions) - 1

    # Calculate grid corner coordinates
    grid_coordinates = {
        "top_left": (x_positions[0], y_positions[0]),
        "top_right": (x_positions[-1], y_positions[0]),
        "bottom_left": (x_positions[0], y_positions[-1]),
        "bottom_right": (x_positions[-1], y_positions[-1])
    }

    return GridPage(
        page_number=page_number,
        rows=rows,
        cols=cols,
        cell_width=cell_width,
        cell_height=cell_height,
        grid_coordinates=grid_coordinates,
        cells=cells
    )


def process_page(
        pdf_path: Path,
        page_number: int,
        dpi: int = 200,
        debug: bool = False
) -> Optional[GridPage]:
    """
    Process a single page and extract grid data.

    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to process (1-indexed)
        dpi: Resolution for PDF rendering
        debug: Enable debug output

    Returns:
        GridPage object with extracted data, or None if processing fails
    """
    # Convert PDF page to image
    try:
        image = pdf_page_to_image(pdf_path, page_number, dpi)
    except Exception as e:
        print(f"Error converting page {page_number}: {e}", file=sys.stderr)
        return None

    # Process the image
    return process_image(image, f"page {page_number}", page_number, debug)


def save_debug_image(
        pdf_path: Path,
        page_number: int,
        output_path: Path,
        dpi: int = 200
):
    """
    Save a debug image showing detected pattern boundary and grid lines.

    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to process (1-indexed)
        output_path: Path to save the debug image
        dpi: Resolution for PDF rendering
    """
    image = pdf_page_to_image(pdf_path, page_number, dpi)

    # Draw pattern boundary in red
    boundary = detect_pattern_boundary(image, debug=False)
    if boundary is not None:
        x, y, width, height = boundary
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)

    # Detect and draw grid lines
    x_positions, y_positions, _, _ = detect_grid(image, debug=False)

    # Draw detected grid lines in green
    for x in x_positions:
        cv2.line(image, (x, 0), (x, image.shape[0]), (0, 255, 0), 1)

    for y in y_positions:
        cv2.line(image, (0, y), (image.shape[1], y), (0, 255, 0), 1)

    cv2.imwrite(str(output_path), image)
    print(f"Debug image saved to {output_path}")
