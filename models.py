"""
Data models for cross-stitch grid extraction.
"""

from dataclasses import dataclass, asdict


@dataclass
class GridCell:
    """Represents a single cell in the cross-stitch grid."""
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    color_rgb: tuple[int, int, int]
    color_hex: str


@dataclass
class CompactGridCell:
    """Compact representation of a grid cell (for API output)."""
    row: int
    col: int
    color: str  # hex color string


@dataclass
class GridPage:
    """Represents extracted grid data from a single page."""
    page_number: int
    rows: int
    cols: int
    cell_width: float
    cell_height: float
    cells: list[GridCell]

    def to_compact_dict(self) -> dict:
        """Convert to compact dictionary format for smaller JSON output."""
        # Build color palette
        color_set = {}
        color_palette = []

        for cell in self.cells:
            hex_color = rgb_to_hex(cell.color_rgb)
            if hex_color not in color_set:
                color_set[hex_color] = len(color_palette)
                color_palette.append(hex_color)

        # Create compact cells (row, col, color_index)
        compact_cells = [
            [cell.row, cell.col, color_set[rgb_to_hex(cell.color_rgb)]]
            for cell in self.cells
        ]

        return {
            "page_number": self.page_number,
            "rows": self.rows,
            "cols": self.cols,
            "cell_width": self.cell_width,
            "cell_height": self.cell_height,
            "palette": color_palette,
            "cells": compact_cells
        }


@dataclass
class ExtractionResult:
    """Complete extraction result."""
    source_file: str
    pages: list[GridPage]

    def to_compact_dict(self) -> dict:
        """Convert to compact dictionary format for smaller JSON output."""
        return {
            "source_file": self.source_file,
            "pages": [page.to_compact_dict() for page in self.pages]
        }


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def quantize_color(rgb: tuple[int, int, int], levels: int = 32) -> tuple[int, int, int]:
    """
    Quantize color to reduce near-duplicates.

    Args:
        rgb: RGB tuple
        levels: Number of levels per channel (lower = more quantization)

    Returns:
        Quantized RGB tuple
    """
    step = 256 // levels
    return (
        (rgb[0] // step) * step + step // 2,
        (rgb[1] // step) * step + step // 2,
        (rgb[2] // step) * step + step // 2,
    )
