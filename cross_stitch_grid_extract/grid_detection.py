"""
Grid detection algorithms for cross-stitch patterns.

Provides multiple methods for detecting grid structure in images:
- Hough line detection
- Frequency analysis
- Autocorrelation-based edge detection
"""

from typing import Optional

import cv2
import numpy as np


def detect_lines(image: np.ndarray, debug: bool = False) -> tuple[list, list]:
    """
    Detect horizontal and vertical lines using Hough transform.

    Args:
        image: BGR image array
        debug: Enable debug output

    Returns:
        Tuple of (horizontal_lines, vertical_lines) where each line is (x1, y1, x2, y2)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10
    )

    if lines is None:
        return [], []

    horizontal = []
    vertical = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate angle
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        # Classify as horizontal or vertical (with 10 degree tolerance)
        if angle < 10 or angle > 170:
            horizontal.append((x1, y1, x2, y2))
        elif 80 < angle < 100:
            vertical.append((x1, y1, x2, y2))

    if debug:
        print(f"  Detected {len(horizontal)} horizontal, {len(vertical)} vertical lines")

    return horizontal, vertical


def detect_grid_by_frequency(image: np.ndarray, debug: bool = False) -> tuple[list[int], list[int]]:
    """
    Detect grid by analyzing pixel intensity patterns.
    Uses the fact that grid lines create regular intensity valleys.

    Args:
        image: BGR image array
        debug: Enable debug output

    Returns:
        Tuple of (x_positions, y_positions)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sum along each axis to find where grid lines are
    # Grid lines are typically darker, so they create valleys
    horizontal_profile = np.mean(gray, axis=1)  # Average across columns -> vertical position
    vertical_profile = np.mean(gray, axis=0)  # Average across rows -> horizontal position

    # Find valleys (local minima) in the profiles
    def find_valleys(profile: np.ndarray, min_distance: int = 5) -> list[int]:
        """Find local minima in a 1D profile."""
        # Smooth the profile
        kernel_size = 3
        smoothed = np.convolve(profile, np.ones(kernel_size) / kernel_size, mode='same')

        # Find local minima
        valleys = []
        for i in range(min_distance, len(smoothed) - min_distance):
            window = smoothed[i - min_distance:i + min_distance + 1]
            if smoothed[i] == np.min(window) and smoothed[i] < np.mean(smoothed):
                valleys.append(i)

        return valleys

    y_positions = find_valleys(horizontal_profile)
    x_positions = find_valleys(vertical_profile)

    if debug:
        print(f"  Frequency analysis: {len(x_positions)} x, {len(y_positions)} y positions")

    return x_positions, y_positions


def detect_grid_by_cell_edges(
        image: np.ndarray,
        min_cell_size: int = 8,
        max_cell_size: int = 30,
        debug: bool = False
) -> tuple[list[int], list[int], float, float]:
    """
    Detect grid by finding regular patterns in edge density.
    Works better for coloured cell patterns where lines between cells may be thin.

    Args:
        image: BGR image array
        min_cell_size: Minimum expected cell size in pixels
        max_cell_size: Maximum expected cell size in pixels
        debug: Enable debug output

    Returns:
        Tuple of (x_positions, y_positions, cell_width, cell_height)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 30, 100)

    # Project edges onto axes
    v_projection = np.sum(edges, axis=0)  # Sum down columns -> x positions
    h_projection = np.sum(edges, axis=1)  # Sum across rows -> y positions

    def find_grid_spacing(projection: np.ndarray, min_size: int, max_size: int) -> Optional[int]:
        """Use autocorrelation to find the dominant grid spacing."""
        # Normalize
        proj = projection.astype(float)
        proj = proj - np.mean(proj)

        # Autocorrelation
        correlation = np.correlate(proj, proj, mode='full')
        correlation = correlation[len(correlation) // 2:]  # Take positive lags only

        # Find peaks in autocorrelation (these indicate periodic spacing)
        peaks = []
        for i in range(min_size, min(max_size * 3, len(correlation) - 1)):
            if correlation[i] > correlation[i - 1] and correlation[i] > correlation[i + 1]:
                if correlation[i] > 0.1 * correlation[0]:  # Significant peak
                    peaks.append((i, correlation[i]))

        if not peaks:
            return None

        # Find the most likely cell size (first significant peak)
        peaks.sort(key=lambda x: x[1], reverse=True)
        for lag, _ in peaks:
            if min_size <= lag <= max_size:
                return lag

        return peaks[0][0] if peaks else None

    cell_width = find_grid_spacing(v_projection, min_cell_size, max_cell_size)
    cell_height = find_grid_spacing(h_projection, min_cell_size, max_cell_size)

    if debug:
        print(f"  Detected cell size via autocorrelation: {cell_width} x {cell_height}")

    if cell_width is None or cell_height is None:
        return [], [], 0, 0

    # Find grid origin by looking for strong edge response
    def find_grid_positions(projection: np.ndarray, spacing: int) -> list[int]:
        """Find grid line positions given known spacing."""
        # Find the best starting offset
        best_offset = 0
        best_score = 0

        for offset in range(spacing):
            positions = list(range(offset, len(projection), spacing))
            score = sum(projection[p] for p in positions if p < len(projection))
            if score > best_score:
                best_score = score
                best_offset = offset

        # Generate positions
        positions = list(range(best_offset, len(projection), spacing))
        return positions

    x_positions = find_grid_positions(v_projection, cell_width)
    y_positions = find_grid_positions(h_projection, cell_height)

    if debug:
        print(f"  Grid positions: {len(x_positions)} x, {len(y_positions)} y")

    return x_positions, y_positions, float(cell_width), float(cell_height)


def cluster_lines(lines: list, axis: int, tolerance: int = 10) -> list[int]:
    """
    Cluster lines by their position on the given axis and return unique positions.

    Args:
        lines: List of (x1, y1, x2, y2) tuples
        axis: 0 for x (vertical lines), 1 for y (horizontal lines)
        tolerance: Maximum distance to consider lines as the same

    Returns:
        Sorted list of unique line positions
    """
    if not lines:
        return []

    # Extract positions (average of start and end for the relevant axis)
    if axis == 0:  # Vertical lines - use x coordinate
        positions = [(line[0] + line[2]) / 2 for line in lines]
    else:  # Horizontal lines - use y coordinate
        positions = [(line[1] + line[3]) / 2 for line in lines]

    positions = sorted(positions)

    # Cluster nearby positions
    clusters = []
    current_cluster = [positions[0]]

    for pos in positions[1:]:
        if pos - current_cluster[-1] <= tolerance:
            current_cluster.append(pos)
        else:
            clusters.append(int(np.mean(current_cluster)))
            current_cluster = [pos]

    clusters.append(int(np.mean(current_cluster)))

    return clusters


def calculate_grid_spacing(positions: list[int]) -> Optional[float]:
    """
    Calculate the most common spacing between grid lines.

    Args:
        positions: List of grid line positions

    Returns:
        Median spacing between consecutive positions, or None if insufficient data
    """
    if len(positions) < 2:
        return None

    spacings = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]

    if not spacings:
        return None

    # Use median to be robust against outliers
    return float(np.median(spacings))


def detect_pattern_boundary(image: np.ndarray, debug: bool = False) -> Optional[tuple[int, int, int, int]]:
    """
    Detect the boundary of the cross-stitch pattern in the image.

    Tries two methods:
    1. Detect thick black border contours (typical for pattern charts)
    2. Fallback: Find bounding box of colored content

    Args:
        image: BGR image array
        debug: Enable debug output

    Returns:
        Tuple of (x, y, width, height) for the pattern boundary, or None if not found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Method 1: Detect thick black border using contours
    # Apply threshold to isolate dark lines
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Look for large rectangular contours
    best_rect = None
    best_area = 0
    min_area = (w * h) * 0.1  # At least 10% of image
    max_area = (w * h) * 0.9  # At most 90% of image

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check if it's roughly rectangular (4-8 vertices is acceptable)
            if 4 <= len(approx) <= 8:
                x, y, bw, bh = cv2.boundingRect(contour)

                # Check aspect ratio is reasonable (not too extreme)
                aspect = bw / bh if bh > 0 else 0
                if 0.3 < aspect < 3.0 and area > best_area:
                    best_area = area
                    best_rect = (x, y, bw, bh)

    if best_rect is not None:
        if debug:
            print(f"  Found pattern boundary via contour: {best_rect}")
        return best_rect

    # Method 2: Content-based detection (fallback)
    # Find all non-white pixels (content)
    # White pixels typically have high values (>240 in grayscale)
    content_mask = gray < 240

    # Find bounding box of content
    coords = np.column_stack(np.where(content_mask))
    if len(coords) == 0:
        if debug:
            print("  No pattern boundary found")
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add small margin
    margin = 10
    x = max(0, x_min - margin)
    y = max(0, y_min - margin)
    width = min(w - x, x_max - x_min + 2 * margin)
    height = min(h - y, y_max - y_min + 2 * margin)

    # Verify this is a reasonable region (not too small, not entire image)
    content_area = width * height
    if content_area < min_area or content_area > max_area:
        if debug:
            print("  Content-based boundary rejected (invalid size)")
        return None

    if debug:
        print(f"  Found pattern boundary via content: ({x}, {y}, {width}, {height})")

    return (x, y, width, height)


def detect_grid(
        image: np.ndarray,
        method: str = "auto",
        crop_to_pattern: bool = True,
        debug: bool = False
) -> tuple[list[int], list[int], float, float]:
    """
    Detect the grid structure in the image.

    Args:
        image: BGR image array
        method: Detection method - "hough", "autocorr", or "auto"
        crop_to_pattern: If True, detect pattern boundary and crop before grid detection
        debug: Enable debug output

    Returns:
        Tuple of (x_positions, y_positions, cell_width, cell_height)
    """
    # Crop to pattern boundary if requested
    offset_x = 0
    offset_y = 0

    if crop_to_pattern:
        boundary = detect_pattern_boundary(image, debug)
        if boundary is not None:
            x, y, width, height = boundary
            image = image[y:y+height, x:x+width]
            offset_x = x
            offset_y = y
            if debug:
                print(f"  Cropped to pattern boundary: {width}x{height} at ({x}, {y})")
        elif debug:
            print(f"  No pattern boundary detected, using full image")

    if method == "auto":
        # Try autocorrelation first (works better for coloured cells)
        x_pos, y_pos, cw, ch = detect_grid_by_cell_edges(image, debug=debug)

        if len(x_pos) > 5 and len(y_pos) > 5:
            if debug:
                print(f"  Using autocorrelation method")
            # Adjust positions for crop offset
            x_pos = [x + offset_x for x in x_pos]
            y_pos = [y + offset_y for y in y_pos]
            return x_pos, y_pos, cw, ch

        # Fall back to Hough line detection
        if debug:
            print(f"  Falling back to Hough line detection")
        method = "hough"

    if method == "autocorr":
        x_pos, y_pos, cw, ch = detect_grid_by_cell_edges(image, debug=debug)
        # Adjust positions for crop offset
        x_pos = [x + offset_x for x in x_pos]
        y_pos = [y + offset_y for y in y_pos]
        return x_pos, y_pos, cw, ch

    # Hough line method
    horizontal, vertical = detect_lines(image, debug)

    # Cluster lines to get unique grid positions
    x_positions = cluster_lines(vertical, axis=0)
    y_positions = cluster_lines(horizontal, axis=1)

    if debug:
        print(f"  Found {len(x_positions)} vertical grid lines")
        print(f"  Found {len(y_positions)} horizontal grid lines")

    # Calculate cell dimensions
    cell_width = calculate_grid_spacing(x_positions) or 0
    cell_height = calculate_grid_spacing(y_positions) or 0

    if debug:
        print(f"  Cell size: {cell_width:.1f} x {cell_height:.1f} pixels")

    # Adjust positions for crop offset
    x_positions = [x + offset_x for x in x_positions]
    y_positions = [y + offset_y for y in y_positions]

    return x_positions, y_positions, cell_width, cell_height
