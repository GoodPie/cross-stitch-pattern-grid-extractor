# Cross-Stitch Grid Extractor

A Python tool that extracts grid cell data (position, dimensions, and colors) from cross-stitch pattern PDFs and images. Perfect for digitizing patterns, analyzing color distributions, or converting patterns to other formats.

## Features

- **Automatic Grid Detection** - Uses OpenCV algorithms to identify grid lines in patterns
- **Multiple Input Sources** - Process PDFs or images from URLs
- **Color Extraction** - Samples and extracts RGB/hex colors from each grid cell
- **Flexible Page Selection** - Process individual pages or page ranges (for PDFs)
- **Multiple Output Formats** - Standard or compact JSON output (compact format reduces file size by 50-60%)
- **Debug Visualization** - Generate annotated images showing detected grid lines
- **High Quality Rendering** - Configurable DPI for accurate pattern extraction

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or uv package manager

### Install Dependencies

Using pip:
```bash
pip install opencv-python numpy pdf2image Pillow requests
```

Using uv:
```bash
uv pip install opencv-python numpy pdf2image Pillow requests
```

### System Requirements

The `pdf2image` library requires `poppler` to be installed:

- **macOS**: `brew install poppler`
- **Ubuntu/Debian**: `sudo apt-get install poppler-utils`
- **Windows**: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases)

## Usage

### Basic Usage

Extract grid data from a PDF:

```bash
python grid-extract.py pattern.pdf --pages 1-4
```

Extract grid data from an image URL:

```bash
python grid-extract.py --url https://example.com/pattern.jpg
```

### Command Line Options

```
python grid-extract.py [document] [options]

Input Arguments (one required):
  document              Path to the PDF document (optional if --url is provided)
  --url                 URL of an image to process

PDF-Specific Arguments:
  --pages, -p          Pages to process (e.g., "1", "1-4", "1,3,5-7") - required for PDF

Optional Arguments:
  --output, -o         Output JSON file path (default: <document>_grid.json or grid_output.json)
  --dpi                DPI for PDF rendering (default: 200)
  --compact            Use compact JSON format (reduces file size)
  --debug              Enable debug output
  --debug-image        Save debug images showing detected grid lines (PDF only)
```

### Examples

**PDF Processing:**

Extract a single page:
```bash
python grid-extract.py pattern.pdf --pages 1
```

Extract multiple pages with custom output:
```bash
python grid-extract.py pattern.pdf --pages 1-4 --output my_pattern.json
```

Extract with compact format and debug visualization:
```bash
python grid-extract.py pattern.pdf --pages 1,3,5 --compact --debug-image
```

High DPI extraction for fine details:
```bash
python grid-extract.py pattern.pdf --pages 1-10 --dpi 300
```

**URL Processing:**

Extract from a URL:
```bash
python grid-extract.py --url https://example.com/pattern.jpg
```

Extract from URL with custom output and compact format:
```bash
python grid-extract.py --url https://example.com/pattern.jpg --output my_pattern.json --compact
```

Extract from URL with debug output:
```bash
python grid-extract.py --url https://example.com/pattern.jpg --debug
```

## Output Format

### Standard Format

The standard JSON format contains detailed information for each cell:

```json
{
  "source_file": "pattern.pdf",
  "pages": [
    {
      "page_number": 1,
      "rows": 55,
      "cols": 56,
      "cell_width": 20.0,
      "cell_height": 20.0,
      "cells": [
        {
          "row": 0,
          "col": 0,
          "x": 268,
          "y": 628,
          "width": 20,
          "height": 20,
          "color_rgb": [255, 255, 255],
          "color_hex": "#ffffff"
        }
      ]
    }
  ]
}
```

### Compact Format

The compact format uses a color palette to reduce redundancy (use `--compact` flag):

```json
{
  "source_file": "pattern.pdf",
  "pages": [
    {
      "page_number": 1,
      "rows": 55,
      "cols": 56,
      "cell_width": 20.0,
      "cell_height": 20.0,
      "palette": ["#ffffff", "#000000", "#ff0000"],
      "cells": [
        [0, 0, 0],
        [0, 1, 0],
        [0, 2, 1]
      ]
    }
  ]
}
```

Each cell in compact format is represented as `[row, col, color_index]` where `color_index` refers to the palette array.

## How It Works

1. **Image Loading** - Loads images from PDF pages or URLs
2. **Grid Detection** - Uses autocorrelation and Hough line transform to detect grid lines
3. **Pattern Boundary Detection** - Identifies the actual pattern area within the image
4. **Color Sampling** - Samples the center region of each cell to extract colors
5. **Data Export** - Generates structured JSON output with all grid data

## Troubleshooting

### No grid detected

- Try increasing the DPI: `--dpi 300`
- Use `--debug-image` to visualize what lines are detected
- Ensure your PDF has clear, visible grid lines

### Colors appear incorrect

- The tool samples from the center of each cell to avoid grid lines
- Increase DPI for more accurate color sampling
- Check that your PDF is high quality (not scanned at low resolution)

### Large output files

- Use the `--compact` flag to reduce file size 
- The compact format is ideal for patterns with repeated colors

## Requirements

- Python >= 3.9
- opencv-python
- numpy
- pdf2image
- Pillow
- requests
- poppler (system dependency for PDF processing)

## Contributing

This is a personal project for extracting cross-stitch pattern data. Feel free to fork and adapt for your own use cases!
