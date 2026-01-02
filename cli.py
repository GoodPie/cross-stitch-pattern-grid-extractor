"""
Command-line interface for grid extraction.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from grid_processor import process_page, save_debug_image
from models import ExtractionResult


def parse_pages(pages_str: str) -> list[int]:
    """
    Parse page specification string.

    Examples:
        "1" -> [1]
        "1,3,5" -> [1, 3, 5]
        "1-4" -> [1, 2, 3, 4]
        "1-3,5,7-9" -> [1, 2, 3, 5, 7, 8, 9]

    Args:
        pages_str: String specification of pages

    Returns:
        Sorted list of unique page numbers

    Raises:
        ValueError: If the page specification is invalid
    """
    pages = []

    for part in pages_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))

    return sorted(set(pages))


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Extract grid data from cross-stitch pattern PDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s pattern.pdf --pages 1-4
  %(prog)s pattern.pdf --pages 1,3,5 --output grid_data.json
  %(prog)s pattern.pdf --pages 1 --debug --debug-image
        """
    )

    parser.add_argument(
        'document',
        type=Path,
        help='Path to the PDF document'
    )

    parser.add_argument(
        '--pages', '-p',
        type=str,
        required=True,
        help='Pages to process (e.g., "1", "1-4", "1,3,5-7")'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output JSON file path (default: <document>_grid.json)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='DPI for PDF rendering (default: 200)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )

    parser.add_argument(
        '--debug-image',
        action='store_true',
        help='Save debug images showing detected grid lines'
    )

    parser.add_argument(
        '--compact',
        action='store_true',
        help='Use compact JSON format (reduces file size by compressing the output JSON)'
    )

    args = parser.parse_args()

    # Validate input
    if not args.document.exists():
        print(f"Error: File not found: {args.document}", file=sys.stderr)
        sys.exit(1)

    # Parse pages
    try:
        pages = parse_pages(args.pages)
    except ValueError as e:
        print(f"Error parsing pages: {e}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        print(f"Processing pages: {pages}")

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = args.document.with_suffix('.grid.json')

    # Process each page
    results = []

    for page_num in pages:
        # Save debug image if requested
        if args.debug_image:
            debug_path = args.document.parent / f"{args.document.stem}_page{page_num}_debug.png"
            try:
                save_debug_image(args.document, page_num, debug_path, args.dpi)
            except Exception as e:
                print(f"Error saving debug image: {e}", file=sys.stderr)

        # Process page
        grid_page = process_page(args.document, page_num, args.dpi, args.debug)

        if grid_page:
            results.append(grid_page)
            if args.debug:
                print(f"  Page {page_num}: {grid_page.rows}x{grid_page.cols} grid")

    if not results:
        print("No grid data extracted", file=sys.stderr)
        sys.exit(1)

    # Build result
    extraction_result = ExtractionResult(
        source_file=str(args.document),
        pages=results
    )

    # Save output
    with open(output_path, 'w') as f:
        if args.compact:
            json.dump(extraction_result.to_compact_dict(), f, indent=2)
        else:
            json.dump(asdict(extraction_result), f, indent=2)

    print(f"Grid data saved to {output_path}")
    print(f"Processed {len(results)} page(s)")

    # Summary
    total_cells = sum(len(page.cells) for page in results)
    print(f"Total cells extracted: {total_cells}")


if __name__ == '__main__':
    main()
