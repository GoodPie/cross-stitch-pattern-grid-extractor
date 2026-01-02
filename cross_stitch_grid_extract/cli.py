"""
Command-line interface for grid extraction.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .grid_processor import process_image, process_page, save_debug_image
from .image_loader import load_image_from_url
from .models import ExtractionResult


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
  %(prog)s --url https://example.com/pattern.jpg --output grid_data.json
        """
    )

    parser.add_argument(
        'document',
        type=Path,
        nargs='?',
        help='Path to the PDF document'
    )

    parser.add_argument(
        '--url',
        type=str,
        help='URL of an image to process'
    )

    parser.add_argument(
        '--pages', '-p',
        type=str,
        help='Pages to process (e.g., "1", "1-4", "1,3,5-7") - required for PDF processing'
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

    # Validate input - must have either document or URL
    if not args.document and not args.url:
        print("Error: Must provide either a document path or --url", file=sys.stderr)
        sys.exit(1)

    if args.document and args.url:
        print("Error: Cannot specify both document and --url", file=sys.stderr)
        sys.exit(1)

    # Process based on input type
    results = []
    source_file = None

    if args.url:
        # Process URL
        if args.pages:
            print("Warning: --pages argument is ignored for URL processing", file=sys.stderr)

        if args.debug_image:
            print("Warning: --debug-image is not supported for URL processing", file=sys.stderr)

        if args.debug:
            print(f"Fetching image from URL: {args.url}")

        try:
            image = load_image_from_url(args.url)
        except Exception as e:
            print(f"Error loading image from URL: {e}", file=sys.stderr)
            sys.exit(1)

        # Process the image
        grid_page = process_image(image, args.url, page_number=1, debug=args.debug)

        if grid_page:
            results.append(grid_page)
            if args.debug:
                print(f"  {grid_page.rows}x{grid_page.cols} grid")

        source_file = args.url

        # Determine output path
        output_path = args.output
        if output_path is None:
            output_path = Path('grid_output.json')

    else:
        # Process PDF document
        if not args.document.exists():
            print(f"Error: File not found: {args.document}", file=sys.stderr)
            sys.exit(1)

        if not args.pages:
            print("Error: --pages argument is required for PDF processing", file=sys.stderr)
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

        source_file = str(args.document)

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
