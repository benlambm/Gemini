#!/usr/bin/env python3
"""Upload images to ImageKit and print hosted URLs.

Reads IMAGEKIT_PRIVATE_KEY from environment (required).
Reads IMAGEKIT_FOLDER from environment (optional, default /lms-content/).
"""
import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path

from imagekitio import ImageKit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload images to ImageKit and get hosted URLs.",
        epilog="Env vars: IMAGEKIT_PRIVATE_KEY (required), IMAGEKIT_FOLDER (optional)",
    )
    parser.add_argument("files", nargs="+", type=Path, help="Image file(s) to upload")
    parser.add_argument(
        "--folder",
        default=None,
        help="ImageKit folder (default: $IMAGEKIT_FOLDER or /lms-content/)",
    )
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument(
        "--private", action="store_true", help="Upload as private file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output full response as JSON",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output URLs, no status messages",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    private_key = os.environ.get("IMAGEKIT_PRIVATE_KEY")
    if not private_key:
        print("Error: IMAGEKIT_PRIVATE_KEY environment variable not set", file=sys.stderr)
        return 1

    folder = args.folder or os.environ.get("IMAGEKIT_FOLDER", "/lms-content/")
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else ["upload-cli"]

    # Validate all files exist before uploading any
    for f in args.files:
        if not f.is_file():
            print(f"Error: File not found: {f}", file=sys.stderr)
            return 1
        mime, _ = mimetypes.guess_type(str(f))
        if mime and not mime.startswith("image/"):
            print(f"Warning: {f} may not be an image (detected: {mime})", file=sys.stderr)

    ik = ImageKit(private_key=private_key)

    exit_code = 0
    for f in args.files:
        if not args.quiet:
            print(f"Uploading {f.name}...", file=sys.stderr)
        try:
            result = ik.files.upload(
                file=f.read_bytes(),
                file_name=f.name,
                folder=folder,
                use_unique_file_name=True,
                is_private_file=args.private,
                tags=tags,
            )

            if result and hasattr(result, "url") and result.url:
                if args.json_output:
                    print(json.dumps({
                        "file": f.name,
                        "url": result.url,
                        "file_id": getattr(result, "file_id", None),
                    }))
                else:
                    print(result.url)
            else:
                print(f"Error: No URL returned for {f.name}", file=sys.stderr)
                exit_code = 1
        except Exception as e:
            print(f"Error uploading {f.name}: {e}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
