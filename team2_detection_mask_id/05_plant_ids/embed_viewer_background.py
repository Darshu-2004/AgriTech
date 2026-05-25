from __future__ import annotations

import argparse
import base64
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a new standalone Plant ID viewer with the background image embedded into the HTML."
    )
    parser.add_argument("--viewer", required=True, help="Path to the existing viewer.html file.")
    parser.add_argument("--image", required=True, help="Path to the background PNG image to embed.")
    parser.add_argument("--output", required=True, help="Path to the new embedded viewer HTML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer_path = Path(args.viewer).resolve()
    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()

    if not viewer_path.exists():
        raise FileNotFoundError(f"Viewer not found: {viewer_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Background image not found: {image_path}")

    html = viewer_path.read_text(encoding="utf-8")
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_url = f"data:image/png;base64,{image_b64}"

    original_line = '          url: BACKGROUND_OVERLAY.image_path,'
    replacement_line = '          url: (BACKGROUND_OVERLAY.image_url || BACKGROUND_OVERLAY.image_path),'
    if original_line not in html:
        raise RuntimeError("Could not find the background image URL line in the source viewer.")
    html = html.replace(original_line, replacement_line, 1)

    marker = "const BACKGROUND_OVERLAY = "
    if marker not in html:
        raise RuntimeError("Could not find BACKGROUND_OVERLAY in the source viewer.")

    start = html.index(marker) + len(marker)
    end = html.index(";", start)
    object_literal = html[start:end]
    if '"image_url":' not in object_literal:
        object_literal = object_literal[:-1] + f', "image_url": "{image_url}"' + object_literal[-1:]
    html = html[:start] + object_literal + html[end:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Embedded viewer written to: {output_path}")


if __name__ == "__main__":
    main()
