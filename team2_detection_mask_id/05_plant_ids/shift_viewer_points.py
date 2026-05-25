from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a new viewer HTML with shifted plant point rendering."
    )
    parser.add_argument("--viewer", required=True, help="Path to the source viewer HTML.")
    parser.add_argument("--output", required=True, help="Path to the shifted viewer HTML.")
    parser.add_argument("--shift-x", type=float, default=6.0, help="Horizontal point shift in screen pixels.")
    parser.add_argument("--shift-y", type=float, default=0.0, help="Vertical point shift in screen pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer_path = Path(args.viewer).resolve()
    output_path = Path(args.output).resolve()

    if not viewer_path.exists():
        raise FileNotFoundError(f"Viewer not found: {viewer_path}")

    html = viewer_path.read_text(encoding="utf-8")
    translate_block = (
        '          "circle-translate": '
        f'[{args.shift_x}, {args.shift_y}],\n'
        '          "circle-translate-anchor": "viewport",\n'
    )

    plants_marker = '          "circle-stroke-color": "#101010",\n'
    if plants_marker not in html:
        raise RuntimeError("Could not find plants circle paint block.")
    html = html.replace(plants_marker, translate_block + plants_marker, 1)

    selected_marker = '          "circle-radius": 9,\n'
    if selected_marker not in html:
        raise RuntimeError("Could not find selected plant ring paint block.")
    html = html.replace(
        selected_marker,
        selected_marker
        + '          "circle-translate": '
        + f'[{args.shift_x}, {args.shift_y}],\n'
        + '          "circle-translate-anchor": "viewport",\n',
        1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Shifted viewer written to: {output_path}")


if __name__ == "__main__":
    main()
