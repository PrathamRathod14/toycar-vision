"""Generate a chessboard image for printing and measuring."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=int, default=9, help="inner corners across")
    parser.add_argument("--rows", type=int, default=6, help="inner corners down")
    parser.add_argument("--square-pixels", type=int, default=100)
    parser.add_argument("--border-pixels", type=int, default=50)
    parser.add_argument("--output", default="chessboard.png")
    args = parser.parse_args()
    if min(args.columns, args.rows, args.square_pixels) < 2 or args.border_pixels < 0:
        parser.error("Invalid chessboard dimensions")
    width = (args.columns + 1) * args.square_pixels + 2 * args.border_pixels
    height = (args.rows + 1) * args.square_pixels + 2 * args.border_pixels
    image = np.full((height, width), 255, dtype=np.uint8)
    for row in range(args.rows + 1):
        for column in range(args.columns + 1):
            if (row + column) % 2 == 0:
                x = args.border_pixels + column * args.square_pixels
                y = args.border_pixels + row * args.square_pixels
                image[y:y + args.square_pixels, x:x + args.square_pixels] = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise SystemExit(f"Could not write {output}")
    print(f"Saved {output}. Print flat and measure one actual square in millimetres.")


if __name__ == "__main__":
    main()
