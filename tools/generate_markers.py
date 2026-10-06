from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate printable ArUco tag PNG files")
    parser.add_argument("ids", nargs="+", type=int)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--pixels", type=int, default=600)
    parser.add_argument("--output-dir", default="markers")
    args = parser.parse_args()
    if not hasattr(cv2.aruco, args.dictionary):
        raise SystemExit(f"Unknown dictionary: {args.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for marker_id in args.ids:
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, args.pixels)
        path = output / f"aruco_{marker_id}.png"
        if not cv2.imwrite(str(path), image):
            raise SystemExit(f"Could not write {path}")
        print(path)


if __name__ == "__main__":
    main()

