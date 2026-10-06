from __future__ import annotations

import argparse

from .config import load_config
from .server import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track toy cars and publish UDP telemetry")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration file")
    parser.add_argument("--host", help="override UDP destination host")
    parser.add_argument("--port", type=int, help="override UDP destination port")
    parser.add_argument("--headless", action="store_true", help="disable preview window")
    parser.add_argument("--metrics-csv", help="write per-frame timing/detection data")
    parser.add_argument("--max-frames", type=int, help="stop after N frames (useful for tests)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run(
        config,
        host_override=args.host,
        port_override=args.port,
        headless=args.headless,
        metrics_csv=args.metrics_csv,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

