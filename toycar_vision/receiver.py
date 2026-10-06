from __future__ import annotations

import argparse
import socket


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Toy Car Vision UDP packets")
    parser.add_argument("--host", default="0.0.0.0", help="local interface to bind")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"Listening on {args.host}:{args.port}; Ctrl+C to stop")
    try:
        while True:
            payload, address = sock.recvfrom(65535)
            print(f"{address[0]}:{address[1]} {payload.decode('utf-8').rstrip()}")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()

