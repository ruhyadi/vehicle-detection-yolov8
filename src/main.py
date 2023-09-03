"""
Main function to run API.
usage:
python src/main.py \
    --engine_path tmp/vehicle_kitti_v0_last.onnx \
    --categories car pedestrian cyclist \
    --provider cpu
"""

import rootutils

ROOT = rootutils.autosetup()

import argparse

from fastapi import FastAPI

from src.api.snapshot_api import SnapshotApi, UvicornServer

if __name__ == "__main__":
    """Main function to run API."""
    parser = argparse.ArgumentParser(description="Run API server")
    parser.add_argument(
        "--engine_path",
        type=str,
        required=True,
        help="Path to ONNX runtime engine file",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["car", "pedestrian", "cyclist"],
        help="Categories to detect",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Provider for ONNX runtime engine",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run API server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7000,
        help="Port to run API server",
    )
    args = parser.parse_args()

    app = FastAPI(
        title="Vehicle Detection API",
        description="API for vehicle detection",
        version="1.0.0",
        docs_url="/",
    )

    snapshot_api = SnapshotApi(
        engine_path=args.engine_path,
        categories=args.categories,
        provider=args.provider,
    )
    app.include_router(snapshot_api.router)

    server = UvicornServer(app, host=args.host, port=args.port)
    server.run()
