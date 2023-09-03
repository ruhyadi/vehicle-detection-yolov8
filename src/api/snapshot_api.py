"""Snapshot API module."""

import rootutils

ROOT = rootutils.autosetup()

from fastapi import APIRouter, Depends, FastAPI
import uvicorn

from src.schema.api_schema import SnapshotRequestSchema, SnapshotResponseSchema

class SnapshotApi:
    """Snapshot API module."""

    def __init__(self) -> None:
        """Initialize Snapshot API module."""
        self.app = FastAPI()
        self.router = APIRouter()

        self.setup()

    def setup_engine(self)

    def setup(self) -> None:
        """Setup API router."""

        @self.router.post(
            "/api/v1/detection/snapshot",
            tags=["detection"],
            summary="Detect objects in a snapshot",
            response_model=SnapshotResponseSchema,
            )
        async def snapshot(
            request: SnapshotRequestSchema = Depends(SnapshotRequestSchema.as_form),
        ):
            """Detect objects in a snapshot."""
            return {
                "category": "car",
                "box": [0, 0, 100, 100],
                "score": 0.69,
            }


class UvicornServer:
    """Uvicorn runner."""

    def __init__(
        self, app, host: str, port: int, workers: int = 1, log_level: str = "info"
    ):
        self.app = app
        self.host = host
        self.port = port
        self.workers = workers
        self.log_level = log_level

    def run(self):
        print(f"Starting uvicorn server on {self.host}:{self.port}...")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            workers=self.workers,
            log_level=self.log_level,
        )
