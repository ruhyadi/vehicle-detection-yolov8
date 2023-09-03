"""Snapshot API module."""

import rootutils

ROOT = rootutils.autosetup()

from io import BytesIO
from typing import List

import cv2
import numpy as np
import uvicorn
from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import Response
from PIL import Image

from src.engine.yolo_onnx_engine import YoloOnnxEngine
from src.schema.api_schema import SnapshotRequestSchema, SnapshotResponseSchema


class SnapshotApi:
    """Snapshot API module."""

    def __init__(
        self,
        engine_path: str,
        categories: List[str] = ["car", "pedestrian", "cyclist"],
        provider: str = "cpu",
    ) -> None:
        """Initialize Snapshot API module."""
        self.engine_path = engine_path
        self.categories = categories
        self.provider = provider

        # setup api router
        self.app = FastAPI()
        self.router = APIRouter()

        self.setup_engine()
        self.setup()

    def setup_engine(self) -> None:
        """Setup YOLOv8 ONNX engine."""
        self.engine = YoloOnnxEngine(
            engine_path=self.engine_path,
            categories=self.categories,
            provider=self.provider,
        )
        self.engine.setup()

    def setup(self) -> None:
        """Setup API router."""

        @self.router.post(
            "/api/v1/detection/snapshot",
            tags=["detection"],
            summary="Detect objects in a snapshot",
            response_model=List[SnapshotResponseSchema],
        )
        async def snapshot(
            form: SnapshotRequestSchema = Depends(),
        ):
            """Detect objects in a snapshot."""
            print(f"Request snapshot...")

            # preprocess image
            img = await self.preprocess_img_bytes(await form.image.read())

            # detect
            dets = self.engine.detect(img, conf=form.conf)[0]

            # convert to result schema
            result: List[SnapshotResponseSchema] = []
            for box, score, cat in zip(dets.boxes, dets.scores, dets.categories):
                result.append(
                    SnapshotResponseSchema(
                        category=cat,
                        box=box,
                        score=score,
                    )
                )

            return result

        @self.router.post(
            "/api/v2/detection/snapshot",
            tags=["detection"],
            summary="Detect objects in a snapshot",
        )
        async def snapshot_v2(
            form: SnapshotRequestSchema = Depends(),
        ):
            """Detect objects in a snapshot."""
            print(f"Request snapshot...")

            # preprocess image
            img = await self.preprocess_img_bytes(await form.image.read())

            # detect
            dets = self.engine.detect(img, conf=form.conf)[0]

            # convert to result schema
            result: List[SnapshotResponseSchema] = []
            for box, score, cat in zip(dets.boxes, dets.scores, dets.categories):
                result.append(
                    SnapshotResponseSchema(
                        category=cat,
                        box=box,
                        score=score,
                    )
                )

            # draw detection result
            for res in result:
                cv2.rectangle(
                    img,
                    (res.box[0], res.box[1]),
                    (res.box[2], res.box[3]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    f"{res.category} {res.score:.2f}",
                    (res.box[0], res.box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # return image
            _, img_bytes = cv2.imencode(".jpg", img)

            return Response(content=img_bytes.tobytes(), media_type="image/jpeg")

    async def preprocess_img_bytes(self, img_bytes: bytes) -> np.ndarray:
        """Convert image bytes to numpy array."""
        img = Image.open(BytesIO(img_bytes))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if PNG convert to RGB
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        return img


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
