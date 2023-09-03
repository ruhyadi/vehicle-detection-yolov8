"""YOLO engine schema."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

from pydantic import BaseModel, Field, validator


class YoloResultSchema(BaseModel):
    """YOLO engine detection result schema."""

    boxes: List[List[int]] = Field([], example=[[0, 0, 100, 100], [50, 50, 150, 150]])
    scores: List[float] = Field([], example=[0.9, 0.8])
    categories: List[str] = Field([], example=["person", "car"])

    @validator("scores", pre=True)
    def scores_validator(cls, v):
        """Round scores to 2 decimal places."""
        return [round(x, 2) for x in v]

    @validator("boxes", pre=True)
    def clip_boxes(cls, v):
        """Clip boxes to image size."""
        # clip minimum to 1 avoid minus
        return [[max(x, 1) for x in box] for box in v]
