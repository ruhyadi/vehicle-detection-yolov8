"""API detection schema."""

import rootutils

ROOT = rootutils.autosetup()

from pydantic import BaseModel, Field, validator
from fastapi import File, UploadFile
from typing import List


class SnapshotRequestSchema(BaseModel):
    """Snapshot request schema."""

    image: UploadFile = File(..., description="Image file to process")
    conf: float = Field(0.25, example=0.25)


class SnapshotResponseSchema(BaseModel):
    """Snapshot response schema."""

    category: str = Field(..., example="car")
    box: List[int] = Field(..., example=[0, 0, 100, 100])
    score: float = Field(..., example=0.69)