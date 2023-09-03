"""ONNX engine schema."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

from pydantic import BaseModel, Field, validator


class OnnxMetadataSchema(BaseModel):
    """ONNX metadata schema."""

    input_name: str = Field(..., example="images")
    input_shape: List = Field(..., example=[1, 3, 224, 224])
    output_name: str = Field(..., example="output")
    output_shape: List = Field(..., example=[1, 8400, 85])

    @validator("input_shape", "output_shape")
    def check_shape(cls, v):
        """If dynamic shape, set to -1."""
        return [-1 if isinstance(i, str) else i for i in v]
