"""YOLO ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Union

import cv2
import numpy as np

from src.engine.onnx_engine import CommonOnnxEngine
from src.schema.yolo_schema import YoloResultSchema
from src.utils.nms_utils import multiclass_nms


class YoloOnnxEngine(CommonOnnxEngine):
    """Yolo ONNX engine module."""

    def __init__(
        self,
        engine_path: str,
        categories: List[str] = ["car", "pedestrian", "cyclist"],
        provider: str = "cpu",
    ) -> None:
        """Initialize YOLO ONNX engine."""
        super().__init__(engine_path, provider)
        self.categories = categories

    def detect(
        self,
        imgs: Union[np.ndarray, List[np.ndarray]],
        conf: float = 0.25,
        nms: float = 0.45,
    ) -> List[YoloResultSchema]:
        """Detect objects in image(s)."""
        imgs, ratios, pads = self.preprocess_imgs(imgs)
        outputs = self.engine.run(None, {self.metadata[0].input_name: imgs})
        outputs = self.postprocess_nms(outputs, ratios, pads, conf, nms=nms)

        return outputs

    def preprocess_imgs(
        self,
        imgs: Union[np.ndarray, List[np.ndarray]],
        mode="center",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Preprocess image(s) (batch) like resize, normalize, padding, etc.

        Args:
            imgs (Union[np.ndarray, List[np.ndarray]]): Image(s) to preprocess.
            mode (str, optional): Padding mode. Defaults to "center".
            normalize (bool, optional): Whether to normalize image(s). Defaults to True.

        Returns:
            np.ndarray: Preprocessed image(s) in size (B, C, H, W).
        """
        assert mode in ["center", "left"], "Invalid mode, must be 'center' or 'left'"
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        # resize and pad
        dst_h, dst_w = [640, 640]
        resized_imgs = np.ones((len(imgs), dst_h, dst_w, 3), dtype=np.float32) * 114
        ratios = np.ones((len(imgs)), dtype=np.float32)
        pads = np.ones((len(imgs), 2), dtype=np.float32)
        for i, img in enumerate(imgs):
            src_h, src_w = img.shape[:2]
            ratio = min(dst_w / src_w, dst_h / src_h)
            resized_w, resized_h = int(src_w * ratio), int(src_h * ratio)
            dw, dh = (dst_w - resized_w) / 2, (dst_h - resized_h) / 2
            img = cv2.resize(img, (resized_w, resized_h))
            if mode == "center":
                top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                img = cv2.copyMakeBorder(
                    img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114
                )
                resized_imgs[i] = img
            elif mode == "left":
                resized_imgs[i][:resized_h, :resized_w, :] = img

            pads[i] = np.array([dw, dh], dtype=np.float32)
            ratios[i] = ratio

        # normalize
        resized_imgs = resized_imgs.transpose(0, 3, 1, 2)
        resized_imgs /= 255.0 if normalize else 1.0
        # resized_imgs = np.ascontiguousarray(resized_imgs).astype(np.float32)

        return resized_imgs, ratios, pads

    def postprocess_nms(
        self,
        outputs: List[np.ndarray],
        ratios: np.ndarray,
        pads: np.ndarray,
        conf: float = 0.25,
        nms: float = 0.45,
    ) -> List[YoloResultSchema]:
        """Postprocess NMS ONNX engine."""
        outputs: np.ndarray = outputs[0]
        outputs = outputs.transpose(0, 2, 1) # (B, 8400, cat + 4)
        boxes = outputs[:, :, :4]
        scores = outputs[:, :, 4:]

        # convert xywh to xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2
        boxes_xyxy[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2
        boxes_xyxy[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2] / 2
        boxes_xyxy[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3] / 2

        # scaling and filtering
        boxes_xyxy[:, :, 0] -= pads[:, 0, None]
        boxes_xyxy[:, :, 1] -= pads[:, 1, None]
        boxes_xyxy[:, :, 2] -= pads[:, 0, None]
        boxes_xyxy[:, :, 3] -= pads[:, 1, None]
        boxes_xyxy /= ratios[:, None, None]

        results: List[YoloResultSchema] = []
        for i in range(outputs.shape[0]):
            dets = multiclass_nms(boxes_xyxy[i], scores[i], nms=nms, conf=conf)
            if dets is None:
                results.append(YoloResultSchema())
                continue

            # filter confidence and class
            mask_score = dets[:, -2] > conf
            mask_class = dets[:, -1] < len(self.categories)
            dets = dets[mask_score & mask_class]

            class_ids = dets[:, -1].astype(np.int32).tolist()
            categories = [self.categories[int(i)] for i in class_ids]
            results.append(
                YoloResultSchema(
                    categories=categories,
                    scores=dets[:, -2].astype(np.float32).tolist(),
                    boxes=dets[:, :-2].astype(np.int32).tolist(),
                )
            )

        return results


if __name__ == "__main__":
    """Debugging."""

    engine = YoloOnnxEngine(
        engine_path="tmp/vehicle_kitti_v0_last.onnx",
        categories=["car", "pedestrian", "cyclist"],
        provider="cpu",
    )
    engine.setup()

    img = cv2.imread("assets/000049.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = engine.detect(img, conf=0.25, nms=0.45)

    for box in results[0].boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))

    cv2.imwrite("tmp/000049.png", img)