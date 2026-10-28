"""Optional COCO vehicle detector. Missing inputs fail explicitly.

Set UBIGRAPH_DETECTION_WEIGHTS to a local Ultralytics COCO checkpoint.
Synthetic counts require payload.demo=true and are never measurements.
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import Agent, Envelope


class DetectionAgent(Agent):
    name = "detection"

    def __init__(self, bus=None, weights=None) -> None:
        if weights is None:
            weights = os.environ.get("UBIGRAPH_DETECTION_WEIGHTS")
        self._load_error = None
        self.weights = str(weights) if weights else None
        self._model = None
        self._backend = "unavailable"
        super().__init__(bus)

    def _register(self) -> None:
        self.handlers = {
            "detect": self._detect,
            "status": self._status,
        }

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.weights or not Path(self.weights).exists():
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
            self._backend = f"ultralytics:{Path(self.weights).name}"
        except Exception as exc:
            self._load_error = repr(exc)
            self._model = None
            self._backend = "unavailable"

    def _status(self, env: Envelope) -> dict:
        self._ensure_model()
        return {"backend": self._backend, "weights": self.weights,
                "available": self._model is not None, "load_error": self._load_error,
                "note": "请配置 UBIGRAPH_DETECTION_WEIGHTS；桩值仅允许显式 demo=true"}

    def _detect(self, env: Envelope) -> dict:
        image = env.payload.get("image")
        self._ensure_model()
        if self._model is not None and image:
            try:
                res = self._model.predict(image, verbose=False)
                boxes = res[0].boxes
                cars = sum(int(cls) == 2 for cls in boxes.cls.tolist()) if boxes is not None else 0
                return {"ok": True, "demo": False, "backend": self._backend, "image": str(image),
                        "cars_detected": cars,
                        "segmented": res[0].masks is not None}
            except Exception as exc:
                return {"ok": False, "backend": self._backend, "cars_detected": None,
                        "error": f"real inference failed: {exc!r}"}
        if env.payload.get("demo") is not True:
            return {"ok": False, "backend": "unavailable", "cars_detected": None,
                    "error": self._load_error or "需要图像和有效检测权重；未执行真实检测"}
        # Explicit demonstration only; never a measurement.
        import hashlib
        h = int(hashlib.md5(str(image).encode()).hexdigest(), 16)
        return {"ok": True, "demo": True, "backend": "stub", "image": str(image),
                "cars_detected": h % 24 + 4, "segmented": False,
                "note": "演示桩值，禁止写入真实观测或用于效果评估"}
