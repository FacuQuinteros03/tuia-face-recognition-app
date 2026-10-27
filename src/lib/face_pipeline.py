"""Pipeline de detección, alineación y preprocesamiento compartido entre las notebooks y la app."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.model_zoo import get_model
from insightface.utils import face_align


@dataclass
class Detection:
    box: np.ndarray  # (4,) x1, y1, x2, y2 en píxeles de la imagen original
    score: float
    kps: np.ndarray | None  # (5, 2) absolutos: ojo izq, ojo der, nariz, comisura izq, comisura der


def load_detector(
    path: str | Path,
    input_size: int = 640,
    score_threshold: float = 0.5,
    providers: list[str] | None = None,
):
    """SCRFD de InsightFace (det_10g.onnx de buffalo_l). Por defecto en CPU, como en la app."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Detector path does not exist: {path}")
    with contextlib.redirect_stdout(io.StringIO()):  # get_model imprime la config de onnxruntime
        detector = get_model(str(path), providers=providers or ["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, input_size=(input_size, input_size), det_thresh=score_threshold)
    return detector


def detect(detector, image_bgr: np.ndarray, min_face_size: int = 0, pad: float = 0.0) -> list[Detection]:
    """Detecta rostros. `pad` agrega un borde negro de pad * lado mayor antes de detectar.

    SCRFD no detecta caras más grandes que ~512 px en la entrada redimensionada: en un primer plano (la cara ocupa toda
    la imagen) falla. Con el borde, la cara queda de un tamaño que sí detecta. Cajas y landmarks se devuelven en
    coordenadas de la imagen original.
    """
    p = int(round(pad * max(image_bgr.shape[:2])))
    padded = cv2.copyMakeBorder(image_bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=0) if p else image_bgr
    bboxes, kpss = detector.detect(padded, max_num=0)
    h, w = image_bgr.shape[:2]
    detections = []
    for i, (x1, y1, x2, y2, score) in enumerate(bboxes):
        box = np.array([x1 - p, y1 - p, x2 - p, y2 - p], dtype=np.float32)
        box[[0, 2]] = box[[0, 2]].clip(0, w)
        box[[1, 3]] = box[[1, 3]].clip(0, h)
        if min(box[2] - box[0], box[3] - box[1]) < min_face_size:
            continue
        kps = kpss[i] - p if kpss is not None else None
        detections.append(Detection(box, float(score), kps))
    return detections


def align(image_bgr: np.ndarray, kps: np.ndarray, face_size: int = 112) -> np.ndarray:
    """Transformación de similitud de los 5 landmarks a la plantilla ArcFace. Devuelve BGR uint8."""
    return face_align.norm_crop(image_bgr, landmark=kps, image_size=face_size)


def to_model_input(faces_bgr: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """BGR uint8 (H, W, 3) o lista -> float32 NCHW en RGB escalado a [-1, 1].

    Es la misma normalización que usa ArcFace, y la usamos también para la CNN custom.
    """
    if isinstance(faces_bgr, np.ndarray) and faces_bgr.ndim == 3:
        faces_bgr = [faces_bgr]
    return cv2.dnn.blobFromImages(
        list(faces_bgr), scalefactor=1.0 / 127.5, mean=(127.5, 127.5, 127.5), swapRB=True
    )


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)
