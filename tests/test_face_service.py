import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lib import face_pipeline as fp
from lib.schemas import EmbeddingRecord
from lib.services.face_service import FaceService

# Landmarks de la plantilla ArcFace (112x112), desplazados para ubicar una cara de ~100 px en la imagen.
TEMPLATE = np.array(
    [[38.29, 51.69], [73.53, 51.50], [56.02, 71.73], [41.55, 92.37], [70.73, 92.20]], dtype=np.float32
)


class FakeDetector:
    """Imita a SCRFD: recibe la imagen (ya con borde) y devuelve caras en posiciones fijas de la imagen ORIGINAL."""

    def __init__(self, faces):
        self.faces = faces  # lista de (x1, y1) de la esquina de cada cara en la imagen original
        self.original_shape = None

    def detect(self, img, max_num=0):
        oh, ow = self.original_shape
        p = (img.shape[0] - oh) // 2  # borde agregado por fp.detect
        bboxes, kpss = [], []
        for x1, y1 in self.faces:
            # Si la imagen es un recorte, la cara se busca relativa al recorte via `offset`
            ox, oy = self.offset
            x, y = x1 - ox + p, y1 - oy + p
            bboxes.append([x, y, x + 100, y + 110, 0.9])
            kpss.append(TEMPLATE * (100 / 112) + [x, y])
        if not bboxes:
            return np.zeros((0, 5), np.float32), None
        return np.array(bboxes, np.float32), np.array(kpss, np.float32)


class FakeStore:
    def __init__(self):
        self.records = []

    def all(self):
        return self.records

    def append(self, record):
        self.records.append(record)


class FakeModel:
    def get_inputs(self):
        return [type("I", (), {"name": "input"})()]

    def run(self, _, feeds):
        x = feeds["input"]
        return [np.tile(x.mean(axis=(2, 3)), (1, 171))[:, :512] + 1.0]


@pytest.fixture
def image():
    return np.full((400, 300, 3), 128, np.uint8)


def make_service(faces, image_shape, monkeypatch):
    det = FakeDetector(faces)
    monkeypatch.setattr(FaceService, "_load_model", lambda self, p: FakeModel())
    monkeypatch.setattr(FaceService, "_load_detector", lambda self, p: det)
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeModel)
    svc = FaceService(FakeStore(), "cosine", 0.5, 112, Path("unused.onnx"), output_path=Path("/tmp/out"))

    original_detect = fp.detect

    def detect_with_shape(detector, img, min_face_size=0, pad=0.0):
        det.original_shape = img.shape[:2]
        det.offset = getattr(det, "next_offset", (0, 0))
        det.next_offset = (0, 0)
        return original_detect(detector, img, min_face_size, pad)

    monkeypatch.setattr(fp, "detect", detect_with_shape)
    return svc, det


def test_to_model_input_matches_training_normalization():
    face = np.zeros((112, 112, 3), np.uint8)
    face[..., 0] = 255  # azul en BGR
    x = fp.to_model_input(face)
    assert x.shape == (1, 3, 112, 112)
    assert x.dtype == np.float32
    # BGR -> RGB: el azul queda en el último canal, escalado a [-1, 1]
    assert np.allclose(x[0, 2], 1.0) and np.allclose(x[0, 0], -1.0)


def test_l2_normalize():
    v = fp.l2_normalize(np.array([[3.0, 4.0]]))
    assert np.allclose(v, [[0.6, 0.8]])


def test_detect_with_pad_returns_original_coordinates(image):
    det = FakeDetector([(50, 60)])
    det.original_shape, det.offset = image.shape[:2], (0, 0)
    (d,) = fp.detect(det, image, pad=0.5)
    assert np.allclose(d.box, [50, 60, 150, 170])
    assert np.allclose(d.kps[0], TEMPLATE[0] * (100 / 112) + [50, 60])


def test_clip_xyxy_keeps_box_inside_image():
    assert FaceService._clip_xyxy(-10, -5, 500, 900, height=400, width=300) == (0, 0, 300, 400)


def test_detect_faces_returns_int_boxes_and_empty_list(image, monkeypatch):
    svc, det = make_service([(50, 60)], image.shape, monkeypatch)
    boxes = svc.detect_faces(image)
    assert boxes == [(50, 60, 150, 170)]
    assert all(isinstance(v, int) for v in boxes[0])

    det.faces = []
    assert svc.detect_faces(image) == []


def test_align_face_keypoints_are_relative_to_box(image, monkeypatch):
    svc, det = make_service([(50, 60)], image.shape, monkeypatch)
    box = (50, 60, 150, 170)
    mx, my = int(100 * svc.align_margin), int(110 * svc.align_margin)
    det.next_offset = (max(0, 50 - mx), max(0, 60 - my))  # align_face detecta sobre un recorte con margen
    face = svc.align_face(image, box)
    assert face.image.shape == (112, 112, 3) and face.image.dtype == np.uint8
    assert face.keypoints.shape == (5, 2)
    assert np.allclose(face.keypoints, TEMPLATE * (100 / 112), atol=1e-3)


def test_embedding_has_expected_dim_and_unit_norm(image, monkeypatch):
    svc, det = make_service([(50, 60)], image.shape, monkeypatch)
    det.next_offset = (0, 5)
    emb = svc.extract_embedding_from_face(svc.align_face(image, (50, 60, 150, 170)))
    assert isinstance(emb, list) and len(emb) == 512
    assert all(type(v) is float for v in emb)
    assert np.isclose(np.linalg.norm(emb), 1.0)


def test_register_with_two_faces_fails(image, monkeypatch, tmp_path):
    svc, det = make_service([(10, 10), (160, 200)], image.shape, monkeypatch)
    path = tmp_path / "two.jpg"
    import cv2

    cv2.imwrite(str(path), image)
    with pytest.raises(ValueError, match="Exactly one face"):
        svc.register_identity("x", str(path), {})


def test_identify_returns_unknown_below_threshold(monkeypatch, image):
    svc, _ = make_service([], image.shape, monkeypatch)
    svc.store.append(EmbeddingRecord(id_imagen="1", embedding=[1.0, 0.0], path="p", etiqueta="ana"))
    assert svc.identify([1.0, 0.0])[0] == "ana"
    assert svc.identify([0.0, 1.0])[0] == "unknown"
