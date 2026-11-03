"""Data augmentation para rostros ya alineados (112x112, BGR uint8). Se aplica solo en train."""

from __future__ import annotations

import albumentations as A
import cv2


def build_train_augmentation(face_size: int = 112) -> A.Compose:
    return A.Compose(
        [
            # La cara es aproximadamente simétrica: el espejo conserva la identidad.
            A.HorizontalFlip(p=0.5),
            # Errores chicos de alineación (los landmarks nunca son perfectos).
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.04, 0.04),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            # Iluminación y balance de color distintos según la foto.
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=0, p=0.3),
            A.ToGray(p=0.05),
            # Fotos fuera de foco, movidas o con ruido de sensor.
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ],
                p=0.2,
            ),
            A.GaussNoise(std_range=(0.02, 0.06), p=0.15),
            # Compresión de imágenes de redes sociales.
            A.ImageCompression(quality_range=(40, 95), p=0.3),
            # Oclusiones parciales: anteojos, mano, pelo, barbijo.
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(0.1, 0.25),
                hole_width_range=(0.1, 0.25),
                fill=0,
                p=0.3,
            ),
            A.Resize(face_size, face_size),
        ]
    )
