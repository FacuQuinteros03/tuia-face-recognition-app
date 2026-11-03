"""CNN custom (Modelo A): ResNet chica que mapea un rostro alineado 112x112 a un embedding L2-normalizado."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.act = nn.PReLU(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and c_in == c_out
            else nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class FaceCNN(nn.Module):
    """112x112x3 -> embedding de `embedding_dim`.

    pooling="gap": promedio global (pocas features, invariante a la posición).
    pooling="gdc": convolución depthwise 7x7 (como MobileFaceNet): aprende un peso por posición, útil porque las
    caras están alineadas y cada zona (ojos, nariz, boca) cae siempre en el mismo lugar.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        blocks: tuple[int, ...] = (2, 2, 2, 2),
        pooling: str = "gdc",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.PReLU(32)
        )  # 56x56
        layers, c_in = [], 32
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(BasicBlock(c_in, w, stride=2 if j == 0 and i > 0 else 1))
                c_in = w
        self.body = nn.Sequential(*layers)  # 56 -> 28 -> 14 -> 7
        if pooling == "gdc":
            self.pool = nn.Sequential(nn.Conv2d(c_in, c_in, 7, groups=c_in, bias=False), nn.BatchNorm2d(c_in))
        elif pooling == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(pooling)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, embedding_dim, bias=False), nn.BatchNorm1d(embedding_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """Devuelve el embedding SIN normalizar (la pérdida decide cómo usarlo)."""
        return self.head(self.pool(self.body(self.stem(x))))


class SoftmaxHead(nn.Module):
    """Clasificador lineal común: el embedding es la capa previa."""

    def __init__(self, embedding_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, n_classes)

    def forward(self, emb, labels=None):
        return self.fc(emb)


class MarginHead(nn.Module):
    """Logits con margen angular sobre embedding y pesos normalizados.

    CosFace: s * (cos(θ) - m) para la clase correcta. ArcFace: s * cos(θ + m).
    Obliga a que cada identidad ocupe una región angular compacta, que es lo que usa la app (similitud coseno).
    """

    def __init__(self, embedding_dim: int, n_classes: int, kind: str = "cosface", s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.kind, self.s, self.m = kind, s, m

    def forward(self, emb, labels=None):
        cos = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        if labels is None:
            return self.s * cos
        onehot = F.one_hot(labels, cos.size(1)).bool()
        if self.kind == "cosface":
            target = cos - self.m
        else:
            theta = torch.acos(cos)
            target = torch.where(theta + self.m < math.pi, torch.cos(theta + self.m), cos - self.m * math.sin(self.m))
        return self.s * torch.where(onehot, target, cos)


class EmbeddingExport(nn.Module):
    """Envoltorio para exportar a ONNX: devuelve el embedding ya normalizado L2."""

    def __init__(self, backbone: FaceCNN):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return F.normalize(self.backbone(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
