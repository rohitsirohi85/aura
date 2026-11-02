from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


EXPRESSION_CLASSES = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
)
ACTION_UNITS = (1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26)


class FacialAffectNet(nn.Module):
    """Shared ConvNeXt encoder with expression, AU, and valence/arousal heads."""

    def __init__(
        self,
        pretrained: bool = False,
        pretrained_path: str | Path | None = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained and pretrained_path is None else None
        encoder = convnext_tiny(weights=weights)
        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state)
        self.features = encoder.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.expression_head = nn.Linear(512, len(EXPRESSION_CLASSES))
        self.au_head = nn.Linear(512, len(ACTION_UNITS))
        self.va_head = nn.Linear(512, 2)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.pool(self.features(images))
        shared = self.shared(features)
        return {
            "expression": self.expression_head(shared),
            "action_units": self.au_head(shared),
            "valence_arousal": self.va_head(shared),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "FacialAffectNet":
        model = cls(pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        return model
