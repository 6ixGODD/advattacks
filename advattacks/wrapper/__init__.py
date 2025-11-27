from __future__ import annotations

import torch


class Wrapper:
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
