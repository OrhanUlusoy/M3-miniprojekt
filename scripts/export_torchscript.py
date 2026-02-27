from __future__ import annotations

"""Exporterar en demo-modell till TorchScript.

Ersätt innehållet med export av er K2-modell.
Output: artifacts/model.ts
"""

from pathlib import Path

import torch
from torch import nn


def main() -> None:
    artifact_path = Path(__file__).resolve().parent.parent / "artifacts" / "model.ts"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    class ToyRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = ToyRegressor()

    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32))
        model.linear.bias.copy_(torch.tensor([0.0], dtype=torch.float32))

    scripted = torch.jit.script(model)
    scripted.save(str(artifact_path))
    print(f"Wrote: {artifact_path}")


if __name__ == "__main__":
    main()
