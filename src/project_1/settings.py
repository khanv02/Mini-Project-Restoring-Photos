"""Shared algorithm names and defaults; no GUI dependencies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    label: str
    needs_mask: bool = False
    uses_gaussian: bool = False


ALGORITHMS = (
    AlgorithmSpec("gaussian", "Gaussian Filter", uses_gaussian=True),
    AlgorithmSpec("inpainting", "Inpainting", needs_mask=True),
    AlgorithmSpec("combined", "Combined (Inpainting + Gaussian)", True, True),
)
DEFAULT_KERNEL_SIZE = 5
DEFAULT_SIGMA = 1.2
DEFAULT_RADIUS = 3
DEFAULT_METHOD = "telea"
DEFAULT_SHARPEN_AMOUNT = 0.5
DEFAULT_SHARPEN_SIGMA = 1.0
