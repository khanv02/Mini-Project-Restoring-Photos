"""Pixel-level scores for binary masks with explicit empty-mask conventions."""

import numpy as np

from project_1.algorithms.scratch_mask import validate_binary_mask


def binary_mask_scores(predicted: np.ndarray, expected: np.ndarray) -> dict:
    validate_binary_mask(expected, expected.shape)
    validate_binary_mask(predicted, expected.shape)
    selected, truth = predicted > 0, expected > 0
    intersection = int(np.count_nonzero(selected & truth))
    predicted_pixels, expected_pixels = int(selected.sum()), int(truth.sum())
    union = predicted_pixels + expected_pixels - intersection
    return {
        "precision": intersection / predicted_pixels if predicted_pixels else float(expected_pixels == 0),
        "recall": intersection / expected_pixels if expected_pixels else 1.0,
        "IoU": intersection / union if union else 1.0,
    }
