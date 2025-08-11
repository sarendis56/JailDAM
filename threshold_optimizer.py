#!/usr/bin/env python3
"""
Shared Threshold Optimizer

Implements the threshold selection logic consistent with reference/reference_load_datasets.py:
- Build a small balanced validation split from training data only (deterministic per-class sampling)
- Analyze score distributions and set an adaptive threshold search range
- Grid-search threshold maximizing: 0.8 * balanced_accuracy + 0.2 * F1

Assumptions:
- Scores are 1-D numpy array where higher values indicate more likely "unsafe"
- Labels are 0 for benign/safe and 1 for unsafe/malicious
"""

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score


@dataclass
class ThresholdResult:
    threshold: float
    best_balanced_acc: float
    best_f1: float
    score_range: tuple


class ThresholdOptimizer:
    def __init__(self, max_samples_per_class: int = 150, random_state: int | None = None):
        self.max_samples_per_class = max_samples_per_class
        self.random_state = random_state
        self.threshold = 0.5

    def _balanced_validation_indices(self, y: np.ndarray) -> np.ndarray:
        """Return indices for a small balanced validation split from training labels y.
        Deterministic, evenly spaced sampling per class, up to max_samples_per_class per class.
        """
        y = np.asarray(y)
        indices = []
        for cls in [0, 1]:
            cls_idx = np.where(y == cls)[0]
            if cls_idx.size == 0:
                continue
            sample_size = min(self.max_samples_per_class, cls_idx.size)
            if sample_size < cls_idx.size:
                # Evenly spaced deterministic sampling
                sampled = np.linspace(0, cls_idx.size - 1, sample_size, dtype=int)
                indices.extend(cls_idx[sampled])
            else:
                indices.extend(cls_idx)
        return np.array(indices, dtype=int)

    def _adaptive_range(self, scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Compute adaptive search range based on class score distributions."""
        scores = np.asarray(scores)
        y = np.asarray(y)
        benign_scores = scores[y == 0]
        unsafe_scores = scores[y == 1]

        if benign_scores.size == 0 or unsafe_scores.size == 0:
            # Fallback to score percentiles
            lo, hi = np.percentile(scores, [5, 95])
            width = hi - lo
            return lo - 0.2 * width, hi + 0.2 * width

        b_mean, b_std = np.mean(benign_scores), np.std(benign_scores)
        u_mean, u_std = np.mean(unsafe_scores), np.std(unsafe_scores)
        separation = abs(u_mean - b_mean) / (b_std + u_std + 1e-8)

        if separation > 5.0:
            midpoint = 0.5 * (b_mean + u_mean)
            range_width = 2.0 * min(max(b_std, 1e-8), max(u_std, 1e-8))
            return midpoint - range_width, midpoint + range_width
        else:
            lo, hi = np.percentile(scores, [5, 95])
            width = hi - lo
            return lo - 0.2 * width, hi + 0.2 * width

    def fit_from_scores(self, train_scores: np.ndarray, train_labels: np.ndarray) -> ThresholdResult:
        """Fit optimal threshold using training scores/labels only.
        Returns ThresholdResult containing the chosen threshold and metrics on validation split.
        """
        train_scores = np.asarray(train_scores)
        train_labels = np.asarray(train_labels)
        val_idx = self._balanced_validation_indices(train_labels)
        if val_idx.size == 0 or np.unique(train_labels[val_idx]).size < 2:
            # Fallback
            self.threshold = 0.5
            return ThresholdResult(threshold=0.5, best_balanced_acc=0.0, best_f1=0.0, score_range=(0.0, 1.0))

        scores = train_scores[val_idx]
        labels = train_labels[val_idx]

        # Compute adaptive search range
        lo, hi = self._adaptive_range(scores, labels)
        thresholds = np.linspace(lo, hi, 200)

        best_score = -1.0
        best_threshold = 0.5
        best_bal_acc = 0.0
        best_f1 = 0.0

        for t in thresholds:
            y_pred = (scores > t).astype(int)
            try:
                bal_acc = balanced_accuracy_score(labels, y_pred)
                f1 = f1_score(labels, y_pred, zero_division=0)
                combined = 0.8 * bal_acc + 0.2 * f1
                if combined > best_score:
                    best_score = combined
                    best_threshold = t
                    best_bal_acc = bal_acc
                    best_f1 = f1
            except Exception:
                continue

        self.threshold = float(best_threshold)
        return ThresholdResult(threshold=float(best_threshold), best_balanced_acc=float(best_bal_acc), best_f1=float(best_f1), score_range=(float(lo), float(hi)))

