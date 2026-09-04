"""Predictor for the Zhou et al. L3 -10/-35 promoter library."""

from .model import L3ConditionalCNN, L3PositionAwareCNN

__all__ = ["L3ConditionalCNN", "L3PositionAwareCNN"]
