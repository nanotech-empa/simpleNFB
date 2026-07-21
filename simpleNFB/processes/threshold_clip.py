"""Threshold clip: zero out samples at/above a cutoff (was DAT's thresholdToggle)."""

import numpy as np

NAME = 'Threshold Clip'
KIND = '1d'
DESCRIPTION = 'Zero samples at or above a value cutoff (y = y * (y < threshold)).'
PARAMS = [
    {'name': 'threshold', 'label': 'val', 'type': 'float', 'default': 100.0,
     'tooltip': 'samples >= this value are zeroed'},
]


def process(y, *, threshold=100.0):
    y = np.asarray(y, dtype=float)
    return y * (y < threshold)
