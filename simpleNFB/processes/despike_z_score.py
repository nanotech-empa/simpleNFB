"""Despike (Z-score): modified Z-score spike removal (Iglewicz-Hoaglin).

Moved verbatim from process_utils.despike_z_score (v0.2.0); see that module's
shim for the back-compat wrapper.
"""

import numpy as np

NAME = 'Despike (Z-score)'
KIND = '1d'
DESCRIPTION = 'Modified Z-score spike removal (Iglewicz-Hoaglin).'
PARAMS = [
    {'name': 'window_size', 'label': 'w', 'type': 'int', 'default': 10, 'min': 2, 'max': 100, 'step': 1,
     'tooltip': 'half-width of the local window (full window = 2*window_size)'},
    {'name': 'threshold', 'label': 't', 'type': 'float', 'default': 3.0, 'step': 0.5,
     'tooltip': 'modified Z-score cutoff above which a point is a spike'},
]


def process(y, *, window_size=10, threshold=3.0):
    """For each sample, compare it to the local median/MAD of a symmetric
    window; replace it with the local median if its modified Z-score
    (0.6745*|x-median|/MAD) exceeds threshold."""
    y = np.asarray(y, dtype=float)
    out = np.copy(y)
    window_size = int(window_size)
    for i in range(window_size, len(y) - window_size):
        window = y[i - window_size: i + window_size]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        modified_z = 0.0 if mad == 0 else 0.6745 * abs(y[i] - median) / mad
        if modified_z > threshold:
            out[i] = median
    return out
