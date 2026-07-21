"""Uniform (boxcar) moving average via convolution.

Moved verbatim from process_utils.moving_average; edge samples are
normalised by the number of in-range samples (no zero-padding bias).
"""

import numpy as np

NAME = 'Moving Average'
KIND = '1d'
DESCRIPTION = 'Boxcar moving average (convolution, edge-normalised).'
PARAMS = [
    {'name': 'window_size', 'label': 'w', 'type': 'int', 'default': 5, 'min': 1, 'max': 101, 'step': 2,
     'tooltip': 'number of points averaged per sample (odd for symmetry)'},
]


def process(y, *, window_size=5):
    y = np.asarray(y, dtype=float)
    window_size = max(1, int(window_size))
    kernel = np.ones(window_size)
    smoothed = np.convolve(y, kernel, mode='same')
    counts = np.convolve(np.ones_like(y), kernel, mode='same')
    return smoothed / counts
