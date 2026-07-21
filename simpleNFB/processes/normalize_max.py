"""Normalize to max: divide by the trace's own maximum (was DAT's flattenBtn)."""

import numpy as np

NAME = 'Normalize (Max)'
KIND = '1d'
DESCRIPTION = 'Divide by the maximum value of the trace (y = y / max(y)).'
PARAMS = []


def process(y):
    y = np.asarray(y, dtype=float)
    return y / np.max(y)
