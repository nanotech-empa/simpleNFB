"""Offset Traces: add i*offset to the i-th trace (was DAT's offsetToggle, waterfall stagger)."""

import numpy as np

NAME = 'Offset Traces'
KIND = 'batch'
DESCRIPTION = 'Stagger traces vertically: y_i = y_i + i * offset.'
PARAMS = [
    {'name': 'offset', 'label': 'amt', 'type': 'float', 'default': 0.1e-12,
     'tooltip': 'vertical offset applied per trace index'},
]


def process(xs, ys, *, offset=0.1e-12):
    out = [np.asarray(y, dtype=float) + i * offset for i, y in enumerate(ys)]
    return xs, out
