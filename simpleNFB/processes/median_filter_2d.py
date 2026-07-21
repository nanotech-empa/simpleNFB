"""Median Filter (2D): despeckle (was SXM's medianToggle)."""

from scipy.ndimage import median_filter

NAME = 'Median Filter (2D)'
KIND = '2d'
DESCRIPTION = '2-D median filter (scipy.ndimage.median_filter).'
PARAMS = [
    {'name': 'size', 'label': 'size', 'type': 'int', 'default': 3, 'min': 1, 'max': 20, 'step': 1,
     'tooltip': 'median kernel size'},
]


def process(img, *, size=3):
    return median_filter(img, size=size)
