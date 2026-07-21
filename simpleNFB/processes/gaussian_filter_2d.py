"""Gaussian Filter (2D): Gaussian blur (was SXM's gaussianToggle)."""

from scipy.ndimage import gaussian_filter

NAME = 'Gaussian Filter'
KIND = '2d'
DESCRIPTION = 'Gaussian blur (scipy.ndimage.gaussian_filter).'
PARAMS = [
    {'name': 'sigma', 'label': 'sigma', 'type': 'int', 'default': 2, 'min': 0, 'max': 10, 'step': 1,
     'tooltip': 'Gaussian kernel size'},
]


def process(img, *, sigma=2):
    return gaussian_filter(img, sigma)
