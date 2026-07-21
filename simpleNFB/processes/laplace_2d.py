"""Laplace (2D): edge detection (was SXM's laplacToggle)."""

from scipy.ndimage import gaussian_laplace

NAME = 'Laplace (edge detection)'
KIND = '2d'
DESCRIPTION = 'Gaussian-Laplace edge detection, sign-flipped for display.'
PARAMS = [
    {'name': 'sigma', 'label': 'sigma', 'type': 'int', 'default': 1, 'min': 1, 'max': 10, 'step': 1,
     'tooltip': 'Laplace kernel size'},
]


def process(img, *, sigma=1):
    return -gaussian_laplace(img, sigma)
