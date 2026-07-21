"""Fix Zero (2D): rescale so the image minimum is zero (was SXM's fixZeroBtn)."""

import numpy as np

NAME = 'Fix Zero (2D)'
KIND = '2d'
DESCRIPTION = 'Subtract the image minimum (ignoring NaNs) so it sits at zero.'
PARAMS = []


def process(img):
    return img - np.nanmin(img)
