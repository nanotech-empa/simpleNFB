"""Invert: flip the sign of the image data (was SXM's invertBtn)."""

NAME = 'Invert'
KIND = '2d'
DESCRIPTION = 'Multiply image data by -1.'
PARAMS = []


def process(img):
    return img * -1
