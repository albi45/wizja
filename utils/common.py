import cv2
import numpy as np

from consts.common import IMG_SIZE


def resize(img):
    if img.shape[0] > IMG_SIZE or img.shape[1] > IMG_SIZE:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=interpolation)
    img = np.resize(img, (1, IMG_SIZE, IMG_SIZE, 1))
    return img
