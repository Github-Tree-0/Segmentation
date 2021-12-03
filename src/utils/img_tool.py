from scipy import ndimage
import numpy as np

def imfill(img, ignore_size=None):
    # Conduct binary hole filling
    if ignore_size != None:
        return ndimage.binary_fill_holes(img, np.ones((ignore_size, ignore_size)))
    return ndimage.binary_fill_holes(img)