import numpy as np
from PIL import Image


def load_image(fname):
    # TODO docstring
    return np.array(Image.open(fname))