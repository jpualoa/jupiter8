import numpy as np

def center_crop(input_img, out_width, out_height):
    """Return a center cropped copy of an image array

    Args:
        input_img:  numpy array
        out_width:  output x dimension
        out_height: output y dimension
    """
    img = np.copy(input_img)
    x, y = img.shape[:2]
    if (out_width > x) or (out_height > y):
        raise RuntimeError("Output shape must be smaller than img shape")
    dx = x - out_width
    dy = y - out_height
    cropped = img[dx/2: x - dx/2, dy/2: y - dy/2, :]
    return cropped

