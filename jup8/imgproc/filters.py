import numpy as np

def center_crop(input_img, out_shape):
    """Return a center cropped copy of an image array

    Args:
        input_img:    numpy array
        output_shape: outp dimensions (x,y)
    """
    img = np.copy(input_img)
    x, y = img.shape[:2]
    new_x, new_y = out_shape
    if (new_x > x) or (new_y > y):
        raise RuntimeError("Output shape must be smaller than img shape")
    dx = x - new_x
    dy = y - new_y
    cropped = img[dx/2: x - dx/2, dy/2: y - dy/2]
    return cropped

