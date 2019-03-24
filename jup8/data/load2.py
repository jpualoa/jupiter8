# Standard imports
import logging

# External dependencies
import numpy as np
import cv2
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

# Local imports
from imgproc.filters import center_crop

IMG_HEIGHT = 128
IMG_WIDTH = 128

log = logging.getLogger(__name__)

def load_images(filenames, crop_dim=None, add_channels=True):
    """Returns an array of loaded images

    Args:
        filenames:    list of image files
        resize_dim:   (rows, cols) image should be resized tp
        crop_dim:     (rows, cols) to crop to (if provided)
        add_channels: expand dims to add channel
    """
    images = []
    for f in filenames:
        # Read image
        img = cv2.imread(f, 0)
        # Crop image
        in_rows, in_cols = img.shape
        if crop_dim:
            out_rows, out_cols = crop_dim
            img = center_crop(img, out_rows, out_cols)
        # Add channel dimension
        if add_channels:
            img = np.expand_dims(img, axis=2)
        images.append(img)
    return np.array(images)

def get_training_generator(images, labels, batch_size=32, test_size=0.1, repeat_samples=0):
    """Returns training and validator generators

    Args:
        images: array of image data
        labels: list of encoded (int) labels for images
    """
    gen_train = image.ImageDataGenerator(rescale=1./255,
                                     rotation_range=20,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     horizontal_flip=True,
                                     data_format='channels_last')
    
    gen_val = image.ImageDataGenerator(rescale=1./255,
                                       data_format='channels_last')

    # Partition training and validation data
    x_train, x_val, y_train, y_val = train_test_split(images, labels,
        test_size=test_size)

    # Repeat training samples
    if repeat_samples:
        log.debug("Repeating training samples %d times" % repeat_samples)
        x_train = np.repeat(x_train, repeat_samples, axis=0)
        y_train = np.repeat(y_train, repeat_samples, axis=0)

    log.debug("x_train: %s, y_train: %s" % (str(x_train.shape),str(y_train.shape)))
    log.debug("x_val: %s, y_val: %s" % (str(x_val.shape),str(y_val.shape)))

    # Get data generators
    train = gen_train.flow(x_train, y_train, batch_size=batch_size)
    val = gen_val.flow(x_val, y_val, batch_size=batch_size)

    return (train, val)

def get_test_generator(images, labels, batch_size=32):
    """Returns a test data generator

    Args:
        images: array of image data
        labels: list of encoded (int) labels for images
    """
    gen = image.ImageDataGenerator(rescale=1./255,
                                   data_format='channels_last')
    test = gen.flow(images, labels, batch_size=batch_size, shuffle=False)
    return test
