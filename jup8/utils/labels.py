import numpy as np
from collections import namedtuple

Labels = namedtuple('Labels', ['valid', 'fake'])

def get_hard_labels(size):
    """Returns a struct of hard labels

    valid = 1
    fake = 0
    """
    valid = np.ones(size)
    fake = np.zeros(size)
    return Labels(valid=valid, fake=fake)

def get_hard_labels_flipped(size):
    """
    Returns a struct of hard 'flipped' labels

    valid = 0
    fake = 1
    """
    fake = np.ones(size)
    valid = np.zeros(size)
    return Labels(valid=valid, fake=fake)

def get_soft_labels(size):
    """Returns an struct of soft labels

    valid = 0.9 to 1.0
    fake = 0.0 to 0.1
    """
    valid = np.random.randint(9000, 10001, size=size) / 10000.0
    fake = np.random.randint(0, 1001, size=size) / 10000.0
    return Labels(valid=valid, fake=fake)

def get_soft_labels_flipped(size):
    """Returns an struct of soft 'flipped' labels

    valid = 0.0 to 0.1
    fake = 0.9 to 1.0
    """
    fake = np.random.randint(9000, 10001, size=size) / 10000.0
    valid = np.random.randint(0, 1001, size=size) / 10000.0
    return Labels(valid=valid, fake=fake)

LABEL_TYPES = {
    'hard': get_hard_labels,
    'hard_flipped': get_hard_labels_flipped,
    'soft': get_soft_labels,
    'soft_flipped': get_soft_labels_flipped}

def get_labels(size, label_type='hard'):
    try:
        func = LABEL_TYPES[label_type.lower()]
    except KeyError:
        raise ValueError("label type: %s not recognized" % label_type)
    else:
        return func(size)

