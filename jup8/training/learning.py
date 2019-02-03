"""Learning rate helper functions"""
# Standard imports
import math

# External dependencies

def get_lr_metric(optimizer):
    """Custom metric for displaying the current learning rate"""
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def step_decay(epoch, lr_in):
    """Reduces learning rate at fixed epoch intervals
    """
    drop = 0.1
    epochs_drop = 50
    factor = math.pow(drop, math.floor((1 + epoch)/epochs_drop))
    if factor == 0:
        lrate = lr_in
    else: 
        lrate = lr_in * math.pow(drop, math.floor((1 + epoch)/epochs_drop))
    return lrate
