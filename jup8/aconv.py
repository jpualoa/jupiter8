# ==============================================================================
# Build an A-ConvNets network
#
# Reference: "Target Classification Using the Deep Convolutional Networks for
#             SAR Images" by Chen et al.
# ==============================================================================
# Sandard imports

# External dependencies
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras import Input

INPUT_SHAPE = (128, 128, 1)

def get_aconv(input_shape=INPUT_SHAPE):
    input_tensor = Input(shape=input_shape)
    x = Conv2D(16, (5,5), activation='relu')(input_tensor)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(32, (5,5), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (6,6), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, (5,5), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(input_tensor, x)
    return model

if __name__ == '__main__':
    model = get_aconv()
    model.summary()
