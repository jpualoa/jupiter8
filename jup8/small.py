# Sandard imports

# External dependencies
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras import Input

INPUT_SHAPE = (128, 128, 1)

def get_model(input_shape=INPUT_SHAPE):
    input_tensor = Input(shape=input_shape)

    x = Conv2D(18, (9,9), activation='relu')(input_tensor)
    x = MaxPooling2D((6,6))(x)
    x = Conv2D(36, (5,5), activation='relu')(x)
    x = MaxPooling2D((4,4))(x)
    x = Conv2D(120, (4,4), activation='relu')(x)
    x = Dense(120, activation='relu')(x)    
    x = Dense(10, activation='softmax')(x)

    output_tensor = Flatten()(x)
    model = Model(input_tensor, output_tensor)
    return model

if __name__ == '__main__':
    import sys
    model = get_model()
    model.summary()
    if sys.argv[1]:
        from keras.utils import plot_model
        plot_model(model, show_shapes=True, to_file=sys.argv[1])
