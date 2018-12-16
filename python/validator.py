# External dependencies
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold


OPTIMIZER = Adam(0.0002, 0.5)

def get_validator(img_shape, quiet=True):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(3, activation='sigmoid'))
    if not quiet: model.summary()
    return model

class Validator(object):
    def __init__(self, img_shape, model=None, optimizer=OPTIMIZER,
                 loss='categorical_crossentropy', metrics=['accuracy'], quiet=True):
        if model is None:
            model = get_validator(img_shape, quiet)
        self.model = model

    def build(self):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return self.model

    def train(self, data, labels, epochs, batch_size):
        X_train = data / 127.5 - 1
        X_train = np.expand_dims(X_train, axis=3)
        estimator = KerasClassifier(build_fn=self.build, epochs=epochs, batch_size=batch_size, verbose=1)
        kfold = KFold(n_splits=10, shuffle=True)
        results = cross_val_score(estimator, X_train, labels, cv=kfold)
        return results
