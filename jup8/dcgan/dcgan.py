# Standard imports
import os

# Keras dependencies
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam

# Local imports
from gan import GAN

def get_generator(latent_dim, channels, quiet=True):
    model = Sequential()
    model.add(Dense(128 * 32* 32, activation="relu", input_dim=latent_dim))
    model.add(Reshape((32, 32, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    if not quiet: model.summary()
    return model

def build_generator(model, latent_dim):
    """Builds the provided generator model"""
    noise = Input(shape=(latent_dim,))
    img = model(noise)
    return Model(noise, img)

def get_discriminator(img_shape, quiet=True):
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
    model.add(Dense(1, activation='sigmoid'))
    if not quiet: model.summary()
    return model

def build_discriminator(model, img_shape):
    """Builds the provided discriminator model"""
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)

class DCGAN(GAN):
    def __init__(self, latent_dim, img_shape, gen_unbuilt=None, dis_unbuilt=None,
                 optimizer=None, quiet=True, **kwargs):
        """Constructor
    
        Args:
            latent_dim (int)
                length of generatyor input noise vector

            img_shape (tuple)
                shape of training images (row, cols, channels)

            gen_unbuilt (keras.Sequential)
                UN-built generator model as returned by get_generator()

            dis_unbuilt (keras.Sequential)
                UN-built discriminator model as returned by get_discriminator()
        """
        # build Generator
        if gen_unbuilt is None:
            gen_unbuilt = get_generator(latent_dim, img_shape[2], quiet)
        generator = build_generator(gen_unbuilt, latent_dim)

        # build Discriminator
        if dis_unbuilt is None:
            dis_unbuilt = get_discriminator(img_shape, quiet)
        discriminator = build_discriminator(dis_unbuilt, img_shape)

        # setup optimizer
        if optimizer is None:
           optimizer = Adam(0.0002, 0.5)
    
        # init
        super(DCGAN, self).__init__(generator, discriminator, optimizer, latent_dim, **kwargs)

    def save_generator_weights(self, filename):
        """Save the built generator weights"""
        self.generator.save_weights(filename)

    def save_discriminator_weights(self, filename):
        """Save the built discriminator weights"""
        self.discriminator.save_weights(filename)


if __name__ == '__main__':
    import sys
    from keras.utils import plot_model
    outfile = sys.argv[1]
    g = get_generator(100, 1)
    g.summary()
    plot_model(g, to_file=outfile, show_shapes=True)
