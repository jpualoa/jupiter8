# Standard imports
import os
import json
import logging
import datetime
import numpy as np

# Keras dependencies
from keras.layers import Input
from keras.models import Model

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

SAMPLES_DIRNAME = 'samples'

class GAN(object):
    def __init__(self, generator, discriminator, optimizer, latent_dim,
                 loss='binary_crossentropy', metrics=['accuracy']):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

        self.training_results = []
        self._outdir = '.' # Directory to write log files and sample images to

        # ========================
        # Build the combined model
        # ========================
        #
        # Compile discriminator
        self.discriminator.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss=loss, optimizer=optimizer)

    def _get_labels(self, batch_size):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        return (valid, fake)

    def _log_progress(self, epoch, d_loss, g_loss):
        tstamp = datetime.datetime.utcnow()
        progress = {'epoch':epoch, 'd_loss':float(d_loss[0]), 'd_accuracy':100*float(d_loss[1]),
                    'g_loss':float(g_loss), 'time':tstamp.strftime("%Y-%m-%dT%H:%M:%S.%s")}
        self.training_results.append(progress)
        d_loss_var = np.array([x['d_loss'] for x in self.training_results]).var()
        self.training_results[-1]['d_loss_var'] = d_loss_var
        g_loss_var = np.array([x['g_loss'] for x in self.training_results]).var()
        self.training_results[-1]['g_loss_var'] = g_loss_var
        progress = self.training_results[-1]
        print ("{epoch} [D loss: {d_loss:.4f}, acc.: {d_accuracy:.4f}, var:{d_loss_var:.4f}] [G loss: {g_loss:.4f}, var:{g_loss_var:.4f}]".format(**progress))

    def _save_samples(self, epoch):
        samples_dir = os.path.join(self._outdir, SAMPLES_DIRNAME)
        if not os.path.exists(samples_dir):
            os.mkdir(samples_dir, 0o755)
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fname = os.path.join(samples_dir, "synth_%d.png" % epoch)
        fig.savefig(fname)
        plt.close()

    # ==========================================================================
    # Public Interface
    # ==========================================================================
    def set_outdir(self, dirname):
        dirname = os.path.expandvars(dirname)
        if not os.path.exists(dirname): raise ValueError("%s does not exist" % dirname)
        self._outdir = dirname

    @property
    def results(self):
        return self.training_results

    def train(self, data, epochs, batch_size, save_interval=50):
        """Train the networks

        Args:
            data (np.ndarray)
                Array of images to train on

            epochs (int)
                number of training epochs

            batch_size (int)
                number of images to train on each epoch

            save_interval (int)
                save example generated images at this epoch interval
                (does not save examples if set to 0)
        """
        # Rescale -1 to 1
        X_train = data / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid, fake = self._get_labels(batch_size)

        # Clear training_results
        self.training_results = []
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images from the data
            idx = np.random.choice(X_train.shape[0], batch_size, replace=True)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Log progress
            self._log_progress(epoch, d_loss, g_loss)

            # Save generated samples
            if save_interval > 0:
                if epoch % save_interval == 0: self._save_samples(epoch)
