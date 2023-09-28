import cv2
import datetime
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
from skimage import morphology
import tensorflow as tf
import time
from typing import Callable, List, Tuple

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def visualise_model(model)->None:
    display.display(tf.keras.utils.plot_model(model, show_shapes=True, dpi=64))

class ColourGAN:
    """ ColourGAN
        ColourGAN learns to reconstruct full colour images from some reduced input image
    """
    def __init__(
            self,
            generator: tf.keras.Model,
            discriminator: tf.keras.Model,
            inspect_img_fnc: Callable[[tf.Tensor, tf.Tensor], None] | None = None,
            _lambda: int = 100,
            loss_function: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
            generator_optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            discriminator_optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            checkpoint_dir: str | Path = 'training_checkpoints/',
            log_dir: str | Path = 'logs/'
    )->None:

        self.inspect_fnc = inspect_img_fnc

        self.generator = generator
        self._lambda = _lambda
        self.generator_optimizer = generator_optimizer
        
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer

        self.loss_object = loss_function

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator)
        
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def generator_loss(self, disc_generated_output, gen_output, target, inpt):

        # generator loss
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        l1_loss = tf.reduce_mean(tf.abs(target - inpt))

        total_gen_loss = tf.reduce_mean(gan_loss + (self._lambda * l1_loss))

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):

        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
    
    @tf.function
    def train_step(self, input_image, target_image, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target_image], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_pixel_loss = self.generator_loss(disc_generated_output, gen_output, target_image, input_image)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_pixel_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

        return gen_total_loss, gen_gan_loss, gen_pixel_loss, disc_loss

    def fit(self, train_ds, test_ds, steps):
        example_trgt, example_inpt, example_pth = next(iter(test_ds.take(1)))
        start = time.time()

        patience = 5
        count = 0
        best_gen_pixel_loss = float('inf')

        gen_total_loss_history, gen_gan_loss_history, gen_pixel_loss_history, disc_loss_history = []

        for step, (trgt, inpt, _) in train_ds.repeat().take(steps).enumerate():
            if (step % 1000) == 0:
                display.clear_output(wait=True)

                if step != 0:
                    print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

                    print(f"Total Generator Loss: {gen_total_loss:.5f}")
                    gen_total_loss_history.append(gen_total_loss)
                    print(f"Generator GAN Loss: {gen_gan_loss:.5f}")
                    gen_gan_loss_history.append(gen_gan_loss)
                    print(f"Generator Pixel Loss: {gen_pixel_loss:.5f}")
                    gen_pixel_loss_history.append(gen_pixel_loss)
                    print(f"Discriminator Loss: {disc_loss:.5f}")
                    disc_loss_history.append(disc_loss)

                    # Check if gen_pixel_loss has improved
                    if gen_pixel_loss < best_gen_pixel_loss:
                        best_gen_pixel_loss = gen_pixel_loss
                        count = 0  # Reset the count of steps without improvement
                    else:
                        count += 1

                    # Check if patience has been exhausted
                    if count >= patience:
                        print(f'Early stopping at step {step} due to no improvement in gen_pixel_loss.')
                        break  # Terminate training loop

                start = time.time()

                self.inspect_fnc(self.generator, inpt=example_inpt, tar=example_trgt, filepath=example_pth) if self.inspect_fnc else None


                print(f"Step: {step//1000}k")

            gen_total_loss, gen_gan_loss, gen_pixel_loss, disc_loss = self.train_step(input_image=inpt, target_image=trgt, step=step)

            # Training step
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)

            # Save (checkpoint) the model after 5000 steps
            if ((step + 1) % 5000) == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        return gen_total_loss_history, gen_gan_loss_history, gen_pixel_loss_history, disc_loss_history 

    def load_checkpoint(self, checkpoint_dir = None):
        if not checkpoint_dir:
            checkpoint_dir = self.checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))