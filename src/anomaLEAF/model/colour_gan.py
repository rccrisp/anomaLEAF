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
            output_shape: Tuple[int,int,int],
            input_shape: Tuple[int,int,int],
            inspect_img_fnc: Callable[[tf.Tensor, tf.Tensor], None] | None = None,
            _lambda: int = 100,
            loss_function: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
            generator_optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            discriminator_optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            checkpoint_dir: str | Path = 'training_checkpoints/',
            log_dir: str | Path = 'logs/'
    )->None:
        self.output_shape = output_shape
        self.input_shape = input_shape

        self.inspect_fnc = inspect_img_fnc

        self.generator = self.build_generator()
        self._lambda = _lambda
        self.generator_optimizer = generator_optimizer
        
        self.discriminator = self.build_discriminator()
        self.discriminator_optimizer = discriminator_optimizer

        self.loss_object = loss_function

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator)
        
        self.summary_writer = tf.summary.create_file_writer(
            log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        down_stack = [
            downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            downsample(128, 4),  # (batch_size, 64, 64, 128)
            downsample(256, 4),  # (batch_size, 32, 32, 256)
            downsample(512, 4),  # (batch_size, 16, 16, 512)
            downsample(512, 4),  # (batch_size, 8, 8, 512)
            downsample(512, 4),  # (batch_size, 4, 4, 512)
            downsample(512, 4),  # (batch_size, 2, 2, 512)
            downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            upsample(512, 4),  # (batch_size, 16, 16, 1024)
            upsample(256, 4),  # (batch_size, 32, 32, 512)
            upsample(128, 4),  # (batch_size, 64, 64, 256)
            upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_shape[-1], 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='sigmoid')  # (batch_size, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def generator_loss(self, disc_generated_output, gen_output, target, inpt):

        # generator loss
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # calculate the pixel loss of the reconstructed pixels
        # difference = tf.abs(target - inpt)
        # grayscale_differences = tf.reduce_sum(difference, axis=-1)
        # num_differing_pixels = tf.reduce_sum(tf.cast(grayscale_differences > 0, tf.float32), axis=[1, 2]) # calculate the number of reconstructed pixels
        # average_pixel_loss = tf.reduce_sum(tf.abs(target - gen_output), axis=[1,2,3])/(num_differing_pixels+1e-5)

        # batch_pixel_loss = tf.reduce_mean(average_pixel_loss)

        # total_gen_loss = tf.reduce_mean(gan_loss + (self._lambda * batch_pixel_loss))

        l1_loss = tf.reduce_mean(tf.abs(target - inpt))

        total_gen_loss = tf.reduce_mean(gan_loss + (self._lambda * l1_loss))

        return total_gen_loss, gan_loss, l1_loss


    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=self.input_shape, name='input_image')
        tar = tf.keras.layers.Input(shape=self.output_shape, name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

        down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                        kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

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

    def fit(self, train_ds, test_ds, steps):
        example_trgt, example_inpt, example_pth = next(iter(test_ds.take(1)))
        start = time.time()

        for step, (trgt, inpt, _) in train_ds.repeat().take(steps).enumerate():
            if (step % 1000) == 0:
                display.clear_output(wait=True)

                if step != 0:
                    print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

                start = time.time()

                self.inspect_fnc(self.generator, inpt=example_inpt, tar=example_trgt, filepath=example_pth) if self.inspect_fnc else None
                
                print(f"Step: {step//1000}k")

            self.train_step(input_image=inpt, target_image=trgt, step=step)

            # Training step
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)


            # Save (checkpoint) the model once 20% of the steps have been taken
            if ((step + 1) % 5000) == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def load_checkpoint(self, checkpoint_dir = None):
        if not checkpoint_dir:
            checkpoint_dir = self.checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))