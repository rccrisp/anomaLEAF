import cv2
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
from skimage import morphology
import tensorflow as tf
import time
from typing import List

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

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

class ColourGAN:
    """ ColourGAN

    """
    def __init__(
            self,
            input_shape: int = 256,
            output_channels: int = 3,
            _lambda: int = 100,
            generator_optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            discriminator_optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            checkpoint_dir: str | Path = './training_checkpoints'
    )->None:
        self.input_shape = input_shape
        self.output_channels = output_channels

        self.generator = self.build_generator()
        self._lambda = _lambda
        self.generator_optimizer = generator_optimizer
        
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
        self.discriminator_optimizer = discriminator_optimizer

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator)
        
    def visualise_model(model)->None:
        tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=[self.shape, self.shape, 1])

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
        last = tf.keras.layers.Conv2DTranspose(self.output_channels, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh')  # (batch_size, 256, 256, 3)

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

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self._lambda * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[self.shape, self.shape, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.shape, self.shape, 3], name='target_image')

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
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        # with summary_writer.as_default():
        #     tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        #     tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        #     tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        #     tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

    def fit(self, train_ds, test_ds, steps):
        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step) % 1000 == 0:
                display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            generate_images(self.generator, example_input, example_target)
            print(f"Step: {step//1000}k")

            self.train_step(input_image, target, step)

            # Training step
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)


            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 5000 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def load_checkpoint(self, checkpoint_dir = None):
        if not checkpoint_dir:
            checkpoint_dir = self.checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


class ColourANOM:
    """ ColourANOM

    """
    def __init__(
            self,
            train_dataset: tf.data.Dataset,
            test_dataset: tf.data.Dataset,
            model: ColourGAN
    )->None:
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.model = model

    def anomaly_score(self, rgb, reconstructed):
            return np.linalg.norm(reconstructed - rgb)
    
    def threshold(self, score):
        return score > 20
    
    def anomaly_map(self, rgb, reconstructed):
        color_difference = np.abs(reconstructed - rgb)
        return color_difference/np.max(color_difference)

    def predict(self, img_rgb: tf.Tensor)->dict:

        img_grayscale = tf.image.rgb_to_grayscale(img_rgb)
        img_reconstructed = self.model.generator(img_grayscale, training=True)

        pred_score = self.anomaly_score(img_rgb, img_reconstructed)
        pred_label = "ANOMALOUS" if self.threshold(pred_score) else "NORMAL"

        anomaly_map = self.anomaly_map(img_rgb, img_reconstructed)
        
        colour_map = anomaly_map_to_color_map(anomaly_map)
        heatmap = superimpose_anomaly_map(anomaly_map, img_rgb)
        mask = compute_mask(anomaly_map, 20)

        return {"image": img_rgb, 
                "reconstructed": img_reconstructed,
                "anomaly_map": anomaly_map,
                "heatmap": heatmap,
                "pred_score": pred_score,
                "pred_label": pred_label,
                "pred_mask": mask
                }
