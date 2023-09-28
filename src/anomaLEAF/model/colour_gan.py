from IPython import display
import tensorflow as tf

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

class ColourGAN(tf.keras.Model):
    
    def __init__(
            self,
            generator: tf.keras.Model,
            discriminator: tf.keras.Model,
            ):
        super().__init__()
        # discriminator
        self.discriminator = discriminator
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        
        # generator
        self.generator = generator
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
    
    def compile(self, d_optimiser, g_optimser, loss_fn, _lambda):
        super().compile()
        self.d_optimiser = d_optimiser
        self.g_optimiser = g_optimser
        self.loss_fn = loss_fn
        self._lambda=_lambda

    def generator_loss(self, disc_generated_output, y_true, y_pred):

        gan_loss = self.loss_fn(tf.ones_like(disc_generated_output), disc_generated_output)

        pixel_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

        total_loss = gan_loss + pixel_loss*self._lambda

        return total_loss, pixel_loss 

    def discriminator_loss(self, y_real, y_pred):

        # discriminator loss on real images
        real_loss = self.loss_fn(tf.ones_like(y_real), y_real)

        # discriminator loss on generated images
        generated_loss = self.loss_fn(tf.zeros_like(y_pred), y_pred)

        return real_loss + generated_loss

    def train_step(self, data):
        
        # extract input and target
        x, y_true = data

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            y_gen = self.generator(x, training=True)

            # discriminator on real images
            disc_real_output = self.discriminator([x, y_true], training=True)
            # discriminator on generated images
            disc_generated_output = self.discriminator([x, y_gen], training=True)

            gen_total_loss, gen_pixel_loss = self.generator_loss(disc_generated_output, y_true, y_gen)

            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.g_optimiser.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.d_optimiser.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        self.g_loss_tracker.update_state(gen_total_loss)
        self.d_loss_tracker.update_state(disc_loss)

        return {
            "g_loss": self.g_loss_tracker.result(),
            "gen_pixel_loss": gen_pixel_loss,
            "d_loss": self.d_loss_tracker.result()
        }
    
    def test_step(self, data):
        
        # extract input and target
        x, y_true = data

        y_gen = self.generator(x, training=False)

        # discriminator on real images
        disc_real_output = self.discriminator([x, y_true], training=False)
        # discriminator on generated images
        disc_generated_output = self.discriminator([x, y_gen], training=False)

        gen_total_loss, gen_pixel_loss = self.generator_loss(disc_generated_output, y_true, y_gen)

        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        self.g_loss_tracker.update_state(gen_total_loss)
        self.d_loss_tracker.update_state(disc_loss)

        return {
            "g_loss": self.g_loss_tracker.result(),
            "gen_pixel_loss": gen_pixel_loss,
            "d_loss": self.d_loss_tracker.result()
        }
    