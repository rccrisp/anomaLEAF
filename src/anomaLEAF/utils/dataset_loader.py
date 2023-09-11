import os
from pathlib import Path
import tensorflow as tf
import tensorflow_io as tfio
from typing import Callable, List, Tuple

def load(image_file: str,
         img_size: int,
         image_transformation: Callable[[tf.Tensor], tf.Tensor] = None,
         channels: List[int] = [0,1,2]) -> Tuple[tf.Tensor, tf.Tensor]:
    
    if image_transformation is not None:
        # Check if image_transformation is a callable function from tf.image
        if not callable(image_transformation) or not hasattr(tfio.experimental.color, image_transformation.__name__):
            raise ValueError("image_transformation should be a callable function from tfio.experimental.color or None")

    real_image = tf.image.decode_image(tf.io.read_file(image_file), channels=3, dtype=tf.float32, expand_animations=False)
    real_image = tf.image.resize(real_image, [img_size, img_size])

    # Apply image transformation if it's not None
    transformed_image = image_transformation(real_image) if image_transformation is not None else real_image
    channel_list = tf.split(transformed_image, num_or_size_splits=transformed_image.shape[-1], axis=-1)
    selected_channels = [channel_list[idx] for idx in channels]
    reduced_image = tf.concat(selected_channels, axis=-1)
    
    return reduced_image, real_image

class Dataset():
    """ Dataset
    
    """
    def __init__(
            self,
            normal_dir: str | Path,
            anomalous_dir: str | Path,
            batch_size: int = 256,
            buffer_size: int = 512,
            img_size: int = 256,
            colourspace_transformation: Callable[[tf.Tensor], tf.Tensor] = None,
            channels: List[int] = [0,1,2],
            verbose: bool = True
    )-> None:
        
        if not all(channel in [0, 1, 2] for channel in channels):
            raise ValueError("Channel should only include values of 0, 1, or 2.")

        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.img_size = img_size
        self.colourspace = colourspace_transformation
        self.channels = channels
    
        self.normal_dir = Path(normal_dir)
        self.normal_dataset = tf.data.Dataset.list_files(str(self.normal_dir) + '/*')
        self.normal_dataset = self.normal_dataset.map(self.load_train_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.normal_dataset = self.normal_dataset.shuffle(buffer_size=self.buffer_size)
        self.normal_dataset = self.normal_dataset.batch(self.batch_size)

        self.anomalous_dir = Path(anomalous_dir)
        self.anomalous_dataset = tf.data.Dataset.list_files(str(self.anomalous_dir) + '/*')
        self.anomalous_dataset = self.anomalous_dataset.map(self.load_test_image)
        self.anomalous_dataset = self.anomalous_dataset.batch(self.batch_size)

         # Print directory, image size, batch size, number of batches, number of channels, and transformation
        if verbose:
            self.print_dataset_info(self.normal_dir, self.normal_dataset, 'normal')
            self.print_dataset_info(self.anomalous_dir, self.anomalous_dataset, 'anomalous')

    def print_dataset_info(self, dataset_dir, dataset, dataset_type):
        # Take one element from the dataset to get the image sizes and other information
        reduced, full, path = next(iter(dataset))
        reduced_image_size = reduced.shape[1:]  # Exclude batch size and channels
        full_image_size = full.shape[1:]  # Exclude batch size and channels 
        batch_size = reduced.shape[0]  # Batch size
        num_batches = len(dataset) # number of batches

        print(f"{dataset_type.capitalize()} Dataset Directory:", dataset_dir)
        print(f"{dataset_type.capitalize()} Reduced Image Size:", reduced_image_size)
        print(f"{dataset_type.capitalize()} Full Image Size:", full_image_size)
        print(f"{dataset_type.capitalize()} Extracted Channel Indices:", self.channels)
        print(f"{dataset_type.capitalize()} Batch Size:", batch_size)
        print(f"{dataset_type.capitalize()} Number of Batches:", num_batches)
        print(f"{dataset_type.capitalize()} Colourspace Transformation:", self.colourspace)
        print()


    def load_train_image(self, file_path):
        reduced, real = load(file_path, img_size=224, channels=self.channels)
        # add in transformations here later
        reduced, real = self.random_jitter(reduced, real)
        real = self.normalize(real, 255.0)
        reduced = self.normalize(reduced, 100.0)    # this is specifically for luminance channel
        return reduced, real, file_path

    def load_test_image(self, file_path):
        reduced, real = load(file_path,img_size=224, channels=self.channels)
        real = self.normalize(real, range=255.0)
        reduced = self.normalize(reduced, range=100.0)    # this is specifically for luminance channel
        return reduced, real, file_path
    
    @tf.function()
    def random_jitter(self,reduced_img, full_img):
        # Resizing to 286x286
        reduced_img = tf.image.resize(reduced_img, [286, 286],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        full_img = tf.image.resize(full_img, [286, 286],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Generate a random seed
        seed = tf.random.uniform([], maxval=tf.int32.max, dtype=tf.int32)

        # Random cropping back to 256x256 with the same random seed
        reduced_img = tf.image.stateless_random_crop(reduced_img, size=reduced_img.shape, seed=[seed, 0])
        full_img = tf.image.stateless_random_crop(full_img, size=full_img.shape, seed=[seed, 0])

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            reduced_img = tf.image.flip_left_right(reduced_img)
            full_img = tf.image.flip_left_right(full_img)

        return reduced_img, full_img

    # Normalizing the images to [0, 1]
    def normalize(self,image, range=None):
        max_value = range if range else tf.reduce_max(image)
        return image/max_value
    
