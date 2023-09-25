import os
from pathlib import Path
import tensorflow as tf
import tensorflow_io as tfio
from typing import Callable, List, Tuple, Dict

class Dataset():
    """ Dataset
    
    """
    def __init__(
            self,
            normal_dir: str | Path,
            anomalous_dir: str | Path,
            channel_extraction: Dict[Callable[[tf.Tensor], tf.Tensor], List[int]],
            batch_size: int = 256,
            buffer_size: int = 512,
            img_size: int = 256,
            verbose: bool = True
    )-> None:

        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.img_size = img_size
        self.channel_extraction = channel_extraction
    
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
        print(f"{dataset_type.capitalize()} Reduced Feature Input:")
        for transform, channels in self.channel_extraction.items():
            print(f"\t{transform.__name__ if transform is not None else transform} {channels}")
        print(f"{dataset_type.capitalize()} Batch Size:", batch_size)
        print(f"{dataset_type.capitalize()} Number of Batches:", num_batches)
        print()


    def load_train_image(self, file_path):
        reduced, real = load(file_path, img_size=self.img_size, channel_extraction=self.channel_extraction)
        # add in transformations here later
        reduced, real = self.random_jitter(reduced, real)
        return reduced, real, file_path

    def load_test_image(self, file_path):
        reduced, real = load(file_path,img_size=self.img_size, channel_extraction=self.channel_extraction)

        return reduced, real, file_path
    
    @tf.function()
    def random_jitter(self,reduced_img, full_img):
        reduced_shape = reduced_img.shape
        full_shape = full_img.shape

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            reduced_img = tf.image.flip_left_right(reduced_img)
            full_img = tf.image.flip_left_right(full_img)

        return reduced_img, full_img

    # Normalizing the images to [0, 1]
    def normalize(self, image, range=None):
        max_value = range if range else tf.reduce_max(image)
        return image/max_value
    
