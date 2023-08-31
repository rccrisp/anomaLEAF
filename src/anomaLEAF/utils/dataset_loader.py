import os
from pathlib import Path
import tensorflow as tf

def load(image_file):

  real_image = tf.image.decode_image(tf.io.read_file(image_file), channels=3)

  grayscale_image = tf.image.rgb_to_grayscale(real_image)

  real_image = tf.cast(real_image, tf.float32)
  grayscale_image = tf.cast(grayscale_image, tf.float32)

  return grayscale_image, real_image


class Dataset():
    """ Dataset
    
    """
    def __init__(
            self,
            normal_dir: str | Path,
            test_dir: str | Path,
            batch_size: int = 1
    )-> None:
        self.normal_dir = Path(normal_dir)
        self.normal_dataset = tf.data.Dataset.list_files(str(self.normal_dir) + '/*')
        
        self.test_dir = Path(test_dir)
        self.test_dataset = tf.data.Dataset.list_files(str(self.test_dir) + '/*')
        
        self.batch_size = batch_size

    def create_dataset(self):
        self.normal_dataset = self.normal_dataset.map(self.load_image_train)
        self.normal_dataset = self.normal_dataset.shuffle(buffer_size=1000)
        self.normal_dataset = self.normal_dataset.batch(self.batch_size)
        
        self.test_dataset = self.test_dataset.map(self.load_image_test)
        self.test_dataset = self.test_dataset.batch(self.batch_size)

    def add_data_to_dataset(self, additional_dir: str | Path, is_normal: bool):
        additional_dir = Path(additional_dir)
        try: 
            additional_dataset = tf.data.Dataset.list_files(str(additional_dir) + '/*')
        except :
            print("Warning: Specified directory ({additional_dir}) was empty and was not added to dataset")
            return
        
        additional_dataset = additional_dataset.map(self.load_image_train)
        # You can optionally shuffle the additional data before adding it to the dataset
        # additional_dataset = additional_dataset.shuffle(buffer_size=BUFFER_SIZE)
        additional_dataset = additional_dataset.batch(self.batch_size)
        
        if is_normal:
            # Concatenate the additional data to the existing normal_dataset
            self.normal_dataset = self.normal_dataset.concatenate(additional_dataset)
        else:
            self.test_dataset = self.test_dataset.concatenate(additional_dataset)

    
    def load_image_train(self, file_path):
        grayscale, rgb = load(file_path)
        # add in transformations here later
        grayscale, rgb = self.normalize(grayscale, rgb)
        return grayscale, rgb

    def load_image_test(self, file_path):
        grayscale, rgb = load(file_path)
        grayscale, rgb = self.normalize(grayscale, rgb)

        return grayscale, rgb
    
    # Normalizing the images to [-1, 1]
    def normalize(self,input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image