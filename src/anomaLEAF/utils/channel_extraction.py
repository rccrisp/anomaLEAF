import os
from pathlib import Path
import tensorflow as tf
import tensorflow_io as tfio
from typing import Callable, List, Tuple, Dict

def load(image_file: str,
         img_size: int,
         channel_extraction: Dict[Callable[[tf.Tensor], tf.Tensor], List[int]]
         ) -> Tuple[tf.Tensor, tf.Tensor]:

    # Check if channel_extraction is a dictionary
    if not isinstance(channel_extraction, dict):
        raise TypeError("channel_extraction must be a dictionary")

    # Check the types of keys (callable) and values (list of integers)
    for transform, channels in channel_extraction.items():
        if transform != None:
            if not callable(transform):
                raise TypeError(f"The key {transform} in channel_extraction is not callable")
        if not all(channel in [0, 1, 2] for channel in channels):
            raise ValueError(f"The value {channels} associated with key {transform} contains invalid channels")
        
    real_image = tf.image.decode_image(tf.io.read_file(image_file), channels=3, expand_animations=False)
    real_image = tf.image.convert_image_dtype(real_image, dtype=tf.float32)
    real_image = tf.image.resize(real_image, [img_size, img_size])

    # Initialize reduced_image
    reduced_image = tf.zeros((img_size, img_size, 0), dtype=tf.float32)

    # Apply image transformation if it's not None
    for transform, channels in channel_extraction.items():
        transformed_image = transform(real_image) if transform is not None else real_image
        channel_list = tf.split(transformed_image, num_or_size_splits=transformed_image.shape[-1], axis=-1)
        selected_channels = [channel_list[channel] for channel in channels]
        reduced_image = tf.concat([reduced_image, tf.concat(selected_channels, axis=-1)], axis=-1)
    
    return reduced_image, real_image

def generate_images(model, inpt, tar, pixel_range=1, filepath=None):
    prediction = model(inpt, training=True)

    # Determine the number of channels in the input image
    num_channels = inpt.shape[-1]

    # Initialize an empty list to store images for display
    display_list = []

    for i in range(0, num_channels):
        display_list.append(inpt[0, :, :, i])

    # Add the ground truth and predicted images to the display list
    display_list.extend([tar[0], prediction[0]])

    num_images = len(display_list)  # Number of images to display
    num_cols = num_images  # Set the number of columns for subplots

    # Define titles for each image element
    titles = ['Channel {}'.format(i) for i in range(num_channels)] + ['Ground Truth', 'Predicted Image']

    plt.figure(figsize=(5 * num_images, 5))  # Adjust figsize based on the number of images

    if filepath != None:
        print(filepath[0])

    for i in range(num_images):
        plt.subplot(1, num_cols, i + 1)
        plt.title(titles[i])
        plt.imshow(display_list[i]*pixel_range) # recover full pixel range for RGB image display
        plt.axis('off')
    plt.show()