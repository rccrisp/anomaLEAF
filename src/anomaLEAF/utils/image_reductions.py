import random
import cv2
import tensorflow_io as tfio
from typing import Callable, List, Tuple, Dict
from PIL import Image
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_grayscale_patched(img_path: str,
                           img_size: int = 256,
                           patch_num: int = -1):
    
    # takes an image with the background removed and returns a bounding box of the leaf as coordinates
    def leaf_focus(image):
        # Find all non-zero pixels in the image
        non_zero_pixels = cv2.findNonZero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        if non_zero_pixels is not None:
            # Get the bounding box around the non-zero pixels
            rect = cv2.boundingRect(non_zero_pixels)
            x, y, w, h = rect
            # Crop the image to the bounding box
            cropped_image = image[y:y+h, x:x+w]
            return cropped_image, rect
        else:
            return None, None
        
    def patch_grayscale(image, bounding_box, n, patch_idx):
        if image is None or bounding_box is None:
            return None

        left, upper, right, lower = bounding_box

        w = right - left
        h = lower - upper

        # Calculate the size of each grid square
        grid_width = w // n
        grid_height = h // n

        if patch_idx < 0 or patch_idx >= n * n:
            # Choose a random grid square to convert to grayscale
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
        else:
            # Use the specified grayscale_patch index
            x = patch_idx % n
            y = patch_idx // n

        # Calculate the coordinates of the selected grid square
        grid_left = left + x * grid_width
        grid_upper = upper + y * grid_height
        grid_right = grid_left + grid_width
        grid_lower = grid_upper + grid_height

        # Crop the selected grid square
        grid_square = image[grid_upper:grid_lower, grid_left:grid_right]

        # Convert the selected grid square to grayscale using OpenCV
        grayscale_grid_square = cv2.cvtColor(grid_square, cv2.COLOR_BGR2GRAY)
        grayscale_grid_square = cv2.cvtColor(grayscale_grid_square, cv2.COLOR_GRAY2BGR)  # Convert it back to BGR

        # Create a mask for the selected grid square
        grid_mask = np.zeros_like(grid_square, dtype=np.uint8)

        # Replace the selected grid square in the original image with the grayscale grid square
        modified_image = image.copy()
        modified_image[grid_upper:grid_lower, grid_left:grid_right] = grayscale_grid_square

        # Create an image with a grid to visualize the division
        grid_image = image.copy()
        for i in range(1, n):
            x_pos = left + i * grid_width
            y_pos = upper + i * grid_height
            cv2.line(grid_image, (x_pos, upper), (x_pos, lower), (0, 255, 0), 1)
            cv2.line(grid_image, (left, y_pos), (right, y_pos), (0, 255, 0), 1)

        return modified_image, grid_image
    
    # Load the image
    full_image = cv2.imread(img_path)
    
    # Get the cropped image and bounding box
    full_image, bounding_box = leaf_focus(full_image)

    # Create the reduced_image
    reduced_image, grid_image = patch_grayscale(image=full_image, bounding_box=bounding_box, n=patch_num, patch_idx=-1)

    full_image_tensor = tf.convert_to_tensor(full_image)
    full_image_tensor = tf.image.convert_image_dtype(full_image_tensor, dtype=tf.float32)
    full_image_tensor = tf.image.resize(full_image_tensor, [img_size, img_size])

    reduced_image_tensor = tf.convert_to_tensor(reduced_image)
    reduced_image_tensor = tf.image.convert_image_dtype(reduced_image_tensor, dtype=tf.float32)
    reduced_image_tensor = tf.image.resize(reduced_image_tensor, [img_size, img_size])

    grid_image_tensor = tf.convert_to_tensor(grid_image)
    grid_image_tensor = tf.image.convert_image_dtype(grid_image_tensor, dtype=tf.float32)
    grid_image_tensor = tf.image.resize(grid_image_tensor, [img_size, img_size])

    return full_image_tensor, reduced_image_tensor, grid_image_tensor

def load_channel_extraction(image_path: str,
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
    
    return real_image, reduced_image

def generate_images_channel_extraction(model, inpt, tar, pixel_range=1, filepath=None):
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