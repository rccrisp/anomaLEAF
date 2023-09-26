import random
import cv2
import tensorflow_io as tfio
from typing import Callable, List, Tuple, Dict
from PIL import Image
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def image_to_tensor(image: np.ndarray, 
                    size: Tuple[int, int]) -> tf.Tensor:
    tensor = tf.convert_to_tensor(image)
    tensor = tf.image.convert_image_dtype(tensor, dtype=tf.float32)
    tensor = tf.image.resize(tensor, size)
    
    return tensor

def grayscale_patches(image: np.ndarray,
                        patch_num: int,
                        size: Tuple[int, int]) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[Tuple[int, int, int, int]]]:
    """
    """

    image_tensor = image_to_tensor(image, size)

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR) # convert back to 3 channel

    # Get dimensions of the grayscale image
    height, width, _ = grayscale_image.shape

    # Calculate the size of each grid square
    grid_size = height // patch_num

    grayscale_patches = []
    colour_patches = []
    patch_coordinates = []

    for i in range(patch_num):
        for j in range(patch_num):
            # Calculate the coordinates for the current grid square
            x1, y1 = i * grid_size, j * grid_size
            x2, y2 = (i + 1) * grid_size, (j + 1) * grid_size

            patch_image = image.copy()
            patch_image[x1:x2,y1:y2] = grayscale_image[x1:x2,y1:y2]

            patch_image_tensor = image_to_tensor(patch_image, size)

            grayscale_patches.append(patch_image_tensor)
            patch_coordinates.append((x1,x2,y1,y2))
            colour_patches.append(image_tensor)
            
    return colour_patches, grayscale_patches, patch_coordinates
      
    
def grayscale_patches_with_borders(image: np.ndarray,
                                    patch_num: int,
                                    size: Tuple[int, int],
                                    border_area_ratio: float) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    """
        
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR) # convert back to 3 channel

    # Get dimensions of the grayscale image
    height, width, _ = grayscale_image.shape
    
    # Calculate the size of each grid square
    grid_size = height // patch_num
    
    grayscale_patches = []
    color_patches = []
    patch_global_coordinates = []
    patch_local_coordinates = []

    for i in range(patch_num):
        for j in range(patch_num):
            # Calculate the coordinates for the current grid square
            x1, y1 = i * grid_size, j * grid_size
            x2, y2 = (i + 1) * grid_size, (j + 1) * grid_size

            # Create grayscale patch
            modified_image = image.copy()
            modified_image[x1:x2, y1:y2] = grayscale_image[x1:x2, y1:y2]

            # Calculate the border size based on the area
            border_size = int(np.sqrt((border_area_ratio * abs(x2-x1)*abs(y2-y1)) / 4))

            # calculate the coordinates of the expanded patch
            if width <= border_size:
                x1_large, x2_large = 0, width
            else:
                if x1 - border_size/2 < 0:
                    x1_large = 0
                    x2_large = x2 + border_size/2 + abs(x1-border_size/2)
                elif x2 + border_size/2 > width:
                    x2_large = width
                    x1_large = x1-border_size/2 - abs(x2+border_size/2-width)
                else :
                    x1_large, x2_large = x1-border_size/2, x2+border_size/2

            # Cast the coordinates to integers
            x1_large, x2_large = int(x1_large), int(x2_large)

            if height <= border_size:
                y1_large, y2_large = 0, height
            else:
                if y1 - border_size/2 < 0:
                    y1_large = 0
                    y2_large = y2 + border_size/2 + abs(y1 - border_size/2)
                elif y2 + border_size/2 > height:
                    y2_large = height
                    y1_large = y1 - border_size/2 - abs(y2 + border_size/2 - height)
                else:
                    y1_large, y2_large = y1 - border_size/2, y2 + border_size/2

            # Cast the coordinates to integers
            y1_large, y2_large = int(y1_large), int(y2_large)
            
            rgb_patch = image[x1_large:x2_large, y1_large:y2_large]
            rgb_patch_tensor = image_to_tensor(rgb_patch, size)
            color_patches.append(rgb_patch_tensor)
            
            grayscale_patch_with_border = modified_image[x1_large:x2_large, y1_large:y2_large]
            grayscale_patch_with_border_tensor = image_to_tensor(grayscale_patch_with_border, size)
            grayscale_patches.append(grayscale_patch_with_border_tensor)

            patch_global_coordinates.append((x1,x2,y1,y2))
            patch_local_coordinates.append((x1_large-x1, x2_large-x2, y1_large-y1, y2_large-y2))
            
    return color_patches, grayscale_patches, patch_global_coordinates, patch_local_coordinates


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
        
    real_image = tf.image.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
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

