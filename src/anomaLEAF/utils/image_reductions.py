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
                        patch_size: Tuple[int,int],
                        tile_size: Tuple[int, int]) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    """

    max_width = image.shape[0]
    max_height = image.shape[1]

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR) # convert back to 3 channel

    patch_rows = grayscale_image.shape[0] // patch_size[0] + (1 if image.shape[0] % patch_size[0] != 0 else 0)
    patch_cols = grayscale_image.shape[1] // patch_size[1] + (1 if image.shape[1] % patch_size[1] != 0 else 0)

    grayscale_patches = []
    color_patches = []
    patch_global_coordinates = []
    patch_local_coordinates = []

    x_diff = tile_size[0]-patch_size[0]
    y_diff = tile_size[1]-patch_size[1]

    tensor_size = tile_size

    for i in range(patch_rows):
        for j in range(patch_cols):
            # Calculate the coordinates for the current grid square
            x1, y1 = i * patch_size[0], j * patch_size[1]
            x2, y2 = (i + 1) * patch_size[0], (j + 1) * patch_size[1]

            # Create grayscale patch
            modified_image = image.copy()
            modified_image[x1:x2, y1:y2] = grayscale_image[x1:x2,y1:y2]
            
            # if the tile is larger than the original image
            if max_width < tile_size[0]:
                tensor_size[0] = max_width
                x1_offset = x1
                x2_offset = max_width-x2
            else:
                if x1-x_diff//2 < 0:
                    x1_offset = x1
                    x2_offset = x_diff-x1_offset
                elif max_width < x2+x_diff//2:
                    x2_offset = max_width-x2
                    x1_offset = x_diff-x2_offset
                else:
                    x1_offset = x_diff//2
                    x2_offset = x_diff//2
            
            if max_height < tile_size[1]:
                tensor_size[1] = max_height
                y1_offset = y1
                y2_offset = max_height-y2
            else:
                if y1-y_diff//2 < 0:
                    y1_offset = y1
                    y2_offset = y_diff - y1_offset
                elif max_height < y2+y_diff//2:
                    y2_offset = max_height-y2
                    y1_offset = y_diff-y2_offset
                else:
                    y1_offset = y_diff//2
                    y2_offset = y_diff//2
            

            
            x1_large, x2_large, y1_large, y2_large = x1-x1_offset, x2+x2_offset, y1-y1_offset, y2+y2_offset

            rgb_patch = image[x1_large:x2_large, y1_large:y2_large]
            rgb_patch_tensor = image_to_tensor(rgb_patch, tensor_size)
            color_patches.append(rgb_patch_tensor)
            
            grayscale_patch_with_border = modified_image[x1_large:x2_large, y1_large:y2_large]
            grayscale_patch_with_border_tensor = image_to_tensor(grayscale_patch_with_border, tensor_size)
            grayscale_patches.append(grayscale_patch_with_border_tensor)

            # coordinates of grayscale patch in global image
            patch_global_coordinates.append((x1,x2,y1,y2))
            
            # coordinates of grayscale patch in bordered tile
            patch_local_coordinates.append((x1_offset, patch_size[0]+x1_offset, y1_offset, patch_size[1]+y1_offset))
       
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

