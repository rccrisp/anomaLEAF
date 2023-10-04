import numpy as np
from typing import List, Tuple
import tensorflow as tf

class anomaLEAF:

    def __init__(self, 
                 generator: tf.keras.Model) -> None:
        
        self.generator = generator

    def reconstruct_patch(self,
                          x: tf.Tensor,
                          verbosity: int = 0
                          ) -> np.ndarray:

        y = self.generator.predict(np.expand_dims(x, axis=0),verbose=verbosity)

        return y
        
    def reconstruct(self,
                    original_image: np.ndarray,
                    inpt: List[tf.Tensor], 
                    patch_global_coordinates: List[Tuple[int, int, int, int]],
                    patch_local_coordinates: List[Tuple[int, int, int, int]],
                    verbosity: int = 0)-> Tuple[np.ndarray, np.ndarray]:
        """
        Performs colour reconstruction on grayscale patches

        Returns: the original image and the reconstructed image
        
        """

        reconstructed_image = np.zeros_like(original_image, dtype=float)

        for patch, global_coordinates, local_coordinates in zip(inpt, patch_global_coordinates, patch_local_coordinates):
            
            reconstructed_patch = self.reconstruct_patch(patch,verbosity=verbosity)
            
            # extract global coordinates; the position of the patch in the original image
            x1_global, x2_global, y1_global, y2_global = global_coordinates
            
            # extract the local coordinates; the position of the grayscale area in the patch
            x1_local, x2_local, y1_local, y2_local = local_coordinates

            # paste the reconstructed patch into the reconstructed image
            reconstructed_image[x1_global:x2_global, y1_global:y2_global] = reconstructed_patch[0][x1_local:x2_local, y1_local:y2_local]

        return original_image/255.0, reconstructed_image

    def anomaly_detection_loss(self, 
                                loss: np.ndarray,
                                threshold: float, 
                                scaling_factor: float, 
                                pixel_step: float)->Tuple[float, np.ndarray]:
        """
        A non-linear loss function that heavily penalises large pixel reconstruction loss. 
        The loss function is

                y = x*a^(b*(x-t))

                where

                y = anomaly score (array): metric for anomaly classification
                x = reconstruction loss (array): the difference between the real and reconstructed image
                a = scaling factor (float): how severely loss is penalised
                b = pixel step (float): how severely loss variation is penalised
                t = threshold (float):  reconstruction loss below this value is assumed to be normal and is minimised
        """
    
        anomaly_score = loss*np.power(scaling_factor, pixel_step*(loss - threshold))

        return np.sum(anomaly_score), anomaly_score
    



    