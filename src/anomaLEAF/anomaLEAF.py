import numpy as np
from typing import List, Tuple
import tensorflow as tf

class anomaLEAF:

    def __init__(self, 
                 generator: tf.keras.Model) -> None:
        
        self.generator = generator
        
    def reconstruct(self,
                    original_image: np.ndarray,
                    inpt: List[tf.Tensor], 
                    patch_global_coordinates: List[Tuple[int, int, int, int]],
                    patch_local_coordinates: List[Tuple[int, int, int, int]])-> Tuple[np.ndarray, np.ndarray]:
        """
        
        """

        # reconstruct the image using the trained generator
        reconstructed_image = np.zeros_like(original_image, dtype=float)

        for patch, global_coordinates, local_coordinates in zip(inpt, patch_global_coordinates, patch_local_coordinates):
            reconstructed_patch = self.generator.predict(np.expand_dims(patch, axis=0))
            x1_global, x2_global, y1_global, y2_global = global_coordinates
            x1_local, x2_local, y1_local, y2_local = local_coordinates

            extracted = reconstructed_patch[0][x1_local:x2_local, y1_local:y2_local]
            reconstructed_image[x1_global:x2_global, y1_global:y2_global] = extracted

        return original_image/255.0, reconstructed_image
    

    



    