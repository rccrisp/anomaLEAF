import numpy as np
import cv2
from skimage import morphology

def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute anomaly color heatmap.

    Args:
        anomaly_map (np.ndarray): Final anomaly map computed by the distance metric.
        normalize (bool, optional): Bool to normalize the anomaly map prior to applying
            the color map. Defaults to True.

    Returns:
        np.ndarray: [description]
    """
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
    anomaly_map = anomaly_map * 255
    anomaly_map = anomaly_map.astype(np.uint8)

    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB)
    return anomaly_map

def superimpose_anomaly_map(
    anomaly_map: np.ndarray, image: np.ndarray, alpha: float = 0.4, gamma: int = 0, normalize: bool = False
) -> np.ndarray:
    """Superimpose anomaly map on top of in the input image.

    Args:
        anomaly_map (np.ndarray): Anomaly map
        image (np.ndarray): Input image
        alpha (float, optional): Weight to overlay anomaly map
            on the input image. Defaults to 0.4.
        gamma (int, optional): Value to add to the blended image
            to smooth the processing. Defaults to 0. Overall,
            the formula to compute the blended image is
            I' = (alpha*I1 + (1-alpha)*I2) + gamma
        normalize: whether or not the anomaly maps should
            be normalized to image min-max


    Returns:
        np.ndarray: Image with anomaly map superimposed on top of it.
    """

    anomaly_map = anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=normalize)
    image = image.squeeze()  
    image = image.astype(np.uint8) 
    superimposed_map = cv2.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)
    return superimposed_map

def compute_mask(anomaly_map: np.ndarray, threshold: float, kernel_size: int = 4) -> np.ndarray:
    """Compute anomaly mask via thresholding the predicted anomaly map.

    Args:
        anomaly_map (np.ndarray): Anomaly map predicted via the model
        threshold (float): Value to threshold anomaly scores into 0-1 range.
        kernel_size (int): Value to apply morphological operations to the predicted mask. Defaults to 4.

    Returns:
        Predicted anomaly mask
    """

    anomaly_map = anomaly_map.squeeze()
    mask: np.ndarray = np.zeros_like(anomaly_map).astype(np.uint8)
    mask[anomaly_map > threshold] = 1

    kernel = morphology.disk(kernel_size)
    mask = morphology.opening(mask, kernel)

    mask *= 255

    return mask
