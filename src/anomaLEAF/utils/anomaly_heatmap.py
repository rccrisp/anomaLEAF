import tensorflow as tf
from skimage.color import deltaE_ciede2000

def CIEDE2000(img_a: tf.Tensor, img_b: tf.Tensor) -> tf.Tensor:
    color_diff = tf.py_function(deltaE_ciede2000, inp=[img_a, img_b], Tout=tf.float32)
    return color_diff

def Euclidean(img_a: tf.Tensor, img_b: tf.Tensor) -> tf.Tensor:
    # Calculate the squared Euclidean distance between corresponding pixels
    squared_distance = tf.reduce_sum(tf.square(img_a - img_b), axis=-1)
    
    # Take the square root to get the Euclidean distance
    euclidean_distance = tf.sqrt(squared_distance)
    
    return euclidean_distance

def point_score(heatmap: tf.Tensor) -> tf.Tensor:
    return tf.reduce(heatmap)