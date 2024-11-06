import numpy as np
import tensorflow as tf


def load_mnist():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    # Convert to binary values (1 for dark pixels, -1 for light pixels)
    x_train = np.where(x_train > 127, 1, -1)
    x_test = np.where(x_test > 127, 1, -1)
    return x_train, x_test
