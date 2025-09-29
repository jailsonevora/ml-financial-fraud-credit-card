import tensorflow as tf
import time
from tensorflow.python.client import device_lib


# Use CPU for this test
def get_available_gpus():
    with tf.device('/CPU:0'):
        print("\nRunning matrix multiplication on CPU...")
        start = time.time()
        a = tf.random.normal([3000, 3000])
        b = tf.random.normal([3000, 3000])
        c = tf.matmul(a, b)
        _ = c.numpy()  # Force computation
        print("Time taken (CPU):", time.time() - start, "seconds")

