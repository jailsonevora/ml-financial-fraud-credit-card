import time
import tensorflow as tf
from tensorflow.python.client import device_lib

# Make sure GPU is available
def check_gpu_availability():
    gpus = tf.config.list_physical_devices('GPU')
    print("Detected GPUs:", gpus)

# Use GPU for this test
def get_available_gpus():
    with tf.device('/GPU:0'):
        print("\nRunning matrix multiplication on GPU...")
        start = time.time()
        a = tf.random.normal([3000, 3000])
        b = tf.random.normal([3000, 3000])
        c = tf.matmul(a, b)
        _ = c.numpy()  # Force computation
        print("Time taken (GPU):", time.time() - start, "seconds")

# List available devices
def list_available_devices():
    print("Available devices:")
    for device in tf.config.list_physical_devices():
        print(device)

# Check for GPU
def check_gpu_availability():    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available: ", gpus)
    else:
        print("No GPU found")
    print(device_lib.list_local_devices())

def get_num_available_gpus():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))