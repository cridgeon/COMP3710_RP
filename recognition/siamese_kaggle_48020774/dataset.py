import os
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import json

def list_ISIC_images(folder):
    return sorted([f.split(".")[0] for f in os.listdir(folder) if f.endswith(".jpeg")])

def load_ISIC_tensors(data_dir: str, train_size: int, test_size: int) -> ((tf.Tensor, tf.Tensor), (tf.Tensor, tf.Tensor)):
    """
    Loads all images from a folder into a single 4-D tenso.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing images named like <num1>_slice_<num2>.png.
    train_size : int
        Number of images to use for the training set.
    test_size : int
        Number of images to use for the test set.
    
    Returns
    -------
        A tuple of two tuples: ((X_train, Y_train), (X_test, Y_test))
    """
    real_image_names  = list_ISIC_images(data_dir + "/images")
    metadata = pd.read_csv(data_dir + "/ISIC_2020_Training_GroundTruth.csv")
    image_names = metadata['image_name'].values
    image_names = sorted(list(set(real_image_names).intersection(set([name for name in image_names]))))
    metadata = metadata[metadata['image_name'].isin(image_names)]

    TrainX, TrainY, TestX, TestY = [], [], [], []
    i = 0
    while True:
        if (i == train_size) or (i == len(image_names)):
            break
        name = image_names[i]

        xb = tf.io.read_file(os.path.join(data_dir, "images/" + name + ".jpeg"))
        x  = tf.image.decode_png(xb, channels=1)
        x  = tf.image.resize(x, [w, h], method=tf.image.ResizeMethod.BILINEAR)
        x  = tf.cast(x, tf.float32) / 255.0 - 0.5

        print("BENIGN" if metadata[metadata['image_name'] == name]["target"].values[0] == 0 else "MALIGNANT")

        y  = tf.cast(1 if metadata[metadata['image_name'] == name]["target"].values[0] == 1 else 0, tf.float32)

        TrainX.append(x)
        TrainY.append(y)
    #     xb = tf.convert_to_tensor(ds.pixel_array, dtype=tf.float32)
    #     x  = tf.image.decode_png(xb, channels=3)
    #     x  = tf.cast(x, tf.float32) / 255.0 - 0.5

    #     y  = tf.cast(ds., tf.float32)
    
    # X = tf.stack(Xs, 0)
    # Y = tf.stack(Ys, 0)
    # return X, Y

if __name__ == "__main__":
    data_dir = "data"
    train_size = 100
    test_size = 20
    (X_train, Y_train), (X_test, Y_test) = load_ISIC_tensors(data_dir, train_size, test_size)