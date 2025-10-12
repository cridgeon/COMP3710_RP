import os
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import json

def list_ISIC_images(folder):
    return sorted([f.split(".")[0] for f in os.listdir(folder) if f.endswith(".jpg")])

def load_ISIC_tensors(data_dir: str, num_images: int, split: float, w: int = 3000, h: int = 2000) -> ((tf.Tensor, tf.Tensor), (tf.Tensor, tf.Tensor)):
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
    image_names = list(set(real_image_names).intersection(set([name for name in image_names])))
    metadata = metadata[metadata['image_name'].isin(image_names)]
    np.random.shuffle(image_names)

    X, Y = [], []

    TrainX, TrainY, TestX, TestY = [], [], [], []
    i = 0
    print("Loading data...")
    while True:
        if (i == num_images) or (i == len(image_names)):
            break
        name = image_names[i]
        i += 1

        if (i / num_images) * 100 % 10 == 0:
            print(f"{(i / num_images) * 100:.1f}% done")

        xb = tf.io.read_file(os.path.join(data_dir, "images/" + name + ".jpg"))
        x  = tf.image.decode_jpeg(xb, channels=1)
        x  = tf.image.resize(x, [w, h], method=tf.image.ResizeMethod.BILINEAR)
        x  = tf.cast(x, tf.float32) / 255.0 - 0.5

        y  = tf.cast(1 if metadata[metadata['image_name'] == name]["target"].values[0] == 1 else 0, tf.float32)

        X.append(x)
        Y.append(y)
    
    num_train = int(len(X) * split)
    data_size = len(X)
    for i in range(data_size):
        index = np.random.randint(0, len(X))
        if i < num_train:
            TrainX.append(X[index])
            TrainY.append(Y[index])
        else:
            TestX.append(X[index])
            TestY.append(Y[index])
        X.pop(index)
        Y.pop(index)
    
    return (tf.stack(TrainX, 0), tf.stack(TrainY, 0)), (tf.stack(TestX, 0), tf.stack(TestY, 0))

# if __name__ == "__main__":
#     data_dir = "data"
#     train_size = 3000
#     (X_train, Y_train), (X_test, Y_test) = load_ISIC_tensors(data_dir, train_size, 0.8, 480, 360)
#     print("stats: ")
#     print(X_train.shape, Y_train.shape)
#     print(X_test.shape, Y_test.shape)
#     print("# benign: ", tf.reduce_sum(1 - Y_train).numpy(), tf.reduce_sum(1 - Y_test).numpy())
#     print("# malignant: ", tf.reduce_sum(Y_train).numpy(), tf.reduce_sum(Y_test).numpy())