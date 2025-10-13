import os
from typing import Tuple
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import json

def list_ISIC_images(folder):
    return sorted([f.split(".")[0] for f in os.listdir(folder) if f.endswith(".jpg")])

def load_ISIC_tensors() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
        A tuple of tensors: X_train, Y_train, X_test, Y_test, positive_anchors, negative_anchors
    """

    config = json.load(open("utility/config.json", "r"))
    data_dir = config["data_dir"]
    class_split = config["class_split"]
    max_images = config["max_images"]
    anchor_count = config["anchor_count"]
    train_split = config["train_split"]

    real_image_names  = list_ISIC_images(data_dir + "/images")
    metadata = pd.read_csv(data_dir + "/ISIC_2020_Training_GroundTruth.csv")
    image_names = metadata['image_name'].values

    image_names = list(set(real_image_names).intersection(set([name for name in image_names])))
    metadata = metadata[metadata['image_name'].isin(image_names)]

    #get anchor positive and negatives
    positive_names = metadata[metadata['target'] == 1]['image_name'].values
    negative_names = metadata[metadata['target'] == 0]['image_name'].values

    positives, negatives = [], []

    for i in range(anchor_count):
        index1 = np.random.randint(0, len(positive_names))
        index2 = np.random.randint(0, len(negative_names))

        xb = tf.io.read_file(os.path.join(data_dir, "images/" + positive_names[index1] + ".jpg"))
        x  = tf.image.decode_jpeg(xb, channels=3)
        x  = tf.image.resize(x, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
        x  = tf.cast(x, tf.float32) / 255.0 - 0.5

        positives.append(x)

        xb = tf.io.read_file(os.path.join(data_dir, "images/" + negative_names[index2] + ".jpg"))
        x  = tf.image.decode_jpeg(xb, channels=3)
        x  = tf.image.resize(x, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
        x  = tf.cast(x, tf.float32) / 255.0 - 0.5

        negatives.append(x)

        # Remove the used names from the arrays to avoid duplicates
        positive_names = np.delete(positive_names, index1)
        negative_names = np.delete(negative_names, index2)

    #ensure we dont use anchors for training        
    metadata = metadata[metadata['image_name'].isin(positive_names) | metadata['image_name'].isin(negative_names)]
    metadata = metadata.sample(frac=1)

    # select the dataset
    n_malignant = len(metadata[metadata['target'] == 1])
    real_class_split = n_malignant / metadata.shape[0]
    dataset = pd.DataFrame()
    total_images = (1 / class_split) * n_malignant

    if max_images < total_images:
        dataset = metadata[metadata['target'] == 1].sample(n=int(max_images * class_split))
        dataset = pd.concat([dataset, metadata[metadata['target'] == 0].sample(n=int(max_images * (1 - class_split)))])
    else:
        if class_split > real_class_split:
            n_benign = total_images - n_malignant
            dataset = pd.concat([metadata[metadata['target'] == 1], metadata[metadata['target'] == 0].sample(n=int(n_benign))])
        elif class_split < real_class_split:
            n_benign = len(metadata[metadata['target'] == 0])
            n_malignant = (1 / class_split - 1) * n_benign
            dataset = pd.concat([metadata[metadata['target'] == 0], metadata[metadata['target'] == 1].sample(n=int(n_malignant))])
        else:
            dataset = metadata

    dataset = dataset.sample(frac=1)

    print(f"Using {len(dataset)} images with {len(dataset[dataset['target'] == 1])} malignant and {len(dataset[dataset['target'] == 0])} benign")
    # load the data
    X, Y = [], []
    TrainX, TrainY, TestX, TestY = [], [], [], []
    print("Loading data...")
    for i in range(len(dataset)):
        name = dataset.iloc[i]['image_name']

        if (i / len(dataset)) * 100 % 10 == 0:
            print(f"{(i / len(dataset)) * 100:.1f}% done")

        xb = tf.io.read_file(os.path.join(data_dir, "images/" + name + ".jpg"))
        x  = tf.image.decode_jpeg(xb, channels=3)
        x  = tf.image.resize(x, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
        x  = tf.cast(x, tf.float32) / 255.0 - 0.5

        y  = tf.cast(1 if dataset.iloc[i]['target'] == 1 else 0, tf.float32)

        X.append(x)
        Y.append(y)
    
    num_train = int(len(X) * train_split)
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

    # Average all chosen tensors and normalize them
    def normalize_tensor(tensor, mean, std):
        return (tensor - mean) / (std + 1e-7)

    TrainX = tf.stack(TrainX, 0)
    TrainY = tf.stack(TrainY, 0)
    TestX = tf.stack(TestX, 0)
    TestY = tf.stack(TestY, 0)
    positives = tf.stack(positives, 0)
    negatives = tf.stack(negatives, 0)

    # Compute the mean tensor of all images (train + test + anchors)
    all_images = tf.concat([TrainX, TestX, positives, negatives], axis=0)
    mean_image = tf.reduce_mean(all_images, axis=0)
    std_image = tf.math.reduce_std(all_images, axis=0)

    # Subtract mean and normalize each set
    TrainX = normalize_tensor(TrainX, mean_image, std_image)
    TestX = normalize_tensor(TestX, mean_image, std_image)
    positives = normalize_tensor(positives, mean_image, std_image)
    negatives = normalize_tensor(negatives, mean_image, std_image)

    return tf.stack(TrainX, 0), \
           tf.stack(TrainY, 0), \
           tf.stack(TestX, 0), \
           tf.stack(TestY, 0), \
           tf.stack(positives, 0), \
           tf.stack(negatives, 0)

import matplotlib.pyplot as plt

def plot_random_test_samples(X_test, Y_test, num_samples=8):
    """
    Plots a random sample of test images with their labels.

    Parameters
    ----------
    X_test : tf.Tensor
        Test images tensor.
    Y_test : tf.Tensor
        Test labels tensor.
    num_samples : int
        Number of samples to plot.
    """
    idxs = np.random.choice(X_test.shape[0], num_samples, replace=False)
    images = X_test.numpy()[idxs]
    labels = Y_test.numpy()[idxs]

    plt.figure(figsize=(16, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        img = images[i]
        plt.imshow(img)
        plt.axis('off')
        plt.title("Malignant" if labels[i] == 1 else "Benign")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = "data"
    train_size = 3000
    X_train, Y_train, X_test, Y_test, Panchors, Nanchors = load_ISIC_tensors()
    print("stats: ")
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    print("# benign: ", tf.reduce_sum(1 - Y_train).numpy(), tf.reduce_sum(1 - Y_test).numpy())
    print("# malignant: ", tf.reduce_sum(Y_train).numpy(), tf.reduce_sum(Y_test).numpy())

    plot_random_test_samples(X_test, Y_test, 20)