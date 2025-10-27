import os
import pathlib
from typing import Tuple
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from keras import layers
import keras
import json
import shutil

def list_ISIC_images(folder):
    return sorted([f.split(".")[0] for f in os.listdir(folder) if f.endswith(".jpg")])

def load_image_(name):
    xb = tf.io.read_file(os.path.join("recognition/siamese_kaggle_48020774/data/images/" + name + ".jpg"))
    x  = tf.image.decode_jpeg(xb, channels=3)
    x  = tf.image.resize(x, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
    # x  = tf.cast(x, tf.float32) / 255.0 - 0.5

    return x

def load_tensors_(img_meta, pos_meta, neg_meta, images, labels, positives, negatives):
    for i in range(img_meta.shape[0]):
        name = img_meta.iloc[i]['image_name']

        x = load_image_(name)
        x = PreprocessLayer()(tf.expand_dims(x, 0))[0]
        images.append(x)
        labels.append(pos_meta.iloc[i]['target'])
    for i in range(pos_meta.shape[0]):
        name = pos_meta.iloc[i]['image_name']

        x = load_image_(name)
        x = PreprocessLayer()(tf.expand_dims(x, 0))[0]
        positives.append(x)
    for i in range(neg_meta.shape[0]):
        name = neg_meta.iloc[i]['image_name']

        x = load_image_(name)
        x = PreprocessLayer()(tf.expand_dims(x, 0))[0]
        negatives.append(x)

def GenerateSet_(num, pos_meta: pd.DataFrame, neg_meta: pd.DataFrame, class_split: float):
    X, Y, P, N = [], [], [], []

    n_pos = class_split * num
    n_neg = (1 - class_split) * num
    num = int(n_pos) + int(n_neg)

    im = pd.concat([pos_meta.sample(n=int(n_pos)), neg_meta.sample(n=int(n_neg))]).sample(frac=1).reset_index(drop=True)
    pm = pos_meta.sample(n=num)
    nm = neg_meta.sample(n=num)

    load_tensors_(im, pm, nm, X, Y, P, N)

    X = tf.stack(X, 0)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    P = tf.stack(P, 0)
    N = tf.stack(N, 0)

    return X,Y,P,N

class NormalizationLayer(layers.Layer):
    """
    This layer is responsible for normalizing the input images
    before they are fed into the model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.means = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.stds = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def call(self, inputs):
        inputs = inputs / 255.0
        inputs = (inputs - self.means) / self.stds
        return inputs

class PreprocessLayer(layers.Layer):
    """
    This layer is responsible for preprocessing the input images
    before they are fed into the model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flip = layers.RandomFlip("horizontal_and_vertical")
        self.col = layers.RandomColorJitter([0,255], 0.2, 0.2, 0.1)
        self.norm = NormalizationLayer()

    def call(self, inputs):
        inputs = self.flip(inputs)
        inputs = self.col(inputs)
        inputs = self.norm(inputs)
        return inputs

def shuffle(dataset: tf.data.Dataset) -> tf.data.Dataset:
    num = dataset.cardinality().numpy()
    return dataset.shuffle(num)

def rotate(dataset: tf.data.Dataset, num):
    if (num > dataset.cardinality().numpy()):
        ret = dataset
        add, dataset = rotate(dataset, num - dataset.cardinality().numpy())
        ret = ret.concatenate(add)
        return ret, dataset
    ret = dataset.take(num)
    dataset = dataset.skip(num).concatenate(ret)
    return ret, dataset


class Dataset:
    def __init__(self, data_dir: str, max_images: int, class_split: float, train_split: int):
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

        if (class_split <= 0.02 or class_split >= 1):
            raise ValueError("class_split must be between 0.02 and 1")

        self.data_dir = data_dir
        self.class_split = class_split
        self.train_split = train_split
        self.process_layer = PreprocessLayer()
        self.norm_layer = NormalizationLayer()

        real_image_names  = list_ISIC_images(data_dir + "/images")
        metadata = pd.read_csv(data_dir + "/ISIC_2020_Training_GroundTruth.csv")
        image_names = metadata['image_name'].values

        # only image that exist in both the metadata and the images folder
        image_names = list(set(real_image_names).intersection(set([name for name in image_names])))
        metadata = metadata[metadata['image_name'].isin(image_names)]

        # Check if data_dir contains 'pos' and 'neg' directories
        has_pos_neg_dirs = (
            os.path.isdir(os.path.join(data_dir, "pos")) and
            os.path.isdir(os.path.join(data_dir, "neg"))
        )
        
        if not has_pos_neg_dirs:
            os.makedirs(os.path.join(data_dir, "pos"), exist_ok=True)
            os.makedirs(os.path.join(data_dir, "neg"), exist_ok=True)
            print(f"Created directories: {os.path.join(data_dir, 'pos')} and {os.path.join(data_dir, 'neg')}")
            print("Sorting images into 'pos' and 'neg' folders...")
            for i in range(metadata.shape[0]):
                # move data into pos/neg folders
                name = metadata.iloc[i]['image_name']
                label = metadata.iloc[i]['target']
                src_path = os.path.join(data_dir, "images", name + ".jpg")
                if label == 1:
                    dst_path = os.path.join(data_dir, "pos", name + ".jpg")
                else:
                    dst_path = os.path.join(data_dir, "neg", name + ".jpg")
                shutil.move(src_path, dst_path)
        
        

        pos_root = pathlib.Path(os.path.join(data_dir, "pos"))
        self.pos_ds = shuffle(tf.data.Dataset.list_files(str(pos_root/'*'))).take(max_images)
        # self.pos_ds = pos_ds.map(lambda x: self.load_from_path_(x, True))

        neg_root = pathlib.Path(os.path.join(data_dir, "neg"))
        self.neg_ds = shuffle(tf.data.Dataset.list_files(str(neg_root/'*'))).take(max_images)
        # self.neg_ds = neg_ds.map(lambda x: self.load_from_path_(x, False))

        n_pos = self.pos_ds.cardinality().numpy()
        n_neg = self.neg_ds.cardinality().numpy()
        # self.pos_ds = self.pos_ds.shuffle(n_pos)
        # self.neg_ds = self.neg_ds.shuffle(n_neg)
        n_train_pos = int(n_pos * train_split)
        n_train_neg = int(n_neg * train_split)

        self.train_pos = self.pos_ds.take(n_train_pos)
        # self.train_pos = shuffle(self.train_pos)
        self.train_neg = self.neg_ds.take(n_train_neg)
        # self.train_neg = shuffle(self.train_neg)
        self.test_pos = self.pos_ds.skip(n_train_pos)
        # self.test_pos = shuffle(self.test_pos)
        self.test_neg = self.neg_ds.skip(n_train_neg)
        # self.test_neg = shuffle(self.test_neg)

        print("Dataset initialized.")

    def GenerateTestSet(self, num):
        n_pos = int(self.class_split * num)
        n_neg = int((1 - self.class_split) * num)
        # Ensure total matches num exactly
        actual_num = n_pos + n_neg

        dataset, self.test_pos = rotate(self.test_pos, n_pos)
        d2, self.test_neg = rotate(self.test_neg, n_neg)
        dataset = dataset.concatenate(d2)
        dataset = shuffle(dataset)
        dataset = dataset.map(lambda x: self.load_from_path_(x))
        X, Y = dataset.batch(actual_num).take(1).get_single_element()

        P_dataset, self.pos_ds = rotate(self.pos_ds, actual_num)
        P_batch = P_dataset.map(lambda x: self.load_from_path_(x)).batch(actual_num).take(1).get_single_element()
        P = P_batch[0]  # Extract only the images, not labels
        
        N_dataset, self.neg_ds = rotate(self.neg_ds, actual_num)
        N_batch = N_dataset.map(lambda x: self.load_from_path_(x)).batch(actual_num).take(1).get_single_element()
        N = N_batch[0]  # Extract only the images, not labels

        return X, Y, P, N

    def GenerateTrainSet(self, num):
        n_pos = int(self.class_split * num)
        n_neg = int((1 - self.class_split) * num)
        # Ensure total matches num exactly
        actual_num = n_pos + n_neg

        dataset, self.train_pos = rotate(self.train_pos, n_pos)
        d2, self.train_neg = rotate(self.train_neg, n_neg)
        dataset = dataset.concatenate(d2)
        dataset = shuffle(dataset)
        dataset = dataset.map(lambda x: self.load_from_path_(x))
        X, Y = dataset.batch(actual_num).take(1).get_single_element()

        P_dataset, self.pos_ds = rotate(self.pos_ds, actual_num)
        P_batch = P_dataset.map(lambda x: self.load_from_path_(x)).batch(actual_num).take(1).get_single_element()
        P = P_batch[0]  # Extract only the images, not labels
        
        N_dataset, self.neg_ds = rotate(self.neg_ds, actual_num)
        N_batch = N_dataset.map(lambda x: self.load_from_path_(x)).batch(actual_num).take(1).get_single_element()
        N = N_batch[0]  # Extract only the images, not labels

        return X, Y, P, N
    
    def load_from_path_(self, path, validation=False):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
            img = tf.cast(img, tf.float32)
            if validation:
                img = self.norm_layer(tf.expand_dims(img, 0))[0]
            else:
                img = self.process_layer(tf.expand_dims(img, 0))[0]
            label = 1 if tf.strings.regex_full_match(path, ".*pos.*") else 0
            return img, label

    # def GenerateValidateSet(self, num):
    #     return Dataset.GenerateSet_(num, self.validate_pm, self.validate_nm, self.class_split)

import matplotlib.pyplot as plt

def plot_random_pn_samples(Pos, Neg, n):
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
    pos_idxs = np.random.choice(Pos.shape[0], n, replace=False)
    neg_idxs = np.random.choice(Neg.shape[0], n, replace=False)

    pos_images = Pos.numpy()[pos_idxs]
    neg_images = Neg.numpy()[neg_idxs]

    plt.figure(figsize=(16, 2))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        img = pos_images[i]
        img = (tf.tanh(img) + 1) / 2
        plt.imshow(img)
        plt.axis('off')
        plt.title("Malignant")

        plt.subplot(2, n, n + i + 1)
        img = neg_images[i]
        img = (tf.tanh(img) + 1) / 2
        plt.imshow(img)
        plt.axis('off')
        plt.title("Benign")
    plt.tight_layout()
    plt.show()


def plot_outputs(images, labels, outputs, save_path=None):
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
    n = len(images)
    l = int(np.sqrt(n))

    plt.figure(figsize=(16, 2))
    for i in range(n):
        plt.subplot(l + 1, l, i + 1)
        img = images[i]
        label = "malignant" if labels[i] == 1 else "benign"
        classification = "malignant" if outputs[i] else "benign"
        img = (tf.tanh(img) + 1) / 2
        plt.title(f"Label: {label} -- Class: {classification}")
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    config = json.load(open('recognition/siamese_kaggle_48020774/utility/config.json'))

    dataset = Dataset(
        config["data_dir"],
        config["max_images_in_ds"],
        config["class_split"],
        config["train_split"] 
    )

    X, Y, P, N = dataset.GenerateTestSet(10)

    # X, Y, P, N = dataset.GenerateTestSet(50)

    # plot_random_test_samples(dataset.test_images, 20)
    plot_random_pn_samples(P, N, 10)

    # X_train, Y_train, X_test, Y_test, Panchors, Nanchors = load_ISIC_tensors()
    # print("stats: ")
    # print(X_train.shape, Y_train.shape)
    # print(X_test.shape, Y_test.shape)
    # print("# benign: ", tf.reduce_sum(1 - Y_train).numpy(), tf.reduce_sum(1 - Y_test).numpy())
    # print("# malignant: ", tf.reduce_sum(Y_train).numpy(), tf.reduce_sum(Y_test).numpy())

    # plot_random_test_samples(X_test, Y_test, 20)