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
from config import Config

def list_ISIC_images(folder):
    return sorted([f.split(".")[0] for f in os.listdir(folder) if f.endswith(".jpg")])

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
        # print("Normalised input shape:", inputs.shape)
        return inputs

class PreprocessLayer(layers.Layer):
    """
    This layer is responsible for preprocessing the input images
    before they are fed into the model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flip = layers.RandomFlip("horizontal_and_vertical")
        # self.col = layers.RandomColorJitter([0,255], 0.2, 0.2, 0.1)
        self.norm = NormalizationLayer()

    def call(self, inputs):
        inputs = self.flip(inputs)
        # inputs = self.col(inputs)
        # print("Preprocessed input shape:", inputs.shape)
        inputs = self.norm(inputs)
        return inputs
    
class TrainTestPreprocessor(layers.Layer):
    """
    This layer is responsible for preprocessing the input images
    before they are fed into the model.
    During training, it applies data augmentation.
    During testing, it only normalizes the images.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocess = PreprocessLayer()
        self.norm = NormalizationLayer()

    def call(self, inputs, training=None):
        if training:
            inputs = self.preprocess(inputs)
        inputs = self.norm(inputs)
        return inputs

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

        if (class_split <= 0.02 or class_split >= 0.4):
            raise ValueError("class_split must be between 0.02 and 0.4")

        self.data_dir = data_dir
        self.class_split = class_split
        self.train_split = train_split

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
            
            real_image_names  = list_ISIC_images(data_dir + "/images")
            metadata = pd.read_csv(data_dir + "/ISIC_2020_Training_GroundTruth.csv")
            image_names = metadata['image_name'].values

            # only image that exist in both the metadata and the images folder
            image_names = list(set(real_image_names).intersection(set([name for name in image_names])))
            metadata = metadata[metadata['image_name'].isin(image_names)]
            
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

        def pick_triplet_paths(a_y, p_n):
            # a_y:  (a_path, y) ; p_n: (p_path, n_path)
            a_path, y = a_y
            p_path, n_path = p_n
            y = tf.cast(y, tf.int32)
            pos_path = tf.where(tf.equal(y, 1), p_path, n_path)
            neg_path = tf.where(tf.equal(y, 1), n_path, p_path)
            return (a_path, pos_path, neg_path, y)

        def load_triplet(a_path, p_path, n_path, y):
            def _load(p):
                img = tf.io.read_file(p)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
                return tf.cast(img, tf.float32)
            a = _load(a_path); p = _load(p_path); n = _load(n_path)
            # Keras expects (inputs, label). If your model takes (a,p,n) as inputs:
            y = tf.cast(y, tf.int32)
            return ((a, y, p, n), y)

        # -------------- train/val split with your hard counts ----------------
        pos_files = list_ISIC_images(f"{data_dir}/pos")
        neg_files = list_ISIC_images(f"{data_dir}/neg")
        for (i, f) in enumerate(pos_files):
            pos_files[i] = f"{data_dir}/pos/{f}.jpg"
        for (i, f) in enumerate(neg_files):
            neg_files[i] = f"{data_dir}/neg/{f}.jpg"

        n_pos = len(pos_files)
        n_neg = len(neg_files)
        assert n_pos > 0 and n_neg > 0, "No files found; check data_dir and class folders"

        k_pos = max(1, int(n_pos * train_split))
        k_neg = max(1, int(n_neg * train_split))
        
        # -------------- build labeled path datasets ----------------
        pos_paths = tf.data.Dataset.from_tensor_slices(pos_files)
        neg_paths = tf.data.Dataset.from_tensor_slices(neg_files)

        # attach labels (1 for pos, 0 for neg) — specify num_parallel_calls properly
        pos_labeled = pos_paths.map(lambda p: (p, tf.constant(1, tf.int32)), num_parallel_calls=tf.data.AUTOTUNE)
        neg_labeled = neg_paths.map(lambda p: (p, tf.constant(0, tf.int32)), num_parallel_calls=tf.data.AUTOTUNE)

        # validation anchors (finite; no repeat)
        pos_val_anchors = pos_labeled.take(k_pos).repeat()
        neg_val_anchors = neg_labeled.take(k_neg).repeat()

        # training anchors (infinite; repeat + shuffle)
        pos_train_stream = pos_labeled.skip(k_pos).repeat().shuffle(8192, reshuffle_each_iteration=True)
        neg_train_stream = neg_labeled.skip(k_neg).repeat().shuffle(8192, reshuffle_each_iteration=True)

        # partner pools (images only, infinite). We keep them separate for train/val.
        pos_pool_train = pos_train_stream.map(lambda p, y: p, num_parallel_calls=tf.data.AUTOTUNE)
        neg_pool_train = neg_train_stream.map(lambda p, y: p, num_parallel_calls=tf.data.AUTOTUNE)

        pos_pool_val = pos_val_anchors.map(lambda p, y: p, num_parallel_calls=tf.data.AUTOTUNE).repeat()
        neg_pool_val = neg_val_anchors.map(lambda p, y: p, num_parallel_calls=tf.data.AUTOTUNE).repeat()

        # -------------- TRAIN dataset (infinite) ----------------
        train_anchors = tf.data.Dataset.sample_from_datasets(
            [pos_train_stream, neg_train_stream],
            weights=[class_split, 1.0 - class_split],
            stop_on_empty_dataset=False,
            seed=42,
        )

        train_zipped = tf.data.Dataset.zip((train_anchors,
                                            tf.data.Dataset.zip((pos_pool_train, neg_pool_train))))

        train_triplet_paths = train_zipped.map(pick_triplet_paths, num_parallel_calls=tf.data.AUTOTUNE)

        self.train_ds = (train_triplet_paths
            .map(load_triplet, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(Config.getInstance()['batch_size'], drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))

        # -------------- VALIDATION dataset (finite anchors) ----------------
        val_anchors = tf.data.Dataset.sample_from_datasets(
            [pos_val_anchors, neg_val_anchors],
            weights=[class_split, 1.0 - class_split],
            stop_on_empty_dataset=False,
            seed=12345,
        )

        val_zipped = tf.data.Dataset.zip((val_anchors,
                                        tf.data.Dataset.zip((pos_pool_val, neg_pool_val))))

        val_triplet_paths = val_zipped.map(pick_triplet_paths, num_parallel_calls=tf.data.AUTOTUNE)

        self.test_ds = (val_triplet_paths
            .map(load_triplet, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(Config.getInstance()['batch_size'], drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))

        # (optional) tf.data options for throughput
        opts = tf.data.Options()
        opts.experimental_deterministic = False
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.apply_default_optimizations = True
        self.train_ds = self.train_ds.with_options(opts)
        self.test_ds  = self.test_ds.with_options(opts)
        
        # print("loading dataset from directories...")
        # dataset : tf.data.Dataset = keras.utils.image_dataset_from_directory(
        #     data_dir,
        #     labels="inferred",
        #     label_mode="binary",
        #     class_names=[
        #         "neg",
        #         "pos"
        #     ],
        #     color_mode="rgb",
        #     image_size=(256, 256),
        #     batch_size=None,
        #     interpolation="bilinear",
        # )
        # print("Dataset loaded. Splitting and generating desired class balance...")
        # pos = dataset.filter(lambda x, y: tf.squeeze(tf.equal(y, 1)))
        # neg = dataset.filter(lambda x, y: tf.squeeze(tf.equal(y, 0)))
        
        # # achieve desired class split
        # n_pos = 32542
        # n_neg = 584
        
        # pos_test = pos.take(int(n_pos * train_split))
        # pos_train = pos.skip(int(n_pos * train_split)).repeat() # train can have repeats since we are going to augment
        # neg_test = neg.take(int(n_neg * train_split))
        # neg_train = neg.skip(int(n_neg * train_split)).repeat()
        
        # # combine and preprocess
        # # class-balanced anchors (image,label), augmented + normalized
        # train_ds = tf.data.Dataset.sample_from_datasets(
        #     [pos_train, neg_train],
        #     weights=[class_split, 1.0 - class_split],
        #     stop_on_empty_dataset=False,  # keep going forever
        #     seed=np.random.randint(0, 1_000_000),
        # ).map(lambda x, y: (x, tf.cast(y, tf.int32)),
        #     num_parallel_calls=tf.data.AUTOTUNE)

        # # independent positive/negative pools (images only), also augmented+normalized
        # pos_pool = pos_train.map(lambda x, y: x,
        #                         num_parallel_calls=tf.data.AUTOTUNE)
        # neg_pool = neg_train.map(lambda x, y: x,
        #                         num_parallel_calls=tf.data.AUTOTUNE)

        # # zip once and pick p/n using the SAME y that came with the anchor
        # zipped = tf.data.Dataset.zip((train_ds, tf.data.Dataset.zip((pos_pool, neg_pool))))
        # # element: ((a, y), (p, n))

        # self.train_ds = (zipped
        #     .map(pick_triplet, num_parallel_calls=tf.data.AUTOTUNE)
        #     .batch(Config.getInstance()['batch_size'])
        #     .prefetch(tf.data.AUTOTUNE))
        
        # test_ds = tf.data.Dataset.sample_from_datasets(
        #     [pos_test, neg_test],
        #     weights=[class_split, 1.0 - class_split],
        #     stop_on_empty_dataset=True, 
        #     seed=np.random.randint(0, 1_000_000),
        # ).map(lambda x, y: (x, tf.cast(y, tf.int32)),
        #     num_parallel_calls=tf.data.AUTOTUNE)

        # # independent positive/negative pools (images only), also augmented+normalized
        # pos_pool = pos_test.map(lambda x, y: x,
        #                         num_parallel_calls=tf.data.AUTOTUNE)
        # neg_pool = neg_test.map(lambda x, y: x,
        #                         num_parallel_calls=tf.data.AUTOTUNE)

        # # zip once and pick p/n using the SAME y that came with the anchor
        # zipped = tf.data.Dataset.zip((test_ds, tf.data.Dataset.zip((pos_pool, neg_pool))))
        # # element: ((a, y), (p, n))

        # self.test_ds = (zipped
        #     .map(pick_triplet, num_parallel_calls=tf.data.AUTOTUNE)
        #     .batch(Config.getInstance()['batch_size'])
        #     .prefetch(tf.data.AUTOTUNE))


        print("Dataset initialized.")
        train_size = self.train_ds.cardinality().numpy()
        train_size = train_size if train_size >= 0 else ("UNKNOWN" if train_size == -2 else "INFINITE")
        test_size = self.test_ds.cardinality().numpy()
        test_size = test_size if test_size >= 0 else ("UNKNOWN" if test_size == -2 else "INFINITE")
        print(f"Training set size: {train_size} samples")
        print(f"Validation set size: {test_size} samples")

    def GenerateTestSet(self):
        return self.test_ds

    def GenerateTrainSet(self):
        return self.train_ds

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


    config  = Config.getInstance()

    dataset = Dataset(
        config["data_dir"],
        config["max_images_in_ds"],
        config["class_split"],
        config["train_split"] 
    )

    train_ds = dataset.GenerateTrainSet()
    test_ds = dataset.GenerateTestSet()
    
    print("Dataset test complete")

    # X, Y, P, N = dataset.GenerateTestSet(50)

    # plot_random_test_samples(dataset.test_images, 20)
    # plot_random_pn_samples(P, N, 10)

    # X_train, Y_train, X_test, Y_test, Panchors, Nanchors = load_ISIC_tensors()
    # print("stats: ")
    # print(X_train.shape, Y_train.shape)
    # print(X_test.shape, Y_test.shape)
    # print("# benign: ", tf.reduce_sum(1 - Y_train).numpy(), tf.reduce_sum(1 - Y_test).numpy())
    # print("# malignant: ", tf.reduce_sum(Y_train).numpy(), tf.reduce_sum(Y_test).numpy())

    # plot_random_test_samples(X_test, Y_test, 20)