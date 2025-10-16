import os
from typing import Tuple
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from keras import layers
import json

def list_ISIC_images(folder):
    return sorted([f.split(".")[0] for f in os.listdir(folder) if f.endswith(".jpg")])

class PreprocessLayer(layers.Layer):
    """
    This layer is responsible for preprocessing the input images
    before they are fed into the model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flip = layers.RandomFlip("horizontal_and_vertical")

    def call(self, inputs):
        inputs = self.flip(inputs)
        return inputs

class Dataset:
    def __init__(self, data_dir: str, class_split: float, train_size: int, test_size: int, validate_size: int = 0):
        if (class_split <= 0.02 or class_split >= 1):
            raise ValueError("class_split must be between 0.02 and 1")

        self.data_dir = data_dir
        self.class_split = class_split
        self.train_size = train_size
        self.test_size = test_size
        self.validate_size = validate_size

        self.train_images = []
        self.train_labels = []
        self.train_positives = []
        self.train_negatives = []

        self.test_images = []
        self.test_labels = []
        self.test_positives = []
        self.test_negatives = []

        self.validate_images = []
        self.validate_labels = []
        self.validate_positives = []
        self.validate_negatives = []
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

        real_image_names  = list_ISIC_images(data_dir + "/images")
        metadata = pd.read_csv(data_dir + "/ISIC_2020_Training_GroundTruth.csv")
        image_names = metadata['image_name'].values

        # only image that exist in both the metadata and the images folder
        image_names = list(set(real_image_names).intersection(set([name for name in image_names])))
        metadata = metadata[metadata['image_name'].isin(image_names)]
        positives = metadata[metadata['target'] == 1].sample(frac=1)
        negatives = metadata[metadata['target'] == 0].sample(frac=1)

        # select the dataset
        n_malignant = positives.shape[0]
        real_class_split = n_malignant / metadata.shape[0]
        dataset = pd.DataFrame()
        total_images = ((1 / class_split) * n_malignant) // 1 + 1
        negatives = negatives.sample(n=int(total_images - n_malignant))
        desired_images = self.train_size + self.test_size + self.validate_size

        if desired_images > total_images:
            ratio =  total_images / desired_images
            print("Total # of desired images is greater than # of images on disk! unable to load all requested data.")
            train_size = (train_size * ratio) // 1
            test_size = (test_size * ratio) // 1
            validate_size = (validate_size * ratio) // 1
            print("New image split:")
            print("  Train size   : ", train_size)
            print("  Test size    : ", test_size)
            print("  Validate size: ", validate_size)

        train_positives_metadata = positives.sample(n=int(self.train_size * self.class_split))
        train_negatives_metadata = negatives.sample(n=int(self.train_size * (1 - self.class_split)))
        positives = positives.drop(train_positives_metadata.index)
        negatives = negatives.drop(train_negatives_metadata.index)
        test_positives_metadata = positives.sample(n=int(self.test_size * self.class_split))
        test_negatives_metadata = negatives.sample(n=int(self.test_size * (1 - self.class_split)))
        positives = positives.drop(test_positives_metadata.index)
        negatives = negatives.drop(test_negatives_metadata.index)
        validate_positives_metadata = positives.sample(n=int(self.validate_size * self.class_split))
        validate_negatives_metadata = negatives.sample(n=int(self.validate_size * (1 - self.class_split)))

        def load_image(name):
            xb = tf.io.read_file(os.path.join(data_dir, "images/" + name + ".jpg"))
            x  = tf.image.decode_jpeg(xb, channels=3)
            x  = tf.image.resize(x, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
            x  = tf.cast(x, tf.float32) / 255.0 - 0.5

            return x

        def load_tensors(pos_meta, neg_meta, images, labels, positives, negatives):
            for i in range(pos_meta.shape[0]):
                name = pos_meta.iloc[i]['image_name']

                x = load_image(name)
                images.append(x)
                positives.append(x)
                labels.append(pos_meta.iloc[i]['target'])
            for i in range(neg_meta.shape[0]):
                name = neg_meta.iloc[i]['image_name']

                x = load_image(name)
                images.append(x)
                negatives.append(x)
                labels.append(neg_meta.iloc[i]['target'])

            images = tf.stack(images, 0)
            positives = tf.stack(positives, 0)
            negatives = tf.stack(negatives, 0)

        print("Loading test images...")
        load_tensors(test_positives_metadata, test_negatives_metadata, self.test_images, self.test_labels, self.test_positives, self.test_negatives)
        print("Loading train images...")
        load_tensors(train_positives_metadata, train_negatives_metadata, self.train_images, self.train_labels, self.train_positives, self.train_negatives)
        print("Loading validate images...")
        load_tensors(validate_positives_metadata, validate_negatives_metadata, self.validate_images, self.validate_labels, self.validate_positives, self.validate_negatives)

        total_pos = \
            len(self.train_positives) + \
            len(self.test_positives) + \
            len(self.validate_positives)
        total_neg = \
            len(self.train_negatives) + \
            len(self.test_negatives) + \
            len(self.validate_negatives)
        print(f"Using {train_size + test_size + validate_size} images with {total_pos} malignant and {total_neg} benign")

        # Average all chosen tensors and normalize them
        def normalize_tensor(tensor, mean, std):
            return (tensor - mean) / (std + 1e-7)


        # Compute the mean tensor of all images (train + test + anchors)
        all_images = tf.concat([self.train_images, self.test_images, self.validate_images], axis=0)
        mean_image = tf.reduce_mean(all_images, axis=0)
        std_image = tf.math.reduce_std(all_images, axis=0)

        # Subtract mean and normalize each set
        self.train_images = normalize_tensor(self.train_images, mean_image, std_image)
        self.train_positives = normalize_tensor(self.train_positives, mean_image, std_image)
        self.train_negatives = normalize_tensor(self.train_negatives, mean_image, std_image)
        self.test_images = normalize_tensor(self.test_images, mean_image, std_image)
        self.test_positives = normalize_tensor(self.test_positives, mean_image, std_image)
        self.test_negatives = normalize_tensor(self.test_negatives, mean_image, std_image)
        self.validate_images = normalize_tensor(self.validate_images, mean_image, std_image)
        self.validate_positives = normalize_tensor(self.validate_positives, mean_image, std_image)
        self.validate_negatives = normalize_tensor(self.validate_negatives, mean_image, std_image)
        print("Dataset initialized.")

    def GenerateSet_(num, images, labels, positives, negatives):
        X, Y, P, N = None, None, None, None

        for i in range(num):
            index = np.random.randint(0, len(images))
            indexp = np.random.randint(0, len(positives))
            indexn = np.random.randint(0, len(negatives))

            anchor = PreprocessLayer()(images[index])
            malignant = labels[index]
            if malignant:
                pos = PreprocessLayer()(positives[indexp])
                neg = PreprocessLayer()(negatives[indexn])
            else:
                pos = PreprocessLayer()(negatives[indexn])
                neg = PreprocessLayer()(positives[indexp])

            anchor = tf.reshape(anchor, (-1, 256, 256, 3))
            pos = tf.reshape(pos, (-1, 256, 256, 3))
            neg = tf.reshape(neg, (-1, 256, 256, 3))

            if (X == None):
                X = anchor
                Y = tf.convert_to_tensor([malignant], dtype=tf.float32)
                P = pos-1
                N = neg
                X = tf.reshape(X, (-1, 256, 256, 3))
                Y = tf.reshape(Y, (-1, 1))
                P = tf.reshape(P, (-1, 256, 256, 3))
                N = tf.reshape(N, (-1, 256, 256, 3))
            else:
                X = tf.concat([X, anchor], 0)
                Y = tf.concat([Y, tf.reshape(tf.convert_to_tensor([malignant], dtype=tf.float32), (-1, 1))], 0)
                P = tf.concat([P, pos], 0)
                N = tf.concat([N, neg], 0)
        return X,Y,P,N

    def GenerateTestSet(self, num):
        return Dataset.GenerateSet_(num, self.test_images, self.test_labels, self.test_positives, self.test_negatives)

    def GenerateTrainSet(self, num):
        return Dataset.GenerateSet_(num, self.train_images, self.train_labels, self.train_positives, self.train_negatives)

    def GenerateValidateSet(self, num):
        return Dataset.GenerateSet_(num, self.validate_images, self.validate_labels, self.validate_positives, self.validate_negatives)

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
        img = tf.abs(img)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Malignant")

        plt.subplot(2, n, n + i + 1)
        img = neg_images[i]
        img = tf.abs(img)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Benign")
    plt.tight_layout()
    plt.show()


def plot_random_test_samples(images, num_samples=8):
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
    idxs = np.random.choice(images.shape[0], num_samples, replace=False)
    images = images.numpy()[idxs]

    plt.figure(figsize=(16, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        img = images[i]
        img = tf.abs(img)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    config = json.load(open('recognition/siamese_kaggle_48020774/utility/config.json'))

    dataset = Dataset(
        config["data_dir"], 
        config["class_split"],
        config["train_size"],
        config["test_size"],
        config["validate_size"]    
    )

    X, Y, P, N = dataset.GenerateTestSet(50)

    # plot_random_test_samples(dataset.test_images, 20)
    plot_random_pn_samples(P, N, 5)

    # X_train, Y_train, X_test, Y_test, Panchors, Nanchors = load_ISIC_tensors()
    # print("stats: ")
    # print(X_train.shape, Y_train.shape)
    # print(X_test.shape, Y_test.shape)
    # print("# benign: ", tf.reduce_sum(1 - Y_train).numpy(), tf.reduce_sum(1 - Y_test).numpy())
    # print("# malignant: ", tf.reduce_sum(Y_train).numpy(), tf.reduce_sum(Y_test).numpy())

    # plot_random_test_samples(X_test, Y_test, 20)