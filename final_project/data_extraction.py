import gzip
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

SEED = 1848399


def load_mnist(path, kind='train'):
    """
    Load MNIST (2 classes) data from parameter path
        @author: marco
        @ date: Sun Nov 24 11:23:09 2019
    """
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images_ = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
        images = images_.reshape(len(labels), 784)

    """
    We are only interested in the items with label 5, 7 and 9.
    Only a subset of 1000 samples per class will be used.
    """
    index_label5 = np.where((labels == 5))
    x_label5 = images[index_label5][:1000, :].astype('float64')
    y_label5 = labels[index_label5][:1000].astype('float64')

    index_label7 = np.where((labels == 7))
    x_label7 = images[index_label7][:1000, :].astype('float64')
    y_label7 = labels[index_label7][:1000].astype('float64')

    index_label9 = np.where((labels == 9))
    x_label9 = images[index_label9][:1000, :].astype('float64')
    y_label9 = labels[index_label9][:1000].astype('float64')

    return x_label5, y_label5, x_label7, y_label7, x_label9, y_label9


def load_binary_mnist(path, kind='train'):
    x_label5, y_label5, x_label7, y_label7, _, _ = load_mnist(path, kind)

    # converting labels of classes 5 and 7 into +1 and -1, respectively
    y_label5 = y_label5 / 5.0
    y_label7 = y_label7 / -7.0

    x_label_57 = np.vstack((x_label5, x_label7))
    y_label_57 = np.concatenate((y_label5, y_label7))

    x_label_57 = scale(x_label_57)

    x_train57, x_test57, y_train57, y_test57 = train_test_split(
        x_label_57, y_label_57, test_size=0.3, random_state=SEED, stratify=y_label_57)

    y_train57 = y_train57.reshape(-1, 1)
    y_test57 = y_test57.reshape(-1, 1)

    return x_train57, y_train57, x_test57, y_test57


def load_multi_class_mnist(path, kind='train'):
    x_label5, y_label5, x_label7, y_label7, x_label9, y_label9 = load_mnist(path, kind)

    x_label_579 = np.vstack((x_label5, x_label7, x_label9))
    y_label_579 = np.concatenate((y_label5, y_label7, y_label9))

    x_label_579 = scale(x_label_579)

    x_train579, x_test579, y_train579, y_test579 = train_test_split(
        x_label_579, y_label_579, test_size=0.3, random_state=SEED, stratify=y_label_579)

    y_train579 = y_train579.reshape(-1, 1)
    y_test579 = y_test579.reshape(-1, 1)

    return x_train579, y_train579, x_test579, y_test579
