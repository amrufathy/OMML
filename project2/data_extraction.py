import gzip
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 1848399
random.seed(SEED)
np.random.seed(SEED)


def load_mnist(path, kind='train'):
    """
    Load MNIST data from `path`
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
    We are only interested in the items with label 2, 4 and 6.
    Only a subset of 1000 samples per class will be used.
    """

    index_label2 = np.where((labels == 2))
    x_label2 = images[index_label2][:1000, :].astype('float64')
    y_label2 = labels[index_label2][:1000].astype('float64')

    index_label4 = np.where((labels == 4))
    x_label4 = images[index_label4][:1000, :].astype('float64')
    y_label4 = labels[index_label4][:1000].astype('float64')

    '''
    index_label6 = np.where((labels == 6))
    x_label6 = images[index_label6][:1000, :].astype('float64')
    y_label6 = labels[index_label6][:1000].astype('float64')

    train_x, train_y = [], []

    train_x.extend(x_label2)
    train_x.extend(x_label4)
    train_x.extend(x_label6)

    train_y.extend(y_label2)
    train_y.extend(y_label4)
    train_y.extend(y_label6)
    '''

    # converting labels of classes 2 and 4 into +1 and -1, respectively
    y_label2 = y_label2 / 2.0
    y_label4 = y_label4 / -4.0

    x_label_24 = np.concatenate((x_label2, x_label4))
    y_label_24 = np.concatenate((y_label2, y_label4))

    x_train24, x_test24, y_train24, y_test24 = train_test_split(
        x_label_24, y_label_24, test_size=0.3, random_state=SEED)

    scaler = StandardScaler()
    scaler.fit(x_train24)
    x_train24 = scaler.transform(x_train24)
    x_test24 = scaler.transform(x_test24)

    return x_train24, y_train24, x_test24, y_test24


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = load_mnist('Data', kind='train')
