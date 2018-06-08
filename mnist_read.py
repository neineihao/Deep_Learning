import os
import struct
import numpy as np
import cv2

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    r_lbl = np.zeros((num, 10))
    for i in range(len(lbl)):
        r_lbl[i][lbl[i]] = 1

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
        img = np.divide(img, 255.0)
    get_img = lambda idx: (r_lbl[idx].astype(np.float64), img[idx].reshape(-1).astype(np.float64))
    # Create an iterator which returns each image in turn
    yield num
    while True:
        for i in range(len(lbl)):
            yield get_img(i)

def package_data(f,size):
    r_img = None
    r_label = None
    for i in range(size):
        if i == 0:
            r_label, r_img = next(f)
        else:
            tmp_label, tmp_img = next(f)
            r_label = np.concatenate((r_label, tmp_label))
            r_img = np.concatenate((r_img, tmp_img))
    return reshape(r_img, size), reshape(r_label, size)

def reshape(n_a, size):
    l_size = n_a.shape[0]
    row_size = l_size // size
    col_size = size
    return n_a.reshape(col_size, row_size)

if __name__ == '__main__':
    f = read(path='MNIST_data')
    print(next(f))
    # label, img = next(f)
    # print(img.shape)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # print("label: {}".format(label))
    # print("img : {}".format(img))
    # print(img.shape)
    img, label = package_data(f, 500)
    index = 450
    print(img[index])
    # cv2.imshow('img', img[index].reshape(28,28))
    # cv2.waitKey(0)
    # print(img[99])
    # ascii_show(img[0])