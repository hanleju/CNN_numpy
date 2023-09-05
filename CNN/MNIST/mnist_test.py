import numpy as np
import pickle
from mnist_model import *

def load_mnist_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    images = data[:, 1:] / 255.0
    labels = data[:, 0].astype(int)
    return images, labels

def one_hot_encoding(y, nclass):
    n = y.size
    Y = np.zeros((n,nclass))
    Y[np.arange(n),y] = 1 

    return Y

mnist_file_path = './data/mnist_test.csv'

images, labels = load_mnist_data(mnist_file_path)

images = images.reshape(len(images), 28,28,-1) / 255.0
labels = one_hot_encoding(labels, len(np.unique(labels)))

with open('mnist_model.pkl', 'rb') as f:
    net, loss_func, optimizer = pickle.load(f)

A = net.forward(images)
loss = loss_func.forward(A, labels)
accuracy = np.sum(np.argmax(A, axis=1) == np.argmax(labels, axis=1)) / len(images)

print(f"Num test:{len(images)} Accuracy:{accuracy}")