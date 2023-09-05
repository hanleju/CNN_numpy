import numpy as np
import pickle
from cifar_model import *


def read_npz(filename):
    dataset = np.load(filename)
    data = dataset['data']
    labels = dataset['labels']
    dataset.close()
    
    return data, labels

def one_hot_encoding(y, nclass):
    n = y.size
    Y = np.zeros((n,nclass))
    Y[np.arange(n),y] = 1 

    return Y

test_data, test_labels = read_npz("./data/cifar_test.npz")

test_data = test_data.reshape(-1,3,32,32).swapaxes(1,-1) / 255.0
test_labels = test_labels.reshape(-1).astype(int)
test_labels = one_hot_encoding(test_labels, len(np.unique(test_labels)))

with open('cifar_model.pkl', 'rb') as f:
    net, loss_func, optimizer = pickle.load(f)

A = net.forward(test_data)
loss = loss_func.forward(A, test_labels)
accuracy = np.sum(np.argmax(A, axis=1) == np.argmax(test_labels, axis=1)) / len(test_data)

print(f"Num test:{len(test_data)} Accuracy:{accuracy}")