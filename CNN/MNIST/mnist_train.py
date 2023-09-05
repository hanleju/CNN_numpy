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

mnist_file_path = './data/mnist_train.csv'

images, labels = load_mnist_data(mnist_file_path)

train_data = images.reshape(len(images), 28,28,-1) / 255.0
train_labels = one_hot_encoding(labels, len(np.unique(labels)))

# def split_data(data, labels, split_ratio=0.7):
#     num_samples = len(data)
#     num_train_samples = int(num_samples * split_ratio)
   
#     indices = np.random.permutation(num_samples)
#     shuffled_data = data[indices]
#     shuffled_labels = labels[indices]

#     train_data, val_data = shuffled_data[:num_train_samples], shuffled_data[num_train_samples:]
#     train_labels, val_labels = shuffled_labels[:num_train_samples], shuffled_labels[num_train_samples:]

#     return train_data, train_labels, val_data, val_labels

# train_data, train_labels, val_data, val_labels = split_data(train_data, train_labels, split_ratio=0.7)

print("number", np.argmax(train_labels[0]))

num_iter = 6
batch_size = 128

num_batch = len(train_data)//batch_size
m = num_batch*batch_size 

np.random.seed(12)

net = CNN(in_shape=train_data.shape[1:], out_size=train_labels.shape[1])
loss_func = Cross_entropy()
optimizer = SGD(net,learning_rate=0.001)

for i in range(num_iter):
    
    permutation = np.random.permutation(m)
    
    for j in range(0,m,batch_size):
        
        indices = permutation[j:j+batch_size]
        X_batch, Y_batch = train_data[indices], train_labels[indices]
        
        A = net.forward(X_batch)

        loss = loss_func.forward(A, Y_batch)
        
        dZ = loss_func.backward()
        dZ = net.backward(dZ)

        optimizer.step(net)
        
        accuracy = np.sum(np.argmax(A,axis=1)==np.argmax(Y_batch,axis=1))/len(X_batch)
        print(f"epoch:{i} Loss:{loss} Accuracy:{accuracy}")

with open('mnist2_model11.pkl', 'wb') as f:
    pickle.dump((net, loss_func, optimizer), f)