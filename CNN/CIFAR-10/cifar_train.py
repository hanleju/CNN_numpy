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

train_data, train_labels = read_npz("./data/cifar_train.npz")

train_data = train_data.reshape(-1,3,32,32).swapaxes(1,-1) / 255.0
train_labels = train_labels.reshape(-1).astype(int)
train_labels = one_hot_encoding(train_labels, len(np.unique(train_labels)))

num_iter = 10
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


with open('cifar_model.pkl', 'wb') as f:
    pickle.dump((net, loss_func, optimizer), f)