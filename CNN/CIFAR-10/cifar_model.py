import numpy as np
# from functions import *

class CNN:
    def __init__(self, in_shape, out_size):
        
        in_size = in_shape[0]
        in_chn = in_shape[-1]
        conv1_channel = 12
        self.conv1 = Conv2D(in_chn=in_chn, out_chn=conv1_channel, kernel_size=3, in_shape=in_shape, padding=1, stride=2, bias=False)
        c1 = (in_size-3 + 2*1)//2 + 1
        output_shape = (c1,c1,conv1_channel)
        self.batchnorm1 = BatchNorm(output_shape)
        conv2_channel = 2*conv1_channel
        self.conv2 = Conv2D(in_chn=conv1_channel, out_chn=conv2_channel, kernel_size=3, in_shape=output_shape, padding=1, stride=2, bias=False)
        c2 = (c1-3 + 2*1)//2 + 1
        output_shape = (c2,c2,conv2_channel)
        self.batchnorm2 = BatchNorm(output_shape)
        self.flatten = Flatten()
        linear_in = np.prod(output_shape)
        
        self.softmax = SoftMax(linear_in, out_size)
        
        self.layers = {'conv1': self.conv1, 'batch_norm1': self.batchnorm1, 'conv2': self.conv2, 
                       "batch_norm2": self.batchnorm2, 'softmax': self.softmax}

    def forward(self, X):
        
        X = self.conv1.forward(X)
        X = self.batchnorm1.forward(X)
        
        X = self.conv2.forward(X)
        X = self.batchnorm2.forward(X)

        X = self.flatten.forward(X)
        X = self.softmax.forward(X)

        return X
    
    def backward(self, dZ):
        
        dZ = self.softmax.backward(dZ)
        dZ = self.flatten.backward(dZ)
        
        dZ = self.batchnorm2.backward(dZ)
        dZ = self.conv2.backward(dZ)

        dZ = self.batchnorm1.backward(dZ)
        dZ = self.conv1.backward(dZ)
        
        return dZ
    
    def set_weights(self, weight_list):
        for k, (W, b) in weight_list.items():
            self.layers[k].W = W
            self.layers[k].b = b

    def get_weights(self):
        return {k: (layer.W, layer.b) for k, layer in self.layers.items()}

    def get_dweights(self):
        return {k: (layer.dW, layer.db) for k, layer in self.layers.items()}
    
    
class BatchNorm:
    def __init__(self, input_shape):
        
        d = np.prod(input_shape)
        
        self.w = np.random.randn(d)
        self.b = np.random.randn(d)
        self.W = np.random.randn(d)
    
    def forward(self, x, eps=1e-7):
        
        shape = x.shape
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
        
        N, D = x.shape

        mu = 1./N * np.sum(x, axis = 0)

        xmu = x - mu

        sq = xmu ** 2

        var = 1./N * np.sum(sq, axis = 0)

        sqrtvar = np.sqrt(var + eps)

        ivar = 1./sqrtvar

        xhat = xmu * ivar

        Wx = self.w * xhat

        out = Wx + self.b

        self.cache = (xhat,xmu,ivar,sqrtvar,var,eps)
        
        out = out.reshape(shape)

        return out
    
    def backward(self, dout):
        
        shape = dout.shape
        dout = dout.reshape(dout.shape[0], np.prod(dout.shape[1:]))
        
        xhat,xmu,ivar,sqrtvar,var,eps = self.cache

        N,D = dout.shape

        self.db = np.sum(dout, axis=0)
        dWx = dout

        self.dW = np.sum(dWx*xhat, axis=0)
        dxhat = dWx * self.w

        divar = np.sum(dxhat*xmu, axis=0)
        dxmu1 = dxhat * ivar

        dsqrtvar = -1. /(sqrtvar**2) * divar

        dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

        dsq = 1. /N * np.ones((N,D)) * dvar

        dxmu2 = 2 * xmu * dsq

        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

        dx2 = 1. /N * np.ones((N,D)) * dmu

        dZ = dx1 + dx2
        
        dZ = dZ.reshape(shape)
        
        return dZ
    

class Conv2D:

    def __init__(self, in_chn, out_chn, kernel_size, in_shape, padding=0, stride=1, bias=True):
        
        fan_in = np.prod(in_shape)
        
        self.W = he_normal((kernel_size, kernel_size, in_chn, out_chn), fan_in=fan_in)
        self.b = np.zeros((1,1,1,out_chn))
        self.stride = stride
        self.padding = padding
        self.bias = bias


    def zero_pad(self, X, pad):
        
        X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values = (0,0))

        return X_pad


    def forward(self, A_prev):

        self.A_prev = A_prev

        W = self.W
        b = self.b
        stride = self.stride
        pad = self.padding
        biases = 0

        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        (f, f, n_C_prev, n_C) = W.shape

        n_H = int((n_H_prev - f + 2*pad)/stride) + 1
        n_W = int((n_W_prev - f + 2*pad)/stride) + 1
        
        Z = np.zeros((m, n_H, n_W, n_C))
        
        A_prev_pad = self.zero_pad(A_prev, pad)
        
        windowed_view = np.lib.stride_tricks.sliding_window_view(A_prev_pad, (f,f,n_C_prev), axis=(1,2,3))
        windowed_view = windowed_view[:,::stride,::stride,...]

        for c in range(n_C): 
            out_mul = np.multiply(windowed_view, W[:,:,:,c])
            out_sum = np.sum(out_mul,(-3,-2,-1)) 
            Z[:,:,:,c,None] = out_sum
 
        Z = Z+b
        
        self.Z = Z*(Z>0)
        
        return self.Z

    def backward(self, dZ): 
        W = self.W
        b = self.b
        A_prev = self.A_prev

        stride = self.stride
        pad = self.padding
            
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = W.shape

        (m, n_H, n_W, n_C) = dZ.shape

        dA_prev = np.zeros(A_prev.shape)    
        dW = np.zeros(W.shape)
        db = np.zeros(b.shape)

        A_prev_pad = self.zero_pad(A_prev, pad)
        dA_prev_pad = self.zero_pad(dA_prev, pad)

        dZ = dZ*(self.Z > 0)
        
        for i in range(m):                    
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            
            for h in range(n_H):                   
                for w in range(n_W):              
                    for c in range(n_C):          

                        vert_start = h*stride 
                        vert_end = vert_start + f
                        horiz_start = w*stride
                        horiz_end = horiz_start + f

                        a_slice = a_prev_pad[vert_start: vert_end, horiz_start:horiz_end, :]

                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i,h,w,c]
                        dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                        if self.bias: 
                            db[:,:,:,c] += dZ[i,h,w,c]

            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad,pad:-pad,:]
        
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
        
        self.dZ = dA_prev
        self.dW = dW
        self.db = db
        
        return self.dZ
    
class Flatten:

    def __init__(self):
        pass
    
    def forward(self,Z):
        self.A_prev_shape = Z.shape
        
        self.Z = Z.reshape(Z.shape[0],np.prod(Z.shape[1:]))
        
        return self.Z

    def backward(self, dZ_prev):
        
        self.dZ = dZ_prev.reshape(self.A_prev_shape)

        return self.dZ
    
class SoftMax:

    def __init__(self,in_feature,out_feature):
        self.W = he_normal((in_feature, out_feature), fan_in=in_feature)
        self.b = np.zeros((1, out_feature))
        
    def forward(self, A_prev):
        self.A_prev = A_prev

        Z = A_prev @ self.W + self.b
        
 
        expZ = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        self.A = expZ/np.sum(expZ,axis=-1 , keepdims=True)
        
        
        return self.A

    def backward(self, dZ_prev):

        m = len(self.A)
        
        A_prev = self.A_prev
        
        self.dA = dZ_prev
        
        self.dW = (A_prev.T @ self.dA)
        self.db = np.sum(self.dA, axis=0, keepdims=True)
        
        self.dZ = self.dA @ self.W.T
        
        return self.dZ

class SGD:
    def __init__(self, net, learning_rate=0.001):
        self.net = net
        self.learning_rate = learning_rate
        
    def step(self, net):
        params = net.get_weights()
        dparams = net.get_dweights()
        
        for k, (dW, db) in dparams.items():
            W, b = params[k]
            W -= self.learning_rate * dW
            b -= self.learning_rate * db
            
    def set_lr(self, lr):
        self.learning_rate = lr
     
class Cross_entropy:

    def __init__(self):
        self = self
        
    def forward(self, A, Y):
        self.A_prev = A
        self.Y = Y
        
        m = len(A)

        return 1/m * -np.sum(Y*np.log(A+1e-8))
    
    def backward(self):
        A = self.A_prev
        Y = self.Y
        m = len(A)

        self.dZ = 1/m * (A - Y)
            
        return self.dZ
    

def he_normal(out_shape, fan_in):
    return np.random.randn(*out_shape) * np.sqrt(2./fan_in)
