

import theano
import theano.tensor as T
from theano import function, config, shared, tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from numpy import size
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal import pool
from theano.printing import pydotprint

import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os,cv2
import gzip
import pickle
from six.moves import cPickle
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#%%
plt.style.use('ggplot')
#plt.ion()
srng = RandomStreams()
#############################

########################################

def plot_mnist_digit(image):
    '''Plot a single digit from the mnsist dataset'''
    image = np.reshape(image, [128,128])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap=matplotlib.cm.binary)
    plt.yticks(np.array([]))
    plt.show()


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)
def sigmod(X):
    sm = T.nnet.sigmoid(X)
    return sm
def softmax(X):
    #e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    e_x = T.nnet.softmax(X)
    return e_x  #/ e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    #l1a = rectify(conv2d(X, w, border_mode='full'))         #comvolution layer 1
    l1a = rectify(conv2d(X, w))
    l1 = pool.pool_2d(l1a, (2, 2), ignore_border=True)      #maxpooling layer 1
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))                           #comvolution layer 2
    l2 = pool.pool_2d(l2a, (2, 2),ignore_border=True)                           #maxpooling layer 2
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))                           #comvolution layer 3
    l3b = pool.pool_2d(l3a, (2, 2),ignore_border=True)                          #maxpooling layer 3
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    #l4 = rectify(T.dot(l3, w4))                             #fully connection layer
    #l4 = softmax(T.dot(l3, w4))
    l4 = sigmod(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    #pyx = sigmod(T.dot(l4, w_o))                           #fully connection layer
    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

tStart = time.time()

print("Loading data.....")
##### Load data ###########
f = open('data_train/data_train_315.v1.wht', 'rb')
loaded_obj = cPickle.load(f)
trX = loaded_obj[0]
trY = loaded_obj[1]
teX = loaded_obj[2]
teY = loaded_obj[3] 
f.close()
trX = trX.reshape(-1, 1, 128, 128)
teX = teX.reshape(-1, 1, 128, 128)
print("len trX: ",len(trX))
print("len teX: ",len(teX))


X = T.ftensor4()

Y = T.fmatrix()

##load model cPickle
w = init_weights((6, 1, 15, 15))        #128x128
w2 = init_weights((12, 6, 15, 15))      #57x57
w3 = init_weights((24, 12, 11, 11))     #22x22
w4 = init_weights((24 * 5 * 5, 625))    #5X5
w_o = init_weights((625, 2))


print("Loading weight.....")
# #%%%%%%%%%%%%%%%% Load weight %%%%%%%%%%%%%%
# f = open('weight/model_color_6_12_24_625.wht', 'rb')
# loaded_obj = cPickle.load(f)
# w= loaded_obj[0]
# w2= loaded_obj[1]
# w3= loaded_obj[2]
# w4= loaded_obj[3]
# w_o= loaded_obj[4]
# f.close()
#############################################


print("len w: ",len(w.get_value()))
print("len w2: ",len(w2.get_value()))
print("len w3: ",len(w3.get_value()))
print("len w4: ",len(w4.get_value()))
print("len w_o: ",len(w_o.get_value()))

#print("wieght out: ",w_o)
noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)



cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))  # entropy
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)



train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
LL = []
LL1 = []
LL2 = []
LL3 = []
print("Training model.....")
epoch = 50
for i in range(epoch):
    cost1=0
    for start, end in zip(range(0, len(trX), 7), range(7, len(trX), 7)):
        cost = train(trX[start:end], trY[start:end])
        cost1 = cost1+cost
        LL1.append(cost)
    print("cost: ",cost1)
    print ("%d accuracy: "%i, np.mean(np.argmax(teY, axis=1) == predict(teX)))
    print("%d accuracy train: "%i, np.mean(np.argmax(trY, axis=1) == predict(trX)))
    LL.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    LL2.append(np.mean(np.argmax(trY, axis=1) == predict(trX)))
    LL3.append(cost1)

tEnd = time.time()
print('It cost %f minutes:'% ((tEnd-tStart)/60))

################ save model ##################
print("Save model.....")
with open('weight/model_315v1.6_12_24_625.wht','wb') as f:
    cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #pickle.dump(params,f)
    f.close()
##############################################
#plt.ylabel("cost")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(LL,color="blue", linestyle="-", label="Test")
plt.plot(LL2,color="red", linestyle="-", label="Train")
#plt.plot(LL3, color="green",linestyle="-", label="cost")
#plt.xticks([], [])
#plt.yticks([], [])

#
plt.xlim(0, epoch)
plt.ylim(0, 1)

plt.legend(loc='upper left', frameon=False)
plt.show()
print("finish.")







