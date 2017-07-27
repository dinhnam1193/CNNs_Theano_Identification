import theano
from theano import function, config, shared, tensor
from theano import tensor as T
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from numpy import size
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal import pool
import matplotlib.image as mpimg
import pylab
import matplotlib
import matplotlib.pyplot as plt
import os,cv2,sys
import gzip
import pickle
from six.moves import cPickle
import time
#from keras.utils import np_utils
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# srng = RandomStreams()



def plot_mnist_digit(image):
    '''Plot a single digit from the mnsist dataset'''
    image = np.reshape(image, [128,128])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):  # rectified linear unit (ReLU)
    return T.maximum(X, 0.)
def sigmod(X):
    sm = T.nnet.sigmoid(X)
    return sm
def softmax(X):
    #e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    e_x = T.nnet.softmax(X)
    return e_x  #/ e_x.sum(axis=1).dimshuffle(0, 'x')

# Tat ngau nhien cac node trong mang neural de gian hien tuong qua khop va gian thieu
# viec tinh toan cho mang
def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X
# RMSprop cap nhat cac trong so voi ti le hoc khac nhau cho moi trong so
# phu thuoc vao toc hoc cua nhung lan truoc do va grad cua trong so do
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


#%% Load Image

############### data cats ouput = [0] ####################
#test_image = cv2.imread('data/cats/cat.177.jpg')

############### data dogs ouput = [3] ####################
#test_image = cv2.imread('data/dogs/dog.32.jpg')
test_image = cv2.imread('data/B/Data_20.jpg')
# test_image = cv2.imread('data/A/Data_119.jpg')
#img = mpimg.imread('data_dogs_cats/Humans/rider-1.jpg')
im = test_image

test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255



#print(test_image.shape)
#test_image = test_image.transpose(2, 0, 1).reshape(1, 1, 128, 128)
print(test_image.shape)
f = open('weight/model_315v1.6_12_24_625.wht', 'rb')
loaded_obj = cPickle.load(f)
w= loaded_obj[0]
w2= loaded_obj[1]
w3= loaded_obj[2]
w4= loaded_obj[3]
w_o= loaded_obj[4]
f.close()

input = T.tensor4(name='input')

conv_out = rectify(conv2d(input, w))
l1 = pool.pool_2d(conv_out, (2, 2), ignore_border=False)
f = theano.function([input], l1)
test_image = test_image.reshape(1,1,128,128)
filtered_img = f(test_image)

conv_out2 = rectify(conv2d(input, w2))
l2 = pool.pool_2d(conv_out2, (2, 2), ignore_border=False)
f2 = theano.function([input], l2)
filtered_img2 = f2(filtered_img)

conv_out3 = rectify(conv2d(input, w3))
l3 = pool.pool_2d(conv_out3, (2, 2), ignore_border=False)
f3 = theano.function([input], l3)
filtered_img3 = f3(filtered_img2)

#l1a = conv2d(image, w, border_mode='full')        #comvolution layer 1
#l1a = rectify(conv2d(X, w))
#l1 = pool.pool_2d(l1a, (2, 2), ignore_border=False)      #maxpooling layer 1
#l1 = dropout(l1, 0.0)

print("l1: ",filtered_img.shape)
print("l2: ",filtered_img2.shape)
print("l3: ",filtered_img3.shape)
#plt.imshow(l1a[0,5,:,:])
# pylab.gray()
# ## layer 1
# pylab.subplot(3,3,1)
# pylab.imshow(filtered_img[0,0,:,:])
# pylab.subplot(3,3,2)
# pylab.imshow(filtered_img[0,1,:,:])
# pylab.subplot(3,3,3)
# pylab.imshow(filtered_img[0,2,:,:])
# pylab.subplot(3,3,4)
# pylab.imshow(filtered_img[0,3,:,:])
# pylab.subplot(3,3,5)
# pylab.imshow(filtered_img[0,4,:,:])
# pylab.subplot(3,3,6)
# pylab.imshow(filtered_img[0,5,:,:])
# pylab.subplot(3,3,8)
# pylab.imshow(im)

#################
### layer 2
# pylab.subplot(4,4,1)
# pylab.imshow(filtered_img2[0,0,:,:])
# pylab.subplot(4,4,2)
# pylab.imshow(filtered_img2[0,1,:,:])
# pylab.subplot(4,4,3)
# pylab.imshow(filtered_img2[0,2,:,:])
# pylab.subplot(4,4,4)
# pylab.imshow(filtered_img2[0,3,:,:])
# pylab.subplot(4,4,5)
# pylab.imshow(filtered_img2[0,4,:,:])
# pylab.subplot(4,4,6)
# pylab.imshow(filtered_img2[0,5,:,:])
# pylab.subplot(4,4,7)
# pylab.imshow(filtered_img2[0,6,:,:])
# pylab.subplot(4,4,8)
# pylab.imshow(filtered_img2[0,7,:,:])
# pylab.subplot(4,4,9)
# pylab.imshow(filtered_img2[0,8,:,:])
# pylab.subplot(4,4,10)
# pylab.imshow(filtered_img2[0,9,:,:])
# pylab.subplot(4,4,11)
# pylab.imshow(filtered_img2[0,10,:,:])
# pylab.subplot(4,4,12)
# pylab.imshow(filtered_img2[0,11,:,:])
# pylab.subplot(4,4,14)
# pylab.imshow(im)
#
# #%%%%%%%%%%%%%%%%%%%%%%%5
# ## layer 3
pylab.subplot(5,6,1)
pylab.imshow(filtered_img3[0,0,:,:])
pylab.subplot(5,6,2)
pylab.imshow(filtered_img3[0,1,:,:])
pylab.subplot(5,6,3)
pylab.imshow(filtered_img3[0,2,:,:])
pylab.subplot(5,6,4)
pylab.imshow(filtered_img3[0,3,:,:])
pylab.subplot(5,6,5)
pylab.imshow(filtered_img3[0,4,:,:])
pylab.subplot(5,6,6)
pylab.imshow(filtered_img3[0,5,:,:])
pylab.subplot(5,6,7)
pylab.imshow(filtered_img3[0,6,:,:])
pylab.subplot(5,6,8)
pylab.imshow(filtered_img3[0,7,:,:])
pylab.subplot(5,6,9)
pylab.imshow(filtered_img3[0,8,:,:])
pylab.subplot(5,6,10)
pylab.imshow(filtered_img3[0,9,:,:])
pylab.subplot(5,6,11)
pylab.imshow(filtered_img3[0,10,:,:])
pylab.subplot(5,6,12)
pylab.imshow(filtered_img3[0,11,:,:])
pylab.subplot(5,6,13)
pylab.imshow(filtered_img3[0,12,:,:])
pylab.subplot(5,6,14)
pylab.imshow(filtered_img3[0,13,:,:])
pylab.subplot(5,6,15)
pylab.imshow(filtered_img3[0,14,:,:])
pylab.subplot(5,6,16)
pylab.imshow(filtered_img3[0,15,:,:])
pylab.subplot(5,6,17)
pylab.imshow(filtered_img3[0,16,:,:])
pylab.subplot(5,6,18)
pylab.imshow(filtered_img3[0,17,:,:])
pylab.subplot(5,6,19)
pylab.imshow(filtered_img3[0,18,:,:])
pylab.subplot(5,6,20)
pylab.imshow(filtered_img3[0,19,:,:])
pylab.subplot(5,6,21)
pylab.imshow(filtered_img3[0,20,:,:])
pylab.subplot(5,6,22)
pylab.imshow(filtered_img3[0,21,:,:])
pylab.subplot(5,6,23)
pylab.imshow(filtered_img3[0,22,:,:])
pylab.subplot(5,6,24)
pylab.imshow(filtered_img3[0,23,:,:])
pylab.subplot(5,6,27)
pylab.imshow(im)

pylab.show()

###########################
# h = 10
# c = 6
#
# pylab.subplot(h,c,3)
# pylab.imshow(im)
# pylab.subplot(h,c,7)
# pylab.imshow(filtered_img[0,0,:,:])
# pylab.subplot(h,c,8)
# pylab.imshow(filtered_img[0,1,:,:])
# pylab.subplot(h,c,9)
# pylab.imshow(filtered_img[0,2,:,:])
# pylab.subplot(h,c,10)
# pylab.imshow(filtered_img[0,3,:,:])
# pylab.subplot(h,c,11)
# pylab.imshow(filtered_img[0,4,:,:])
# pylab.subplot(h,c,12)
# pylab.imshow(filtered_img[0,5,:,:])
#
#
# ################
# ## layer 2
# pylab.subplot(h,c,19)
# pylab.imshow(filtered_img2[0,0,:,:])
# pylab.subplot(h,c,20)
# pylab.imshow(filtered_img2[0,1,:,:])
# pylab.subplot(h,c,21)
# pylab.imshow(filtered_img2[0,2,:,:])
# pylab.subplot(h,c,22)
# pylab.imshow(filtered_img2[0,3,:,:])
# pylab.subplot(h,c,23)
# pylab.imshow(filtered_img2[0,4,:,:])
# pylab.subplot(h,c,24)
# pylab.imshow(filtered_img2[0,5,:,:])
# pylab.subplot(h,c,25)
# pylab.imshow(filtered_img2[0,6,:,:])
# pylab.subplot(h,c,26)
# pylab.imshow(filtered_img2[0,7,:,:])
# pylab.subplot(h,c,27)
# pylab.imshow(filtered_img2[0,8,:,:])
# pylab.subplot(h,c,28)
# pylab.imshow(filtered_img2[0,9,:,:])
# pylab.subplot(h,c,29)
# pylab.imshow(filtered_img2[0,10,:,:])
# pylab.subplot(h,c,30)
# pylab.imshow(filtered_img2[0,11,:,:])
#
# #%%%%%%%%%%%%%%%%%%%%%%%5
# ## layer 3
# pylab.subplot(h,c,37)
# pylab.imshow(filtered_img3[0,0,:,:])
# pylab.subplot(h,c,38)
# pylab.imshow(filtered_img3[0,1,:,:])
# pylab.subplot(h,c,39)
# pylab.imshow(filtered_img3[0,2,:,:])
# pylab.subplot(h,c,40)
# pylab.imshow(filtered_img3[0,3,:,:])
# pylab.subplot(h,c,41)
# pylab.imshow(filtered_img3[0,4,:,:])
# pylab.subplot(h,c,42)
# pylab.imshow(filtered_img3[0,5,:,:])
# pylab.subplot(h,c,43)
# pylab.imshow(filtered_img3[0,6,:,:])
# pylab.subplot(h,c,44)
# pylab.imshow(filtered_img3[0,7,:,:])
# pylab.subplot(h,c,45)
# pylab.imshow(filtered_img3[0,8,:,:])
# pylab.subplot(h,c,46)
# pylab.imshow(filtered_img3[0,9,:,:])
# pylab.subplot(h,c,47)
# pylab.imshow(filtered_img3[0,10,:,:])
# pylab.subplot(h,c,48)
# pylab.imshow(filtered_img3[0,11,:,:])
# pylab.subplot(h,c,49)
# pylab.imshow(filtered_img3[0,12,:,:])
# pylab.subplot(h,c,50)
# pylab.imshow(filtered_img3[0,13,:,:])
# pylab.subplot(h,c,51)
# pylab.imshow(filtered_img3[0,14,:,:])
# pylab.subplot(h,c,52)
# pylab.imshow(filtered_img3[0,15,:,:])
# pylab.subplot(h,c,53)
# pylab.imshow(filtered_img3[0,16,:,:])
# pylab.subplot(h,c,54)
# pylab.imshow(filtered_img3[0,17,:,:])
# pylab.subplot(h,c,55)
# pylab.imshow(filtered_img3[0,18,:,:])
# pylab.subplot(h,c,56)
# pylab.imshow(filtered_img3[0,19,:,:])
# pylab.subplot(h,c,57)
# pylab.imshow(filtered_img3[0,20,:,:])
# pylab.subplot(h,c,58)
# pylab.imshow(filtered_img3[0,21,:,:])
# pylab.subplot(h,c,59)
# pylab.imshow(filtered_img3[0,22,:,:])
# pylab.subplot(h,c,60)
# pylab.imshow(filtered_img3[0,23,:,:])
#
# pylab.show()


