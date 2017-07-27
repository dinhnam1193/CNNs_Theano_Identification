
import theano
from theano import function, config, shared, tensor
from theano import tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal import pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle
import cv2
from six.moves import cPickle
from keras import backend as K
srng = RandomStreams()

num_channel=1
#test_image = cv2.imread('data_test/LAM/Data_148.jpg')
#test_image = cv2.imread('data_test/test/t.jpg')
# test_image = cv2.imread('data_test/test/khong/t46.jpg')
test_image = cv2.imread('data/B/Data_71.jpg')
#test_image = cv2.imread('data_mat/A/Data_63.png')


def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))

imgt = histogram_equalize(test_image)
plt.subplot(1,2,1)
plt.imshow(imgt)
plt.subplot(1,2,2)
plt.imshow(test_image)
plt.show()

test_image= cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#test_image= histogram_equalize(test_image)
#test_image = cv2.equalizeHist(test_image)
test_image= cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
im = test_image

print(test_image.shape)


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
    l1a = rectify(conv2d(X, w))         #comvolution layer 1
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

X = T.ftensor4()

Y = T.fmatrix()

print("Loading weight.....")
#%%%%%%%%%%%%%%%% Load weight %%%%%%%%%%%%%%
#f = open('weight/model_mat_2.wht', 'rb')
f = open('weight/model_315v1.6_12_24_625.wht', 'rb')
loaded_obj = cPickle.load(f)
w= loaded_obj[0]
w2= loaded_obj[1]
w3= loaded_obj[2]
w4= loaded_obj[3]
w_o= loaded_obj[4]
f.close()
print("len w: ",len(w.get_value()))
print("len w2: ",len(w2.get_value()))
print("len w3: ",len(w3.get_value()))
print("len w4: ",len(w4.get_value()))
print("len w_o: ",len(w_o.get_value()))
#############################################
noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)
### ve bieu do
y_x1 = py_x
###
cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

#train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
### ve bieu do
predict1 = theano.function(inputs=[X], outputs=y_x1, allow_input_downcast=True)
###
test_image = test_image.reshape(1, 1, 128, 128)
txt = predict(test_image)
txt1= predict1(test_image)
print(predict(test_image))
if txt == 0 :
    print ("khong_pk")
    a = 0
else:
    print("co_pk")
    a = 1
############ ve do thi
txt1 = txt1[0]
names = ['ko_pk',  'co_pk']
ind = range(len(names))
plt.bar(ind, txt1, align='center')
plt.xticks(ind, names)
# Label x, y axit
plt.xlabel('kind')
plt.ylabel('positive')
# Label title of bar char
plt.title('bieu do')

#them gia tri oi cot
for x, y in zip(ind, txt):
    plt.text(x , y , '%d' % y, ha='center', va='bottom')

# Tang truc y don vi
plt.ylim(0, 1)
# show ket qua
plt.show()
#print a
# cv2.imshow('image',im)
############# ket noi voi arduino############
#while not connected:
#    connected = True
####Tell the arduino to blink!
#if a == 0:   #khong_pk
#    ser.write("1")
#    #time.sleep(5)
#    #ser.write("0")
#else:       #co_pk
#    ser.write("2")
###########################################
## close the port and end the program
# cv2.waitKey(0) # wait for closing
# cv2.destroyAllWindows() # Ok, destroy the window
#ser.write("0") # tat led
#ser.close()

                     