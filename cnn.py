import theano
from theano import function, config, shared, tensor
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal import pool

from theano.tensor.signal.downsample import max_pool_2d
import matplotlib
import matplotlib.pyplot as plt
import time
from six.moves import cPickle
#matplotlib inline

srng = RandomStreams()


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
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

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
    l2 = pool.pool_2d(l2a, [2, 2])                           #maxpooling layer 2
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))                           #comvolution layer 3 rule max
    l3b = pool.pool_2d(l3a, [2, 2])                          #maxpooling layer 3
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))                             #fully connection layer
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))                           #fully connection layer
    return l1, l2, l3, l4, pyx

tStart = time.time()


print("Loading data.....")
##### Load data ###########

X = T.ftensor4()
Y = T.fmatrix()

'''w = init_weights((32, 1, 5, 5))
w2 = init_weights((64, 32, 5, 5))
w3 = init_weights((128, 64, 5, 5))
w4 = init_weights((128 * 1 * 1, 625))
w_o = init_weights((625, 10))'''

#w = init_weights((6, 1, 5, 5))
#w2 = init_weights((12, 6, 5, 5))
#w3 = init_weights((24, 12, 5, 5))
#w4 = init_weights((24 * 13 * 13, 625))
#w_o = init_weights((625, 10))
#w_o = init_weights((625, 10))

#print("Loading weight.....")
##%%%%%%%%%%%%%%%% Load weight %%%%%%%%%%%%%%
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


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)



ww = w.get_value()
ww1 = w2.get_value()
ww2 = w3.get_value()

############################################### show filter layer
sx,sy =(2,3)
f,con = plt.subplots(sx,sy, sharex='col', sharey= 'row')

for xx in range(sx):
    for yy in range(sy):
        con[xx,yy].pcolor(ww[yy*2+xx][0])
        #con[xx, yy].pcolor(all_digits[yy * 3 + xx], cmap=plt.cm.gray_r)
plt.show()
print ("Filter layer 1 truoc khi huan luyen")
# view layer 4 FC 4
ww_4 = w4.get_value()
plt.figure()
plt.pcolor(ww_4.T)
plt.show()
################################################ end show fillter layer
def init_rand_img(shape):
   return theano.shared(floatX(np.random.rand(*shape)))

img = init_rand_img([1,1,128,128])
#img = trX[]
start = img.get_value()[0][0]


def dream_model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
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

l1, l2, l3, l4, d_pyx = dream_model(img, w, w2, w3, w4, 0., 0.)

def ascend(img, digit, cost, lr =0.001, rho=0.9, epsilon=1e-6):
   grads = T.grad(cost=cost,wrt=img)
   updates =[]
   for p,g in zip(img, grads):
       acc = theano.shared(p.get_value()*0.)
       acc_new = rho*acc +(1-rho)*g**2
       gradient_scaling = T.sqrt(acc_new+epsilon)
       g=g/gradient_scaling
       updates.append((acc,acc_new))
       updates.append((p,p+lr*g))
   return updates
dream_cost = T.mean(T.nnet.categorical_crossentropy(d_pyx, Y))
params = [img]
dream_updates = ascend(params,y_x,dream_cost,lr=0.1)


dream = theano.function(inputs=[Y], outputs= img, updates=dream_updates, allow_input_downcast=True)

all_digits = [start]
#t=0
#print(trY[30])
for i in range(1,16):
   dream_digits =  dream([[0.,1.]])
   all_digits.append(img.get_value()[0][0])

sx,sy =(4,4)
f,con = plt.subplots(sx,sy, sharex='col', sharey= 'row')
for xx in range(sx):
   for yy in range(sy):
       con[xx, yy].pcolor(all_digits[yy * 4 + xx])#, cmap=plt.cm.gray_r)

plt.show()
