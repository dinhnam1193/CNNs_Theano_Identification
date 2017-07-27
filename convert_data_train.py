import theano
from theano import function, config, shared, tensor
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from numpy import size
import os,cv2
import gzip
import pickle
from six.moves import cPickle
import time
from keras.utils import np_utils
from sklearn.utils import shuffle

from keras import backend as K
K.set_image_dim_ordering('th')

from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

srng = RandomStreams()


def plot_mnist_digit(image):
    '''Plot a single digit from the mnsist dataset'''
    image = np.reshape(image, [128,128])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()


#%%
img_rows=128
img_cols=128
num_channel=1

PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)
img_data_list=[]
for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img = cv2.imread(data_path + '/'+ dataset + '/'+ img)
		input_img = cv2.cvtColor(input_img,cv2.COLOR_RGB2GRAY)
		input_img_resize = cv2.resize(input_img,(128,128))
        # input_img_resize = cv2.equalizeHist(input_img_resize)
		img_data_list.append(input_img_resize) ##  updates img_data_list
#
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print ("len data: ",len(img_data))
# %%

print(img_data.shape)

if num_channel == 1:
    if K.image_dim_ordering() == 'th':
        img_data = np.expand_dims(img_data, axis=1)
        print(img_data.shape)
    else:
        img_data = np.expand_dims(img_data, axis=4)
        print(img_data.shape)

else:
    if K.image_dim_ordering() == 'th':
        img_data = np.rollaxis(img_data, 3, 1)
print(img_data.shape)


# Assigning Labels

# Define the number of classes
num_classes = 2

num_of_samples = img_data.shape[0]
#num_of_samples = size(img_data)
print("num_of_samples: ",num_of_samples)
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:315] = 0
labels[315:] = 1


names = ['co_pk', 'ko_pk']
# convert class labels to on-hot encoding
Ytt = np_utils.to_categorical(labels, num_classes)
print(Ytt[105])
#plot_mnist_digit(img_data[105])

#print(predict(trX))
#print(np.argmax(trY, axis=1))
#im = plt.imshow(img_data[599])
#plt.show()

# Shuffle the dataset
x, y = shuffle(img_data, Ytt, random_state=2)
# Split the dataset
print("Loading data.....")
trX, teX, trY, teY = train_test_split(x, y, test_size=0.2, random_state=3)
#trX = trX.reshape(-1, 1, 128, 128)
#teX = teX.reshape(-1, 1, 128, 128)

trX = trX.astype('float32')
teX = teX.astype('float32')

data_tt = [trX, trY, teX, teY]

################### Save data train #####################
print("Save data_train")
with open('data_train/data_train_315.v1.wht','wb') as f:
    pickle.dump(data_tt,f)
#########################################################
print("finish.")