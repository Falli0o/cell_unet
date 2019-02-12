
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf


# In[ ]:


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
#TRAIN_PATH = ".\data\all_patch"
#TEST_PATH = ".\data\all_binary"

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
#seed = 42
#random.seed = seed
#np.random.seed = seed
np.random.seed(1)


# In[ ]:


f1_data_filenames = [r"/content/gdrive/My Drive/ColabData/final_dataset/f1/patch/%s" % i for i in os.listdir(r"/content/gdrive/My Drive/ColabData/final_dataset/f1/patch")]
f2_data_filenames = [r"/content/gdrive/My Drive/ColabData/final_dataset/f2/patch/%s" % i for i in os.listdir(r"/content/gdrive/My Drive/ColabData/final_dataset/f2/patch")]
f3_data_filenames = [r"/content/gdrive/My Drive/ColabData/final_dataset/f3/patch/%s" % i for i in os.listdir(r"/content/gdrive/My Drive/ColabData/final_dataset/f3/patch")]
f4_data_filenames = [r"/content/gdrive/My Drive/ColabData/final_dataset/f4/patch/%s" % i for i in os.listdir(r"/content/gdrive/My Drive/ColabData/final_dataset/f4/patch")]
f5_data_filenames = [r"/content/gdrive/My Drive/ColabData/final_dataset/f5/patch/%s" % i for i in os.listdir(r"/content/gdrive/My Drive/ColabData/final_dataset/f5/patch")]

f1_label_filenames = [r"/content/gdrive/My Drive/ColabData/final_dataset/f1/mask/%s" % i for i in os.listdir(r"/content/gdrive/My Drive/ColabData/final_dataset/f1/mask")]
f2_label_filenames = [r"/content/gdrive/My Drive/ColabData/final_dataset/f2/mask/%s" % i for i in os.listdir(r"/content/gdrive/My Drive/ColabData/final_dataset/f2/mask")]
f3_label_filenames = [r"/content/gdrive/My Drive/ColabData/final_dataset/f3/mask/%s" % i for i in os.listdir(r"/content/gdrive/My Drive/ColabData/final_dataset/f3/mask")]
f4_label_filenames = [r"/content/gdrive/My Drive/ColabData/final_dataset/f4/mask/%s" % i for i in os.listdir(r"/content/gdrive/My Drive/ColabData/final_dataset/f4/mask")]
f5_label_filenames = [r"/content/gdrive/My Drive/ColabData/final_dataset/f5/mask/%s" % i for i in os.listdir(r"/content/gdrive/My Drive/ColabData/final_dataset/f5/mask")]


# In[ ]:


data_filenames = sorted(f1_data_filenames) + sorted(f2_data_filenames) + sorted(f3_data_filenames) + sorted(f5_data_filenames) + sorted(f4_data_filenames)
label_filenames = sorted(f1_label_filenames) + sorted(f2_label_filenames) + sorted(f3_label_filenames) + sorted(f5_label_filenames) + sorted(f4_label_filenames)


# In[ ]:


X_train = np.empty((15*4,256,256,3), dtype=np.uint8)
Y_train = np.empty((15*4,256,256,1), dtype=np.bool)
X_test = np.empty((15,256,256,3), dtype=np.uint8)
Y_test = np.empty((15,256,256,1), dtype=np.bool)
for i in range(15*4):
    #i = shuffled_idx[:5][k]
    s_xtrain = data_filenames[i]#r".\data\all_patch\%s" % data_filenames[i]
    s_ytrain = label_filenames[i]#r".\data\all_binary\%s" % label_filenames[i]
    img = imread(s_xtrain)
    label = imread(s_ytrain,as_grey=True)
    label[(label==255)] = True
    label[(label==0)] = False
    label = label.astype(np.bool)
    label = label.reshape(256,256,1)
    X_train[i] = img
    Y_train[i] = label

for j in range(15*4,75):
    #j = shuffled_idx[5:][m]
    s_xtest = data_filenames[j]#r".\data\all_patch\%s" % data_filenames[j]
    s_ytest = label_filenames[j]#r".\data\all_binary\%s" % label_filenames[j]
    img = imread(s_xtest)
    label = imread(s_ytest,as_grey=True)
    label[(label==255)] = True
    label[(label==0)] = False
    label = label.astype(np.bool)
    label = label.reshape(256,256,1)
    X_test[j-15*4] = img
    Y_test[j-15*4] = label


# In[ ]:


X_train = X_train/255
X_test = X_test/255


# In[ ]:


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# In[ ]:


# Build U-Net model

def unet(pretrained_weights = None,optimizer_name = 'adam'):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    b1 = BatchNormalization()(c1)
    #c1 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b1)
    #b1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2,2))(b1)
    p1 = Dropout(0.1) (p1)

    c2 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    #b2 = BatchNormalization()(c2)
    #c2 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b2)
    b2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2,2))(b2)
    p2 = Dropout(0.1) (p2)

    c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    #b3 = BatchNormalization()(c3)
    #c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b3)
    b3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2,2))(b3)
    p3 = Dropout(0.1) (p3)

    c4 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    #b4 = BatchNormalization()(c4)
    #c4 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b4)
    b4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2,2))(b4)
    p4 = Dropout(0.1) (p4)

    c5 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    #b5 = BatchNormalization()(c5)
    #c5 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b5)
    c5 = Dropout(0.2) (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.1) (c6)
    #c6 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.1) (c7)
    #c7 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)


    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer_name, loss='binary_crossentropy', metrics=[mean_iou,'accuracy'])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
   


# In[ ]:


adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=10**(-5), amsgrad=False)
#sgd = SGD(lr=0.01, decay= 1e-6, momentum=0.9, nesterov=True) 
sgd = SGD(lr=0.01, decay= 0, momentum=0.9, nesterov=True) 


# In[ ]:


model = unet(optimizer_name = 'adam')


# In[12]:


filepath=r"/content/gdrive/My Drive/ColabData/final_dataset/f1_model_weights/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
model_checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1, save_best_only=True)
callbacks_list = [model_checkpoint],
hist = model.fit(X_train, Y_train,  batch_size=32,epochs=600,callbacks=[model_checkpoint])#validation_split=0.1,
#his = model.fit_generator(
    #train_generator,
    #steps_per_epoch=10,
    #epochs=20,verbose=1,callbacks=[model_checkpoint])


# In[13]:


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test mean_iou:', score[1])
print('Test accuracy:', score[2])


# In[13]:


#score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test mean_iou:', score[1])
#print('Test accuracy:', score[2])


# In[ ]:


#dict_name = {1:{1:2,3:4},2:{3:4,4:5}}
f = open(r'/content/gdrive/My Drive/ColabData/final_dataset/f2_hist2.txt','w')
f.write(str(hist.history.items()))
f.close()


# In[ ]:


re = model.predict(X_test)


# In[16]:


from google.colab import files
from skimage.io import imread,imsave
fig=plt.figure()
plt.axis('off')
plt.imshow(re[0].reshape(256,256),cmap='gray');
imsave('f2_re2.png',re[0].reshape(256,256));
files.download('f2_re2.png')


# In[17]:


fig=plt.figure()
plt.axis('off')
for i in range(len(re)):
    plt.imshow(re[i].reshape(256,256),cmap='gray');
    s = r'model2_re%s.png' % str(i+1)
    imsave(s,re[i].reshape(256,256));
    files.download(s)


# In[18]:


fig=plt.figure()
#plt.axis('off')
plt.imshow(X_test[0].reshape(256,256,3),cmap='gray')
#imsave('f2_patch.png',X_test[0].reshape(256,256,3));
#files.download('f2_patch.png')


# In[20]:


fig=plt.figure()
#plt.axis('off')
plt.imshow(Y_test[0].reshape(256,256),cmap='gray')
#imsave('f2_label.png', Y_test[0].reshape(256,256).astype(np.uint8));
#files.download('f2_label.png')

