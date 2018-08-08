import os

from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Model, load_model
from keras.models import Sequential
import time

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle


data_dir = './hanwriting'
train_data_x = pd.read_csv(os.path.join(data_dir, 'train_X.csv'))
train_data_y = pd.read_csv(os.path.join(data_dir, 'train_Y.csv'))
test_data_x = pd.read_csv(os.path.join(data_dir, 'test_X.csv'))
test_data_y = pd.read_csv(os.path.join(data_dir, 'test_Y.csv'))

# shuffle样本--打乱样本顺序  #(33505,4096)(33505,140)   #(8380,4096)(8380,140)
train_data_x, train_data_y = shuffle(train_data_x, train_data_y, random_state=0)
test_data_x, test_data_y = shuffle(test_data_x, test_data_y, random_state=0)

print('train_data_x:{}'.format(train_data_x.shape))
print('train_data_y:{}'.format(train_data_y.shape))
print('test_data_x:{}'.format(test_data_x.shape))
print('test_data_y:{}'.format(test_data_y.shape))


batch_size = 128
epochs = 5	#50
img_width, img_height = 64, 64

train_x = np.array(train_data_x).reshape(-1, img_height,img_width,1)   #(33505, 64, 64, 1)
test_x = np.array(test_data_x).reshape(-1,img_height,img_width,1)    #(8380, 64, 64, 1)

#import matplotlib.pyplot as plt
#n = train_data_x.iloc[3].values.reshape(64, 64)
#plt.imshow(n, cmap='gray')
#plt.show()

###########   keras-model    ######################

def build_model_keras(Input_shape=(img_height, img_width, 1), classes=train_data_y.shape[1]):
	model=Sequential(name='keras_model')
	model.add(Convolution2D(32, (3, 3), activation='relu', padding='same', name='block1_conv',input_shape=Input_shape))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
	
	model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', name='block2_conv'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
	
	model.add(Flatten(name='flatten'))
	model.add(Dense(1024, activation='relu', name='fc2'))
	model.add(Dropout(0.5))
	
	model.add(Dense(classes, activation='softmax', name='predictions'))
	print(model.summary())
	
	return model


def train_keras(model):
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.fit(train_x,train_data_y,batch_size=batch_size,epochs=epochs)

model = build_model_keras()
start=time.time()
train_keras(model)
end=time.time()
print('用时：{}'.format(end-start))

# 训练完后，在测试集上进行测试
print('test accuracy %g' %  model.evaluate(test_x[:500],test_data_y[:500]))

model.save("./model.h5")
