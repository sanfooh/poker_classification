from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os,sys
import cv2
import numpy as np
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


def create_pic(src_folder,des_folder,train_count,validate_count):
    for dirpath,dirnames,filenames in os.walk(src_folder):
        for filename in filenames:
            (name,extension) = os.path.splitext(filename)
            full_src_name=dirpath+'/'+filename
            train_path= des_folder+'/train/'+name+'/'
            os.makedirs(train_path)
            validate_path=des_folder+'/validation/'+name+'/'
            os.makedirs(validate_path)
            img = load_img(full_src_name)  
            x = img_to_array(img)  
            x = x.reshape((1,) + x.shape)  
            i = 0
            for batch in datagen.flow(x, batch_size=1,save_to_dir=train_path, save_prefix='train', save_format='jpeg'):
                i += 1
                if i > train_count:
                     break 
            i = 0
            for batch in datagen.flow(x, batch_size=1,save_to_dir=validate_path, save_prefix='validate', save_format='jpeg'):
                i += 1
                if i > validate_count:
                     break 

                   
def create_dataset(src_folder):
    train_x=[]
    train_y=[]
    validate_x=[]
    validate_y=[]
    for dirpath,dirnames,filenames in os.walk(src_folder+'/train/'):
            for filename in filenames:
                full_src_name=dirpath+'/'+filename
                (name,extension) = os.path.split(dirpath)
                img=cv2.imread(full_src_name)
                img=cv2.resize(img,(224,224))
                train_x.append(img)
                train_y.append(int(extension))
                
    for dirpath,dirnames,filenames in os.walk(src_folder+'/validation/'):
            for filename in filenames:
                full_src_name=dirpath+'/'+filename
                (name,extension) = os.path.split(dirpath)
                img=cv2.imread(full_src_name)
                img=cv2.resize(img,(224,224))
                validate_x.append(img)
                validate_y.append(int(extension))
    print(np.array(train_x).shape,np.array(train_y).shape,np.array(validate_x).shape,np.array(validate_y).shape)          
    return (np.array(train_x),np.array(train_y)),(np.array(validate_x),np.array(validate_y))

create_pic('download_pic','data',1000,10)
print('create_pic')
#create_dataset('data')

import keras
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from sklearn.metrics import log_loss

import numpy as np

from keras import backend as K
from keras.utils import np_utils

nb_train_samples = 5420 # 3000 training samples
nb_valid_samples = 54 # 100 validation samples
num_classes = 55

K.set_image_dim_ordering('tf')

img_rows, img_cols, img_channel = 224, 224, 3
(X_train, Y_train), (X_valid, Y_valid) = create_dataset('data')
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_valid = np_utils.to_categorical(Y_valid, num_classes)

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
for layer in base_model.layers:
    layer.trainable=False
    
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(num_classes, activation='sigmoid'))

print(base_model.output.shape)

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


print(X_train.shape,Y_train.shape, X_valid.shape, Y_valid.shape )

batch_size = 16 
nb_epoch = 10
model.fit(X_train, Y_train,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            shuffle=True,
            verbose=1,
            validation_data=(X_valid, Y_valid))
model.save('model.h5') 


#test
model1=load_model('model.h5')
img=cv2.imread('download_pic/20.jpg')
img=cv2.resize(img,(224,224))

predict= model1.predict(img[np.newaxis,:,:,:])
print(np.argmax(predict))

