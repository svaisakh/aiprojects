from scipy.misc import imread, imresize, imsave
import tensorflow as tf
import numpy as np
from glob import glob
from os.path import exists
from os import mkdir
from tqdm import tqdm

import keras.backend as K
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Input, Lambda, Cropping2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import Add
from keras.engine.topology import Layer
from keras.models import Model

    
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(40, 40), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, x):
        w_pad, h_pad = self.padding
        return tf.pad(x, ((0, 0), (h_pad, h_pad), (w_pad, w_pad), (0, 0),), 'REFLECT')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])
    
def conv_block(x, num_filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='conv'):
    if type(strides) is int:
        x = Conv2D(num_filters, kernel_size, strides=(
            strides, strides), padding=padding, name=name)(x)
    else:
        strides = int(1/strides)
        x = UpSampling2D((strides, strides), name=name+'_up')(x)
        x = Conv2D(num_filters, kernel_size, padding='same', name=name)(x)

    x = BatchNormalization(name=name + '_bn')(x)
    return Activation(activation, name=name + '_relu')(x) if activation is not None else x

def res_block(x, block_id=None):
    name = 'res_block'
    if block_id is not None:
        name += '_' + str(block_id)

    y = conv_block(x, padding='valid', name=name + '_conv1')
    y = conv_block(y, padding='valid', activation=None, name=name + '_conv2')
    return Add(name=name + '_add')([Cropping2D(2, name=name + '_crop')(x), y])

def get_model():
    K.clear_session()

    img_input = Input(shape=(256, 256, 3), name='img_input')
    x = ReflectionPadding2D(name='reflect')(img_input)
    x = conv_block(x, 32, 9, name='conv1')
    x = conv_block(x, 64, strides=2, name='conv2')
    x = conv_block(x, strides=2, name='conv3')
    for i in range(5):
        x = res_block(x, i + 1)
    x = conv_block(x, 64, strides=1 / 2, name='tconv1')
    x = conv_block(x, 32, strides=1 / 2, name='tconv2')
    x = conv_block(x, 3, activation='tanh', name='conv4')
    x = Lambda(lambda x: (x + 1) * 127.5)(x)

    model = Model(img_input, x)
    return model

model = get_model()

assert exists('weights.h5'), "You need to have the weights.h5 file in the same directory as this script"
model.load_weights('weights.h5')

filenames = glob('images/raw/samples/*.jpg')
print('Got {} images'.format(len(filenames)))

for filename in filenames:
        imsave(filename, norm(imread(filename), axis=2))

generator = ImageDataGenerator()
data = generator.flow_from_directory('images/raw/', shuffle=False)
data.batches_per_epoch = int(data.samples / data.batch_size)

if not exists('images/paintings/'):
    mkdir('images/paintings/')
    
print('Painting begins!')

idx = 0
for i in tqdm(range(data.batches_per_epoch+1)):
    x = data.next()[0]
    batch_size = len(x)
    y = model.predict(x)
    for i, filename in enumerate(filenames[idx:idx+batch_size]):
        imsave('images/paintings/'+filename[19:], y[i])
    idx += batch_size

print('Paintings saved in paintings folder. Enjoy!')
