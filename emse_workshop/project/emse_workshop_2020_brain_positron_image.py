# -*- coding: utf-8 -*-
"""EMSE - Workshop 2020 - Brain Positron Image.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jI2vFFDD_kiolt3UhLY9WnxMo6yyKN60
"""

from os.path import join
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

# load an image
dir_path = join('ADNI', 'NC')
imgs = list(listdir(dir_path))
file_name = imgs[np.random.randint(len(imgs))]
IMAGE_LOCATION = join(dir_path, file_name)

# open the image
IMG = nib.load(IMAGE_LOCATION)

# get the matrix of voxels
FDATA = IMG.get_fdata()
# create axial, sagital coronal arrays
IMG_SIZE = IMG.shape
print(file_name)
print('Image shape: {:s}'.format(str(IMG_SIZE)))
FIX_D1 = int(IMG_SIZE[0]/3)
FIX_D2 = int(IMG_SIZE[1]/2)
FIX_D3 = int(IMG_SIZE[2]/2)

AXIAL = FDATA[:, :, FIX_D3]
CORONAL = FDATA[:, FIX_D2, :]
SAGITAL = FDATA[FIX_D1, :, :]

# display axial, sagital coronal views
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(np.flip(np.transpose(np.flip(AXIAL, 1)), 1), cmap='hot')
plt.title('Axial')
plt.colorbar(fraction=0.036)

plt.subplot(132)
plt.imshow(np.flip(np.transpose(np.flip(CORONAL, 1)), 1), cmap='hot')
plt.title('Coronal')
plt.colorbar(fraction=0.036)

plt.subplot(133)
plt.imshow(np.flip(np.transpose(np.flip(SAGITAL, 1)), 1), cmap='hot')
plt.title('Sagital')
plt.colorbar(fraction=0.036)
plt.tight_layout()

plt.show()

from os import listdir
from os.path import join
from csv import DictReader
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

pct_val = .2
# classes = ['AD', 'NC', 'sMCI', 'pMCI']
classes = ['AD', 'NC']
cfg = dict()
for ds in classes:
  cfg[ds] = list(listdir(join('ADNI', ds)))
  perm = np.random.permutation(len(cfg[ds]))
  sep = int(np.ceil(len(cfg[ds]) * pct_val))
  cfg['%s_va' % ds] = perm[:sep]
  cfg['%s_tr' % ds] = perm[sep:]


def data_generator(cfg, ds, batch):
  nbc = len(classes)
  db, dby, pos_b = np.zeros((batch, 91, 109, 91, 1)), np.zeros((batch, nbc)), 0
  cl, pos_cl = 0, list()
  pos_cl = [cfg['%s_%s' % (cl, ds)].shape[0] for cl in classes]
  perm_cl = [None for cl in classes]
  while True:
    d = cfg['%s_%s' % (classes[cl], ds)]
    if pos_cl[cl] == d.shape[0]:
      pos_cl[cl] = 0
      perm_cl[cl] = np.random.permutation(d.shape[0])
    fn = join('ADNI', classes[cl], cfg[classes[cl]][d[perm_cl[cl][pos_cl[cl]]]])
    pos_cl[cl] += 1

    img = np.expand_dims(nib.load(fn).get_fdata(), -1)
    db[pos_b], dby[pos_b, cl] = img, 1
    cl = (cl + 1) % nbc
    pos_b += 1
    if pos_b == batch:
      yield db, dby
      pos_b, dby = 0, np.zeros((batch, nbc))


def train_model(cfg, epoch=10, lr=1e-3, batch=2):
  # -- Preparatory code --
# Model configuration
  
  """ Create, compile and train a model
  # Arguments
    :param cfg: dict, generator parameters
    :param epoch: int, epoch
    :param lr: float, learning rate
    :param batch: int, batch (train & valid)
   # Returns
    :return: trained model
  """
  g_tr = data_generator(cfg, 'tr', batch)
  g_va = data_generator(cfg, 'va', batch)
  s_tr = np.sum(np.array([cfg['%s_tr' % cl].shape[0] for cl in classes]))
  s_tr = int(np.ceil(s_tr / batch))
  s_va = np.sum(np.array([cfg['%s_va' % cl].shape[0] for cl in classes]))
  s_va = int(np.ceil(s_va / batch))

  model = create_model(IMG_SIZE + (1,), len(classes))
  model.summary()
  from keras.optimizers import adam
  opt = adam(lr=0.001, decay=1e-6)

  model.compile(

      optimizer=opt,
      loss=tf.keras.losses.categorical_crossentropy,
      metrics=['categorical_accuracy']
  )

  cb = list()
  cb.append(tf.keras.callbacks.CSVLogger(join('log.csv')))

  model.fit_generator(
      generator=g_tr, steps_per_epoch=s_tr,
      validation_data=g_va, validation_steps=s_va,
      epochs=epoch, verbose=1, callbacks=cb
  )
  return model


def plot_log(filename, show=None):
    """ Plot log of training / validation learning curve
    # Arguments
        :param filename: str, csv log file name
        :param show: None / str, show graph if none or save to 'show' directory
    """
    # Load csv file
    keys, values, idx = [], [], None
    with open(filename, 'r') as f:
        reader = DictReader(f)
        for row in reader:
            if len(keys) == 0:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                idx = keys.index('epoch')
                continue
            for _, value in row.items():
                values.append(float(value))
        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:, idx] += 1
    # Plot
    fig = plt.figure(figsize=(4, 6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        # training loss
        if key.find('loss') >= 0:   # and not key.find('val') >= 0:
            plt.plot(values[:, idx], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')
    fig.add_subplot(212)
    for i, key in enumerate(keys):
        # acc
        if key.find('acc') >= 0:
            plt.plot(values[:, idx], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')
    if show is not None:
        fig.savefig(join(show, 'log.png'))
    else:
        plt.show()

def create_model(input_shape, nbc):
  from keras.models import Sequential
  from keras.layers import Dense, Flatten,Conv3D,MaxPooling3D
  '''
  lin=tf.keras.layers.Input(shape=input_shape)
  lact=lin
  lact=tf.keras.layers.Conv3D(16,3,use_bias=False)(lact)
  lact=tf.keras.layers.BatchNormalization()(lact)
  lact=tf.keras.layers.Activation('relu')(lact)
  lact=tf.keras.layers.SpatialDropout1D(0.2)
  lact=tf.keras.layers.GlobalMaxPooling3D()(lact)
  #?,16 =une information par filtre-->16filtes:donc 16 info

  lout=tf.keras.layers.Dense(nbc,activation='softmax')(lact)
  model=tf.keras.model.Model(lin,lout)
  return model

  '''

  model=Sequential()
  model.add(Conv3D(32, kernel_size=(3,3,3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))#32 est le nombre de filtres 
  model.add(MaxPooling3D(pool_size=(2, 2, 2)))
  model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
  model.add(MaxPooling3D(pool_size=(2, 2, 2)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(nbc, activation='softmax'))
  return model
   # model = create_model(IMG_SIZE + (1,), len(classes))
 
  

model = train_model(cfg, epoch=100, lr=1e-3, batch=2)
plot_log(join('log.csv'), show=None)