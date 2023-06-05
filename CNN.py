import datetime

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomHeight, RandomWidth, RandomZoom

from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from utils.Dataloader_cnn import read_dataset
import os

#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"



def squeeze_excite_block2D(filters, input):
    se = tf.keras.layers.GlobalAveragePooling2D()(input)
    se = tf.keras.layers.Reshape((1, filters))(se)
    se = tf.keras.layers.Dense(filters // 32, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.multiply([input, se])
    return se


checkpoint_filepath = './cnn'
callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

dim = 170

test_data, test_labels = read_dataset('test', dim)
train_data, train_labels = read_dataset('train', dim)
test_data = test_data.reshape(-1, dim, dim, 1) - 0.5
train_data = train_data.reshape(-1, dim, dim, 1) - 0.5

filters = 170
s = tf.keras.Input(shape=test_data.shape[1:])
x = RandomRotation(0.1)(s)
x = RandomWidth(0.15)(x)
x = RandomHeight(0.15)(x)
x = RandomZoom(-0.2)(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = squeeze_excite_block2D(filters, x)
x = Conv2D(filters, 3, activation='relu', padding='same', dilation_rate = (1,1))(x)
x = Conv2D(filters, 3, activation='relu', padding='same', dilation_rate = (1,1))(x)
x = Conv2D(filters, 3, activation='relu', padding='same', dilation_rate = (1,1))(x)
x = BatchNormalization()(x)
x = squeeze_excite_block2D(filters, x)
x = tf.keras.layers.AveragePooling2D(2)(x)
x = Conv2D(filters, 3, activation='relu', padding='same', dilation_rate = (3,3))(x)
x = Conv2D(filters, 3, activation='relu', padding='same', dilation_rate = (3,3))(x)
x = Conv2D(filters, 3, activation='relu', padding='same', dilation_rate = (3,3))(x)
x = BatchNormalization()(x)
x = squeeze_excite_block2D(filters, x)
x = tf.keras.layers.AveragePooling2D(2)(x)
x = tf.keras.layers.concatenate([tf.keras.layers.GlobalMaxPooling2D()(x),
                                 tf.keras.layers.GlobalAveragePooling2D()(x)])
x = tf.keras.layers.Dense(15, activation='softmax', use_bias=False,
                          kernel_regularizer=tf.keras.regularizers.l1(0.00025))(x)
model = tf.keras.Model(inputs=s, outputs=x)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels, callbacks=[callback, tensorboard_callback], validation_data=(test_data, test_labels),
          epochs=100)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels, callbacks=[callback, tensorboard_callback], validation_data=(test_data, test_labels),
          epochs=70)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels, callbacks=[callback, tensorboard_callback],
          validation_data=(test_data, test_labels), epochs=30)

model.load_weights(checkpoint_filepath)

scores = model.evaluate(test_data, test_labels)
