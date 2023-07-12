# -*- coding: utf-8 -*-

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import tensorflow as tf
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Conv2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomHeight, RandomWidth, RandomZoom
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tf_bi_tempered_loss import BiTemperedLogisticLoss
from tf_bi_tempered_loss import tempered_softmax



dim=300
train_directory = '/home/pg2022/SENet_CNN/data/data/train'
test_directory = '/home/pg2022/SENet_CNN/data/data/test'
# Create separate instances of ImageDataGenerator for train and test data
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=25,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Use the generators to load and preprocess train and test images
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(dim, dim),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(dim, dim),
    batch_size=32,
    class_mode='categorical'
)

pretrained_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(dim, dim, 3))


for layer in pretrained_model.layers:
    layer.trainable = True


def squeeze_excite_block2D(filters, input):
    se = tf.keras.layers.GlobalAveragePooling2D()(input)
    se = tf.keras.layers.Reshape((1, filters))(se)
    se = tf.keras.layers.Dense(filters // 32, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.multiply([input, se])
    return se
checkpoint = ModelCheckpoint('./best_AdaptedSENet_model',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

for layer in pretrained_model.layers:
    layer.trainable = False

# Extract the feature extraction layers

feature_extractor = pretrained_model.layers[-400].output

# Freeze the feature extraction layers
feature_extractor.trainable = False

filters=300
x = Dropout(0.8)(feature_extractor)
x = BatchNormalization()(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = Dropout(0.8)(x)
x = BatchNormalization()(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = Dropout(0.8)(x)
x = BatchNormalization()(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = Conv2D(filters, 3, activation='relu', padding='same')(x)
x = Dropout(0.8)(x)
x = BatchNormalization()(x)
x = squeeze_excite_block2D(filters, x)
x = Dropout(0.5)(x)
#x = tf.keras.layers.concatenate([tf.keras.layers.GlobalMaxPooling2D()(x),
#                                tf.keras.layers.GlobalAveragePooling2D()(x)])
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dense(1000, activation='relu')(x)
output = tf.keras.layers.Dense(15, activation='softmax')(x)

# Create the new model
model = Model(inputs=pretrained_model.input, outputs=output)
#model.summary()

model.compile(optimizer='adam', loss=BiTemperedLogisticLoss(t1=1, t2=1), metrics=['accuracy'])



history1 = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=200,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[checkpoint]
)
model.load_weights('best_AdaptedSENet_model')
model.compile()
model.save("./fullAdaptedSENetNetmodel.keras")
scores = model.evaluate(test_generator)
print (scores)
# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),loss=BiTemperedLogisticLoss(t1=0.85, t2=1.15),metrics=['accuracy'])



history2 = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[checkpoint]
)
model.load_weights('best_AdaptedSENet_model')
model.compile()
model.save("./fullAdaptedSENetNetmodel.keras")
scores = model.evaluate(test_generator)
print (scores)
# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),loss=BiTemperedLogisticLoss(t1=0.9, t2=1.05),metrics=['accuracy'])



history3 = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[checkpoint]
)


model.load_weights('best_AdaptedSENet_model')
scores = model.evaluate(test_generator)
print (scores)
model.compile()
model.save("./fullAdaptedSENetNetmodel.keras")

checkpointf = ModelCheckpoint('./best_AdaptedSENet_model_tuned',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

saved_model_path = "./fullAdaptedSENetNetmodel.kerass"
model.load_weights('best_AdaptedSENet_model')
scores = model.evaluate(test_generator)
print (scores)
