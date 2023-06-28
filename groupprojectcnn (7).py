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

dim=300
train_directory = '/home/pg2022/SENet_CNN/data/data/train'
test_directory = '/home/pg2022/SENet_CNN/data/data/test'

# Create separate instances of ImageDataGenerator for train and test data
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
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

def squeeze_excite_block2D(filters, input):
    se = tf.keras.layers.GlobalAveragePooling2D()(input)
    se = tf.keras.layers.Reshape((1, filters))(se)
    se = tf.keras.layers.Dense(filters // 32, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.multiply([input, se])
    return se

checkpoint = ModelCheckpoint('./best_DenseNet_model',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
checkpoint2 = ModelCheckpoint('./best_DenseNet_model',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

pretrained_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(dim, dim, 3))


for layer in pretrained_model.layers:
    layer.trainable = True

# Modify the last layer for 15 classes

x = pretrained_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(15, activation='softmax')(x)

# Create the new model
model = Model(inputs=pretrained_model.input, outputs=x)

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])



model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[checkpoint, checkpoint2]
)

model.load_weights('best_DenseNet_model')
# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])



model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=15,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[checkpoint, checkpoint2]
)



model.load_weights('best_DenseNet_model')
scores = model.evaluate(test_generator)
print (scores)
model.compile()
model.save("./fullDenseNetmodel.keras")

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

def squeeze_excite_block2D(filters, input):
    se = tf.keras.layers.GlobalAveragePooling2D()(input)
    se = tf.keras.layers.Reshape((1, filters))(se)
    se = tf.keras.layers.Dense(filters // 32, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.multiply([input, se])
    return se

checkpoint3 = ModelCheckpoint('./best_AdaptedSENet_model',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
checkpoint4 = ModelCheckpoint('./best_AdaptedSENet_model',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

saved_model_path = "./fullDenseNetmodel.keras"
pretrained_model = tf.keras.models.load_model(saved_model_path)

for layer in pretrained_model.layers:
    layer.trainable = False

# Extract the feature extraction layers

feature_extractor = pretrained_model.layers[-3].output

# Freeze the feature extraction layers
feature_extractor.trainable = False

filters=150
x = Dropout(0.9)(feature_extractor)
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
x = tf.keras.layers.concatenate([tf.keras.layers.GlobalMaxPooling2D()(x),
                                 tf.keras.layers.GlobalAveragePooling2D()(x)])
output = tf.keras.layers.Dense(15, activation='softmax')(x)

# Create the new model
model = Model(inputs=pretrained_model.input, outputs=output)
#model.summary()

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])



history1 = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[checkpoint3, checkpoint4]
)
model.load_weights('best_AdaptedSENet_model')
model.compile()
model.save("./fullAdaptedSENetNetmodel.keras")
# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])



history2 = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=70,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[checkpoint3, checkpoint4]
)
model.load_weights('best_AdaptedSENet_model')
model.compile()
model.save("./fullAdaptedSENetNetmodel.keras")
# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])



history3 = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[checkpoint3, checkpoint4]
)

model.load_weights('best_AdaptedSENet_model')
model.compile()
model.save("./fullAdaptedSENetNetmodel.keras")

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
trained_model = tf.keras.models.load_model(saved_model_path)

for layer in trained_model.layers:
    layer.trainable = True

trained_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
historyf = trained_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[checkpointf]
)
trained_model.load_weights('best_AdaptedSENet_model_tuned')
trained_model.compile()
trained_model.save("./fullAdaptedSENetNetmodelTuned.keras")

scores = model.evaluate(test_generator)
print (scores)
