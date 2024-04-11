# import necessary libraries
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

base_model = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), # Base model for image classification
                                               include_top=False,
                                               weights='imagenet')

# Freeze all layers in the base model
base_model.trainable = False
# Unfreezing last 8 layers for training
for layer in base_model.layers[-5:]:
    layer.trainable = True

# Model architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.0005))(x)

x = Dropout(0.75)(x)   # Dropout to reduce overfitting
predictions = Dense(1, activation='sigmoid')(x)  

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Setup data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range = [.8,1.2],
    validation_split=0.2)  # Using 20% of the data for validation

train_generator = train_datagen.flow_from_directory(
    '/home/jetson/Documents/image',  # Dataset
    target_size=(192, 192),
    batch_size=64,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    '/home/jetson/Documents/image',  # Dataset
    target_size=(192, 192),
    batch_size=64,
    class_mode='binary',
    subset='validation')

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=5,
)
# Save the model
model.save('distracted_driver_detection_model.keras')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# plotting model output data for visualization
epochs_range = range(len(acc))  

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='center right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='center right')
plt.title('Training and Validation Loss')

plt.show()