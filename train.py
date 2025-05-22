
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_model

data_dir = './dataset/dataset'
model_save_path = './models/cnn_model.h5'

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(512, 512),
    batch_size=8,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(512, 512),
    batch_size=8,
    class_mode='binary',
    subset='validation'
)

model = build_model()
checkpoint = ModelCheckpoint(model_save_path, save_best_only=True)
model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[checkpoint])
