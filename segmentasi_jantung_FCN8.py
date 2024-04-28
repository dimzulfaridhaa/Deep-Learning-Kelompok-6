import os
import cv2
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Setting up the folder path
dataset_path = 'dataset'

# Resize images
SIZE_X = 224
SIZE_Y = 224

# Read and preprocess training images
train_images = []
for subfolder_name in ['es_training', 'ed_training']:
    subfolder_path = os.path.join(dataset_path, "image", subfolder_name)
    for img_path in glob.glob(os.path.join(subfolder_path, "*.jpg")):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE_X, SIZE_Y))
            train_images.append(img)
        except Exception as e:
            print("Error reading image:", e)

train_images = np.array(train_images)
print("Number of training images:", len(train_images))

# Read and preprocess training masks
train_masks = []
class_ranges = [
    ([245, 245, 245], [255, 255, 255]),  # Class 1 (white)
    ([185, 185, 185], [215, 215, 215]),  # Class 2 (light gray)
    ([95, 95, 95], [115, 115, 115]),     # Class 3 (darker gray)
    ([0, 0, 0], [0, 0, 0])
]

for subfolder_name in ['es_training_gt', 'ed_training_gt']:
    subfolder_path = os.path.join(dataset_path, "mask", subfolder_name)
    for mask_path in glob.glob(os.path.join(subfolder_path, "*.jpg")):
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            mask = cv2.resize(mask, (SIZE_X, SIZE_Y))

            class_masks = []
            for (lower, upper) in class_ranges:
                mask_class = cv2.inRange(mask, np.array(lower), np.array(upper))
                mask_class = (mask_class / 255).astype(np.uint8)
                class_masks.append(mask_class)

            multiclass_mask = np.stack(class_masks, axis=-1)
            train_masks.append(multiclass_mask)
        except Exception as e:
            print("Error processing mask:", e)

train_masks = np.array(train_masks)
print("Number of training masks:", len(train_masks))

# Preprocess input
train_images = train_images / 255.0
train_masks = train_masks / 255.0

# Convert masks to float32
train_masks = train_masks.astype('float32')

def Unet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

model = Unet(input_shape=(SIZE_X, SIZE_Y, 3), num_classes=4)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(train_images,
                    train_masks,
                    batch_size=8,
                    epochs=10,
                    verbose=1)

# Plot Loss and Accuracy Graph
loss = history.history['loss']
acc = history.history['accuracy']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, acc, 'r', label='Training Accuracy')

plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()

plt.show()

# Save model
model.save('model_segmentasi_jantung_unet.h5')