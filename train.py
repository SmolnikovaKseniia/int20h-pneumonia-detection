import os
import csv
import random
import pydicom
import numpy as np

from typing import Union
from scipy.ndimage import rotate, zoom
from skimage.transform import warp, AffineTransform
from skimage.transform import resize
from tensorflow import keras
from matplotlib import pyplot as plt


pneumonia_locations = {}
PATH = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'

with open(PATH, 'r') as infile:
    READER = csv.reader(infile)
    next(READER, None)

    for row in READER:
        filename = row[0]
        location = row[1:5]
        pneumonia = row[5]
        
        if pneumonia == '1':
            location = [int(float(i)) for i in location]
            if filename in pneumonia_locations: pneumonia_locations[filename].append(location)
            else: pneumonia_locations[filename] = [location]
              
FOLDER = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images'
filenames = os.listdir(FOLDER)
random.shuffle(filenames)

VALIDATION_SAMPLES_COUNT = 2560
training_filenames = filenames[VALIDATION_SAMPLES_COUNT:]
validation_filenames = filenames[:VALIDATION_SAMPLES_COUNT]

print('Number of training samples:', len(training_filenames))
print('Number of validation samples:', len(validation_filenames))

training_samples_count = len(filenames) - VALIDATION_SAMPLES_COUNT

print('Total number of images:', len(filenames))
print('Number of images with pneumonia:', len(pneumonia_locations))

class Generator(keras.utils.Sequence):
    """
    Attributes
    ----------
    folder (str) : Path to the folder containing DICOM images

    filenames (list[str]) : List of filenames of DICOM images

    pneumonia_locations (dict) : Dictionary containing pneumonia locations for each image

    batch_size (int) : Batch size for training or prediction

    image_size (int) : Size to which the images will be resized

    shuffle (bool) : Whether to shuffle the data between epochs

    augment (bool) : Whether to apply data augmentation

    predict (bool) : Whether the generator is used for prediction or training

    Methods
    -------
    load_image_and_mask(filename) -> tuple[np.ndarray, np.ndarray] : Load and preprocess an image along with its corresponding mask

    load_image_for_prediction(filename) -> np.ndarray : Load and preprocess an image for prediction

    __getitem__(index) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, list[str]]] : Generate one batch of data

    on_epoch_end() -> None : Shuffle filenames at the end of each epoch if shuffle is set to True

    __len__() -> int : Number of batches in the Sequence
    """
   
    def __init__(self, folder: str, filenames: list[str], pneumonia_locations: dict = None, batch_size: int = 32, image_size: int = 256, shuffle: bool = True, augment: bool = False, predict: bool = False) -> None:
        
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()  # Initializing of the generator
        

    def load_image_and_mask(self, filename: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess an image along with its corresponding mask

        Parameters:
            filename (str) : Filename of the DICOM image

        Returns:
            tuple[np.ndarray, np.ndarray] : A tuple containing the preprocessed image and its corresponding mask
        """

        image = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        mask = np.zeros(image.shape)
        filename = filename.split('.')[0]

        if filename in self.pneumonia_locations:
            for location in self.pneumonia_locations[filename]:
                x, y, w, h = location
                mask[y:(y+h), x:(x+w)] = 1

        image = resize(image, (self.image_size, self.image_size), mode='reflect')
        mask = resize(mask, (self.image_size, self.image_size), mode='reflect') > 0.5

        if self.augment:
            if random.random() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            if random.random() < 0.5:
                angle = random.uniform(-0.05, 0.05) * 180 / np.pi
                image = rotate(image, angle, reshape=False, mode='reflect')
                mask = rotate(mask, angle, reshape=False, mode='reflect')

            if random.random() < 0.5:
                translationX = random.uniform(-0.1, 0.1) * image.shape[1]
                translationY = random.uniform(-0.1, 0.1) * image.shape[0]
                transform = AffineTransform(translation=(translationX, translationY))
                image = warp(image, transform, mode='reflect')
                mask = warp(mask, transform, mode='reflect')

            if random.random() < 0.5:
                scalingX = random.uniform(0.8, 1.2)
                scalingY = random.uniform(0.8, 1.2)
                image = zoom(image, (scalingY, scalingX), order=1)
                mask = zoom(mask, (scalingY, scalingX), order=0)

        image = resize(image, (self.image_size, self.image_size), mode='reflect', anti_aliasing=True)
        mask = resize(mask, (self.image_size, self.image_size), mode='reflect', anti_aliasing=False) > 0.5
        image = np.expand_dims(image, -1)
        mask = np.expand_dims(mask, -1)

        return image, mask
    
    
    def load_image_for_prediction(self, filename: str) -> np.ndarray:
        """
        Load and preprocess an image for prediction

        Parameters:
            filename (str) : Filename of the DICOM image

        Returns:
            np.ndarray : Preprocessed image for prediction
        """
        image = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        image = resize(image, (self.image_size, self.image_size), mode='reflect')
        image = np.expand_dims(image, -1)
        
        return image
        
    def __getitem__(self, index: int) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, list[str]]]:
        """
        Generate one batch of data

        Parameters:
            index (int) : Index of the batch

        Returns:
            Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, list[str]]] : Batch of images and masks or batch of images and filenames
        """
        
        filenames = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
        
        if self.predict:
            images = [self.load_image_for_prediction(filename) for filename in filenames]
            images = np.array(images)  
            
            return images, filenames
        
        else:
            data = [self.load_image_and_mask(filename) for filename in filenames]
            images, masks = zip(*data)
            images = np.array(images)
            masks = np.array(masks)
            
            return images, masks


    def on_epoch_end(self) -> None:
        """
        Shuffle filenames at the end of each epoch if shuffle is set to True
        """
        if self.shuffle: random.shuffle(self.filenames)

       
    def __len__(self) -> int:
        """
        Number of batches in the Sequence

        Returns:
            int : Number of batches
        """
        if self.predict: return int(np.ceil(len(self.filenames) / self.batch_size))
        else: return int(len(self.filenames) / self.batch_size)

  def create_downsample(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x = keras.layers.MaxPool2D(2)(x)
    return x

def create_resblock(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return keras.layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=2, depth=4):
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    for d in range(depth):
        channels = channels*2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = keras.layers.UpSampling2D(2**depth)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def iou_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    score = (intersection + 1.)/(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1-score


# сombine BCE loss function and IOU loss function
def iou_bce_loss(y_true, y_pred):
    return 0.5*keras.losses.binary_crossentropy(y_true, y_pred) + 0.5*iou_loss(y_true, y_pred)


# mean iou as metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))


model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)
model.compile(optimizer='adam', loss=iou_bce_loss, metrics=['accuracy', mean_iou])


def cosine_annealing(x):
    lr = 0.001
    epochs = 10
    return lr*(np.cos(np.pi*x/epochs)+1.)/2
learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)


train_images = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images'
train_gen = Generator(train_images, training_filenames, pneumonia_locations, batch_size=32, image_size=256, shuffle=True, augment=True, predict=False)
valid_gen = Generator(train_images, validation_filenames, pneumonia_locations, batch_size=32, image_size=256, shuffle=False, predict=False)

model = model.fit_generator(train_gen, validation_data=valid_gen, callbacks=[learning_rate], epochs=15, workers=4, use_multiprocessing=True)

save_model(model, 'model.h5')
model.save_weights('model_weights.h5')
