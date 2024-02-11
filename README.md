# Project: Pneumonia detection model

## **Description**

This repository contains code for training a pneumonia detection model using chest X-ray images and based on this model use inference.py to to find any darkening in the lungs, to determine the size of these darkenings.

### Data Preparation
- The dataset used is the RSNA Pneumonia Detection Challenge dataset, available on Kaggle.
- The `stage_2_train_labels.csv` file contains pneumonia location information.
- Images are preprocessed and loaded using a custom data generator.

### Model Architecture
- The model architecture consists of a convolutional neural network for semantic segmentation.
- Downsample blocks and residual blocks are used to extract features.
- The network is trained using a combination of IOU loss and binary cross-entropy loss.

### Training
- The model is trained using the Adam optimizer with cosine annealing learning rate scheduler.
- Training and validation data are fed using a custom data generator.
- The model is evaluated based on accuracy and mean IOU metrics.

### Files
- `train.py`: Script for training the model.
- `inference.py`: Script for making predictions using the trained model.
- `requirements.txt`: List of required Python libraries.
- `submission.csv`: Sample submission file.
- `model.h5`: Trained model architecture.
- `model_weights.h5`: Trained model weights.

## **Installation**
**To install this project, you can follow these steps:**

**1.** Clone the repository:
 Copy code and write it in terminal: 
 ~~~terminal
git clone https://github.com/SmolnikovaKseniia/int20h-pneumonia-detection
~~~
 
**2.** Navigate to the project directory:
 Write down a code in terminal: 
 ~~~terminal
cd int20h-pneumonia-detection
~~~

**3.** Install dependencies: 
~~~terminal
pip install -r requirements.txt
~~~

## **Usage**
**To use this project, you can follow these steps:**
 **1.** Data Preparation: Upload training data(folder stage_2_train_images and folder stage_2_train_labels.csv) and copy path to folder and to the file. 
 **2.** After installing of all requirements, you need to change some directories, which contains in variables:
   - Variable 'FOLDER'(train.py, 32 line) path to stage_2_train_images, paste copied code
~~~python
FOLDER = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images'
~~~
   - Variable 'test_image'(inference.py, 6 line) is the folder that contains the images based on which you want to make predictions
~~~python
test_images = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images'
~~~
   - Variable 'PATH'(train.py, 16 line) path to stage_2_train_images, paste copied code
~~~python
PATH = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
~~~

## **Support**
If you have some questions or need some help, contact me :
**smonikova.ksenia@gmail.com**
