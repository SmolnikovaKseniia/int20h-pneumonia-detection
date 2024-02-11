# Project: Pneumonia detection model
## **Description**
The model helps to find any darkening in the lungs, to determine the size of these darkenings. Determines whether the patient has pneumonia

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
 **1.** Data Preparation: Upload training data(stage_2_train_images) and copy path to folder. 
 **2.** After installing of all requirements, you need to change some directories, which contains in variables:
   - Variable '' path to stage_2_train_images, paste copied code
~~~python
~~~
   - Variable 'test_image' is the folder that contains the images based on which you want to make predictions
~~~python
test_images = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images'
~~~
