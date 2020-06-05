# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[img_camera]: ./images/camera.jpg "Camera Images"
[img_augumented]: ./images/augumented.jpg "Augumented Images"

[img_loss]: ./images/model_loss.png "Model Loss"
[img_model]: ./images/model.png "Model"
[img_angles]: ./images/angles_hist.jpg "Angles"



## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* link to [model.h5](https://www.dropbox.com/sh/nzaoxry18h7o9ad/AADBDaXi_TntuEyfteIeVcSOa?dl=0) containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py nvmodel2-RGB.h5
```

#### 3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with five convolutional layers with 3x3 filter sizes and depths between 36 and 64. The model also has four dense layers.

The model includes RELU layers to introduce nonlinearity after each layer, and the data is normalized in the model using a Keras lambda layer. 

```pyt
def nv_model2():
    model = Sequential()

    model.add(Lambda(function=f, input_shape=input_shape))

    model.add(Convolution2D(filters=64, kernel_size=(3,3))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

    model.add(Convolution2D(filters=64, kernel_size=(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

    model.add(Convolution2D(filters=64, kernel_size=(3,3))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

    model.add(Convolution2D(filters=68, kernel_size=(3,3))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

    model.add(Convolution2D(filters=36, kernel_size=(3,3))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))
    
    return model 
```



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have small filter size in large numbers to capture the pattern in the data. 

My first step was to use a convolution neural network model similar to the Nvidia that has 5x5 filter size but initial analysis indicated surprisingly wasn't working for me and car was turning out of the way even in initial start, although loss was reduced to minimum. I changed increased the number of filters in the initial layers and reduced the filter size. I thought this model might be appropriate because large number of filters in the initial layers will provide to capture large number basic structures or patterns.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that nvidia model had low mean squared error but didn't performed on the simulation, however, making minor changes to the model had a low mean squared error on the training set and validation set. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][img_model]

#### 3. Creation of the Training Set & Training Process

I used the default dataset provided by Udacity. I separated the dataset into following three categories:

* Images from Center Camera
* Images from Left Camera
* Images from Right Camera

Images from Center Camera was first used to create training set and valid set in the ratio of 0.8: 0.2 

I then analyzed the training images (Center Camera Images) in the following three categories:

* Driving mostly straight:
  * Images with -0.15 <= angle <= 0.15 
* Driving right to left
  * Image with  angle < -0.15 
* Driving left to right
  * Image with angle > 0.15

The training dataset was highly skewed with large number of images with driving straight and less number of images with driving left or right. Since, so far I considered only center camera images, I decided to add additional images from left and right camera.

In order to add images, I sampled few numbers (more than what was required to balance the images) from original dataset and checked the steering angle. Based on the value of steering angle, the image from either left camera or right camera with angle adjustment of 0.2 was added in to training dataset as below:

* Steering angle of less than -0.15  implies the car is turning towards left, so the image from right camera with angle adjustment was added
* Steering angle of greater than 0.15 implies the car is turning towards right, so the image from left camera with angle adjustment was added

Below images shows the histogram of steering angles of the training dataset after adding additional images.  

![img_angles][img_angles]



#### Generator

In the generator, I trimmed all images so that top $ 30\% $ and below $10\%$ of image is not considered. After cropping the image, I resized the image to (64, 64, 3).

For training phase, I generated a random number between $0$ to $3$ and applied one of transformations based on the random number. The set of transformations comprised random rotation in range of $(-15,15)$, brighting image, flipping image, unchanged.

Below image represents flipping and rotation on the camera images.

![image camera][img_camera]

 

Below image shows flipping and rotation transformation applied on the few images from straight driving, left driving, right driving.

![][img_augumented]

The ideal number of epochs was 50 as evidenced by below figure showing model loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The model loss for 50 epochs has been shown except loss from initial two epochs since the loss of initial epochs was higher.

![][img_loss]



