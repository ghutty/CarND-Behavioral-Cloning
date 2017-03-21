# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

## Overview


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png "Model (NVIDIA) Visualization"
[image2]: ./images/track1.png "Track 1"
[image3]: ./images/track1_left.png "Track 1 Recovery Image"
[image4]: ./images/track1_right.png "Track 1 Recovery Image"
[image5]: ./images/track1_center.png "Track 1 Recovery Image"
[image6]: ./images/track_2.png "Track 2"
[image7]: ./images/track_2_left.png "Track 2 Recovery Image"
[image8]: ./images/track_2_right.png "Track 2 Recovery Image"
[image9]: ./images/track_2_center.png "Track 2 Recovery Image"
[image10]: ./images/model_summary.png "Model Summary"

## Required Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 showing a video car driven in autonomous mode using a CNN model
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The model.py file contains the code for training and saving the convolution neural network based on NVIDIA CNN deep learning model for self driving cars. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### An appropriate model architecture has been employed

The model started with LeNet model and then progressed.
My model consists of a convolution neural network with 5x5 anf 3x3 filter sizes and depths between 24 and 64 (model.py lines 60-64) 
The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 57). 

### Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py lines 67). 
The model was trained and validated on different data sets on 2 tracks to ensure that the model was not overfitting (code line 7-49). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 73).

### Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

## Model Architecture and Training Strategy

### Solution Design Approach

The overall strategy for deriving a model architecture was to get more data.  

  * I used both tracks (model.py lines 7-17)
  * I used all 3 cameras (model.py lines 24-46)
  * Correction of +0.2 on Left Camera and -0.2 on Right Camera Steering Measurement (model.py lines 36-38)
  * Flipped images and used a negative value of the measurements (model.py lines 42-46)

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because as a starting point for a convolution neural network.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that includes dropout.
Then I lower the epochs. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track by the lake and throguh the dirt road to improve the driving behavior in these cases, I had to add the Track 2 data set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes

![alt text][image10]


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]
Ben Firner, Beat Flepp, Karol Zieba, Larry Jackel, Mariusz Bojarski, Urs Muller
https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like starting from left to right and then center.

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.
To augment the data sat, I also flipped images and angles thinking within the code.

![alt text][image6]

These images show what a recovery looks like starting from left to right and then center on track two.

![alt text][image7]
![alt text][image8]
![alt text][image9]

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Simulation Outcome

The car is able to navigate on Track 1 with 2 laps successfully. An issue I observe the model struggled on some corners but was able to correct itself.  Overall, I am happy with the outcome and more interested on using other convolutional neural networks like VGG and GoogLeNet models.


## References:

* Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim

* Ben Firner, Beat Flepp, Karol Zieba, Larry Jackel, Mariusz Bojarski, Urs Muller NVIDIA self driving car model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

* Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backprop- agation applied to handwritten zip code recognition. Neural Computation, 1(4):541–551, Winter 1989. URL: http://yann.lecun.org/exdb/publis/pdf/lecun-89e.pdf.

* Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 25, pages 1097–1105. Curran Associates, Inc., 2012. URL: http://papers.nips.cc/paper/ 4824-imagenet-classification-with-deep-convolutional-neural-networks. pdf.

* L. D. Jackel, D. Sharman, Stenard C. E., Strom B. I., , and D Zuckert. Optical character recognition for self-service banking. AT&T Technical Journal, 74(1):16–24, 1995.

* Large scale visual recognition challenge (ILSVRC). URL: http://www.image-net.org/ challenges/LSVRC/.

* Net-Scale Technologies, Inc. Autonomous off-road vehicle control using end-to-end learning, July 2004. Final technical report. URL: http://net-scale.com/doc/net-scale-dave-report.pdf.

* Dean A. Pomerleau. ALVINN, an autonomous land vehicle in a neural network. Technical report, Carnegie Mellon University, 1989. URL: http://repository.cmu.edu/cgi/viewcontent. cgi?article=2874&context=compsci.

* Danwei Wang and Feng Qi. Trajectory planning for a four-wheel-steering vehicle. In Proceedings of the 2001 IEEE International Conference on Robotics & Automation, May 21–26 2001. URL: http: //www.ntu.edu.sg/home/edwwang/confpapers/wdwicar01.pdf.
 
