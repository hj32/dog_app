# Capstone Project - Dog Identification App

## Contents

1. [Installation](#installation)
2. [Project Motivation](#Project-Motivation)
3. [Project Definition](#Project-Definition)
4. [File Descriptions](#file-descriptions)
5. [Methodology](#Methodology)
6. [Analysis](#Analysis)
7. [Results](#results)
8. [Conclusion](#Conclusion)
9. [Licensing, Authors, and Acknowledgements](#Licensing,-Authors,-Acknowledgements)

## Installation

Libraries that are not already installed, can be installed using conda or pip

The libraries used are.

- numpy
- pandas
- random
- matplotlib
- [seaborn](https://seaborn.pydata.org/)
- [glob](https://docs.python.org/3/library/glob.html) Unix style pathname pattern expansion
- time - time utils
- [os](https://docs.python.org/3/library/os.html) - misc operating system  interfaces
- [cv2](https://github.com/opencv/opencv-python)
- [PIL](https://pypi.org/project/Pillow/) Pillow image API
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [Tensorflow](http://www.tensorflow.org)
- [keras](https://keras.io/about/)

## Project Motivation

- Develop an algorithm to detect human faces in images.
- Develop an algorithm to detect dogs in images.
- Develop an algorithm to classify different breeds of dogs.
- Demonstrate that the algorithms work using suitable examples of images to test the app.

## Project Definition

### Project Overview

Convolutional Neural Networks can be applied to the analysis of images. CNNs are regulrised multilayer perceptron. These models can be applied to the classification of images of specific subjeccts 
This project demonstrates the use of CNNs to classify images of human faces, and of Dogs. Provided with a set of images of dogs of different breeds we will build a model to identify the breed of a dog givent its image.
I want to build models that can perform the dog breed classification task as well as is possible, performance will be measured using suitable metrics.

### Problem Statement

1. Write an algorithm to detect humans in images.
2. Write an algorithm to detect dogs in images.
3. Write an algorithm ot identify a variety of different breeds of dogs. This is a difficult problem because the appearance of some breeds are almost identical.

We will use CNN models to implement solutions to these problems. Using machine leaning algorithms, and also transfer learning based on existing image sets trained for this purpose. Using these models we want to achieve model performance where the majority of classifications are correct. Ideally an accuracy score in excess of 80% would be good.

### Metrics

One strategy to measure the performance of models for a given task is to develop a model and establish a base line of performance. Further models can  be developed with the aim of surpassing the baseline performance that has been established. 
This strategy requires a single metric that can be applied across a series of models that potentially have different levels of performance. 

The neural networks used here are classifiers the metrics loss and  accuracy are used to qualify the performance of these
 CNN algorithms. Accuracy is a useful metric because it is easy to calculate, easy to interpret, and is a single number to summarize the model's capability.

The fit method used here uses the validation loss as a performance metric and stores the weights used by the model when the validation loss value is lowest. These weights are restored from the file to evaluate the test data. 
In this way the best weights(with lowest loss) are used for testing, the performance in testing represents the best model can perform.

![Step5 Loss Curve](/images/Step5_Loss.jpg "Step5 CNN Dog Breed Classifier Loss Curve")

 We need to ensure that the target classes are not severely imbalanced, imbalanced target classes within the sample data can 
 invalidate accuracy as a metric.  In this case some of the classes i.e., are represented three time as many in the dataset.
 However, since the task of classifying dogs is so difficult the errors involved make the errors concerned due to imbalanced classes insignificant.
 So accuracy has been used as a metric here.
 
 ![Histogram Class Labels](/images/Histogram_Class_Histo_h.jpg  "Histogram Class Labels")
 
 Histogram of sample of Counts of Class Labels (images of breeds of dog available in the dataset) There are 133 classes in the dataset.

## File Descriptions

Notebooks - The Jupyter notebooks are listed below

- dog_app.ipynb - Main jupyter notebook
- dog_app_2.ipnyb - Additional file to hold visualizations using data exported by dog_app.ipynb scripts
- dog_app_utils.py - utils for measuring accuracy of algorithms
- metrics_utility.py - f1_score function for metrics
- extract_bottleneck_features.py - utility for extracting bottleneck features from a file.

## Dataset

The project uses the following image sets to train the algorithm these are available in the udacity classroom environment.


- Dog images there are 8351 files in total
[dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) 
- Human images there are 13233 files in total. LFW image set.
[human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip) 
- VGG-16 bottleneck features
[VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz)


## Methodology

Data Pre-processing

Image data needs to be scaled to an appropriate  size in terms of both  x and y dimensions and also i the number of bytes used to store and manipulate if 
resources issues are to be avoided. The image data needs to be presented to the input layer of the keras model as a 4th order tensor in format (nsamples, x_dim, y,dim, channels)
By default keras will accept this format it can be changed if another format e.g. channels first. (nsamples, channels, x_dim, y_dim)
Typically we will present an algorithm as a baseline solution to a problem and then present progressive improvements, if improvements can be made.

Implementation

The dog_app.ipynb notebook is divided into 6 steps.

|Step | Algorithm|
|--------|----------------------------|
|-|-|
|Step 1 | - Human Detector algorithm    |
|Step 2 | - Dog Detector Algorithm     |
|Step 3 | - Create a CNN to Classify Dog Breeds (from Scratch)    |
|Step 4 | - Create a CNN to Classify Dog Breeds    |
|Step 5 | - Create a CNN to Classify Dog Breeds (using Transfer Learning)    |
|Step 6 | - Dog Detector Algorithm     |

I used keras Sequential models built with suitable layers for CNN networks. The data is fitted to the model using a categorical entropy lost function and Adam optimizer.  A softmax activation is used to make the classification, this is the only activation recommended for a categorical cross entropy loss function.

Refinement

During the evaluation of the models presented here I have decided to change the optimization from 'rmsprop' to 'adam' this reduces the variance of the loss and accuracy curves giving us smoother curves. The reduced variance in loss suggests that that choice of optimizer reduces the learning rate.
Having looked at the behaviour of the fscore across the different models I decided to use accuracy instead.  when the accuracy of the model is ~1% the fscore is of the order of 1E-04 so it makes sense to use accuracy across all of the models to provide a comparable performance metric. In this case keras implements categorical accuracy based on the training target classes presented as one hot encoded classes, rather than binary.


## Analysis

Data exploration

See the histogram and description of the dataset and the distribution of dog breed classes.

Data Visuzalization

I have created some visualizations and put them in the second notebook dog_app_2.ipynb. I had trouble putting everything in one notebook.
There are loss vs epoch and accuracy vs epoch plots for the models in steps 1,3,4,5.  There is a plot of fscore vs epoch for step 5.
I have plotted a confusion matrix for step 5 


## Results

Model Evaluation and Validation

The imageset data has been divided into 3 sets train, validate, test. The model has been
 scripted so that the model is trained on the train set and validated against the 
 validation set. Finally the model is tested on the test data and an accuracy score 
 is obtained for each model. Also for the dog breed identification algorithm that combines 
 the dog and human detection algorithms with the dog breed classification algorithm to make a useful application.

|Step   |Function       | Test Accuracy   |
|-------|---------------|------------|
|  1    | Detect Humans | 0.945  |
|  1    | Detect Humans Optional | 1.0 |
|  2    | Detect Dogs  |  1.0 |
|  3    | Create a CNN to Classify Dog Breeds (from Scratch) |0.0457 |
|  4    | Create a CNN to Classify Dog Breeds (using Transfer Learning) VGG-16 |0.405502 |
|  5    | Create a CNN to Classify Dog Breeds (using Transfer Learning) Resnet50 |0.837321 |
|  6    | Write Your Algorithm  |-  |
|  7    | Test Your Algorithm  | 0.8928 |

Justification

The poor performance of the from scratch model in step 3 demonstrates the difficulty of the problem.  We were able to produce a better performing model by using transfer learning using the product of deep  neural networks and significant computing power used on a large dataset, has been invested to produce the bottleneck features files used in steps 4 and 5.
The VGG-16 and Resnet50 models are documented [here](https://keras.io/api/applications/)  By using bottleneck features most of the work has already been done for us

See this [blog](https://www.medium.com) for the results of this project


## Conclusion

Creating a model that will predict the breed of a dog is a challenge. Similarities between breeds make this task difficult.
Deep CNN models can perform this task supprisingly well. Combining the dog detection algorithm with the dog breed classification algorithm to make an algoithm that can ensure the subject in an image is a dog before classification is applied enhances the overall performance by ensuring that the subject is a dog before applying the breed classification to it.

The model should perform better give more images to train on. More layers could be added to the models and the model trained for longer.
Image augmentation is another way of improving the model and making  best use of the images available.


## Licensing, Authors, Acknowledgements

The data is available under a "Creative Commons CC0 1.0 Universal (CC0 1.0) "Public Domain Dedication" license." <http://creativecommons.org/publicdomain/zero/1.0/>

Thank to [neptune AI](https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras) for  inspiration in implementing custom keras metrics

[Keras Documentation](https://keras.io/api/)
