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
- seaborn
- [glob](https://docs.python.org/3/library/glob.html)Unix style pathname pattern expansion
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

### Problem Statement

1. Write an algorithm to detect humans in images.
2. Write an algorithm to detect dogs in images.
3. Write an algorithm ot identify a variety of different breeds of dogs. This is a difficult problem because the appearance of some breeds are almost identical.

### Metrics

The neural networks used here are classifiers the metrics loss and  accuracy are used to qualify the performance of these
 CNN algorithms. Accuracy is a useful metric because it is easy to calculate, easy to interpret, and is a single number to
 summarize the model’s capability.

 We need to ensure that the target classes are not severely imbalanced, imbalanced target classes within the sample data can 
 invalidate accuracy as a metric.  In this case some of the classes i.e. are representd three time as many in the dataset.
 However since the task of classifying dogs is so difficult the errors involved make the errors concerned due to imbalanced classes insignificant.
 So accuracy has been used as a metric here.
 
 ![Histogram Class Labels](/images/Histogram_Class_Labels60.jpg  "Histogram Class Labels")
 
 Histogram of Counts of Class Labels (images of breeds of dog available in the dataset)

## File Descriptions

Notebooks - The Jupyter notebooks are listed below

- dog_app.ipynb - Main jupyter notebook
- dog_app_2.ipnyb - Additional file 
- dog_app_utils.py - utils for measuring accuracy of algorithms
- metrics_utility.py - f1_score function for metrics

## DataSet

The project uses the following image sets to train the algorithm these are available in the udacity classroom environment.



- Dog images there are 8351 files in total
[dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) 
- Human images there are 13233 files in total. LFW imageset.
[human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip) 
- VGG-16 bottleneck features
[VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz)


## Methodology




## Analysis



## Results

|Step   |Function       | Test Accuracy   |
|-------|---------------|------------|
|  1    | Detect Humans | 0.945  |
|  1    | Detect Humans Optional | 0.995   |
|  2    | Detect Dogs  |  1.0 |
|  3    | Create a CNN to Classify Dog Breeds (from Scratch) |0.011   |
|  4    | Create a CNN to Classify Dog Breeds (using Transfer Learning) |0.3887   |
|  5    | Create a CNN to Classify Dog Breeds (using Transfer Learning) |0.817    |
|  6    | Write Your Algorithm  |-  |
|  7    | Test Your Algorithm  | 0.89        |

See this [blog](https://www.medium.com) for the results of this project

## Conclusion

## Licensing, Authors, Acknowledgements

The data is available under a "Creative Commons CC0 1.0 Universal (CC0 1.0) "Public Domain Dedication" license." <http://creativecommons.org/publicdomain/zero/1.0/>

Thank to [neptune AI](https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras) for  inspiration in implementing custom keras metrics
