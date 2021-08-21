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

numpy
pandas
matplotlib
seaborn
[glob](https://docs.python.org/3/library/glob.html)Unix style pathname pattern expansion
time - time utils
[os](https://docs.python.org/3/library/os.html) - misc operating system  interfaces
[cv2](https://github.com/opencv/opencv-python)
[PIL](https://pypi.org/project/Pillow/)Pillow
[scikit-learn](https://scikit-learn.org/stable/install.html)
[Tensorflow](http://www.tensorflow.org)
[keras](https://keras.io/about/)

## Project Motivation

Develop an algorithm to detect human faces in images.
Develop an algorithm to detect dogs in images.
Develop an algorithm to classify different breeds of dogs.
Demonstrate that the algorithms work using suitable examples of images to test the app.

## Project Definition

### Project Overview

Convolutional neural networks (CNN) can be trained to classify images of specific subjects. This project demonstrates this technology with an number of exercises that require the student to create CNN algorithms to solve individual image analysis problems. The project uses the datasets indicated bellow there are a set of human images and a set of dog images labeled by breed.

### Problem Statement

1. Write an algorithm to detect humans in images.
2. Write an algorithm to detect dogs in images.
3. Write an algorithm ot identify a variety of different breeds of dogs. This is a difficult problem because the appearance of some breeds are almos identical.

### Metrics

The neural networks used here are classifiers the metrics loss and  accuracy are used to qualify the performance of these CNN algorithms. Accuracy is a useful metric because it is easy to calculate, easy to interpret, and is a single number to summarize the model’s capability. We need to ensure that the target classes are not severely imbalanced, imbalanced target classes within the sample data can invalidate accuracy as a metric.

## File Descriptions

Notebooks - The Jupyter notebooks are listed below

dog_app.ipynb

## DataSet

The project uses the following image sets to train the algorithm these are available in the udacity classroom environment.

Human images there are 13233 files in total. LFW imageset.

Dog images there are 8351 files in total

## Methodology

## Analysis

## Results

See this [blog](https://www.medium.com) for the results of this project

## Conclusion

## Licensing, Authors, Acknowledgements

The data is available under a "Creative Commons CC0 1.0 Universal (CC0 1.0) "Public Domain Dedication" license." <http://creativecommons.org/publicdomain/zero/1.0/>


