## Introduction
use pure C++ to build and inference CNN frame.

Frame includes convolutional layer, pooling layer, fully connected layer, activate functions and so on.

convolutional layer uses "im2col" method learned from caffe.

MTCNN inference is implemented to test this frame.

## Performance
GPU: one P-4000

acc: 99% matched the official

speed: 5.6s per picture.

## Requirments
Opencv:2.4.1

VS 2013

cublas library
