# AnomalyDetectionUsingAutoencoder

## Overview

We tried comparing three models: (1) autoencoder, (2) deep autoencoder, and (3) convolutional autoencoder in terms of capability of anomaly detection.

In anomaly detection using autoencoders, we train an autoencoder on only normal
dataset. So, when an input data that has different features from normal dataset are fed to
the model, the corresponding reconstruction error will increase. We call such input data "abnormal data" here.

## Model Architecture

#### autoencoder
![autoencoder](https://i.imgur.com/Ccx6TAG.png)  

#### deep autoencoder
![deep_autoencoder](https://i.imgur.com/ladN1EJ.png)  

#### convolutional autoencoder
![conv_autoencoder](https://i.imgur.com/AGlKpwU.png)  

## Datasets

**Normal dataset**  

![mnist](https://i.imgur.com/ia2Cqxf.png)  

* 28\*28\*1 gray-scale images
* training samples: 60,000
* validation samples: 500 (randomly sampled from the original 10,000 samples)

**Abnormal dataset**  

![fashion](https://i.imgur.com/NhjuFnx.png)  

* 28\*28\*1 gray-scale images
* validation samples: 500 (randomly sampled from the original 10,000 samples)

## Evaluation Procedure

1. Train an autoencoder on only normal training samples.
2. Compute losses for validation samples that consists of normal validation samples and
abnormal samples. validation_samples[0\~499] corresponds to the former dataset and
validation_samples[500\~999] corresponds to the latter. 1,000 samples in total.

The above procedure is executed for each three models.

## Result

As described in the last section, sample index 0\~499 means losses computed on 500 normal
validation samples, and 500\~999 means computed on 500 abnormal samples.

##### 1 training epochs

![result_ep1](https://i.imgur.com/HB0Yaa6.png)  

##### 5 training epochs

![result_ep5]()  

##### 10 training epochs

![result_ep10]()
