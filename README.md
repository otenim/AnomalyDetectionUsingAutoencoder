# AnomalyDetectionUsingAutoencoder

## Overview

We tried comparing three models: (1) autoencoder, (2) deep autoencoder, and (3) convolutional autoencoder in terms of capability of anomaly detection.

In anomaly detection using autoencoders, we train an autoencoder on only normal
dataset. So, when an input data that have different features from normal dataset are fed to
the model, the corresponding reconstruction error will increase. We call such input data "abnormal data" here.

## Model Architecture

#### autoencoder
![autoencoder](https://i.imgur.com/Ccx6TAG.png)  

#### deep autoencoder
![deep_autoencoder](https://i.imgur.com/ladN1EJ.png)  

#### convolutional autoencoder
![conv_autoencoder](https://i.imgur.com/AGlKpwU.png)  

## Datasets

Details for these model architectures are written in `models.py`.  

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

1. Train an autoencoder on only normal dataset.
2. Compute losses for validation dataset that consists of normal validation dataset and
abnormal dataset.

The above procedure is to be executed for each three models.

## Result

Sample index (x-axis) 0\~499 means losses computed on 500 normal
validation samples, and 500\~999 means computed on 500 abnormal samples.
You can the losses during samples index 500\~99 increase as expected.

##### 1 training epochs

![result\_ep1](https://i.imgur.com/lrW93M0.png)  

##### 5 training epochs

![result\_ep5](https://i.imgur.com/IY54UIU.png)  

##### 10 training epochs

![result\_ep10](https://i.imgur.com/Gb69PQd.png)  

## Script

`$ python train.py [--result] [--epochs] [--batch_size] [--test_samples]`  

All the above arguments are optional.  

* `--result`: a path to result graph image. the default value is ./result.png
* `--epochs`: training epochs. the default value is 10.
* `--batch\_size`: batch size during training. the default value is 64.
* `--test\_samples`: number of validation samples for each dataset (i.e., normal validation dataset and abnormal dataset). the default value is 500.

ex) `$ python train.py --result ./result.png --epochs 10 --batch_size 64 --test_samples 500`

Dependencies are as follows.  

* Keras==2.1.4
* Tensorflow==1.4.0
* Matplotlib==2.1.2
* Numpy==1.14.0

All the above dependencies can be installed by pip command.
