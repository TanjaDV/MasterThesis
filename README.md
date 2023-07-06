# This repository contains the code for the thesis project of Tanja

### Abstract thesis

We made a self-supervised model with a pretext-task based on geometric transformations.
Our model is based on RotNet, a model that predicts image rotations by Gidaris et al. (https://arxiv.org/pdf/1803.07728.pdf).
We implemented our model in TensorFlow, and were able to reproduce their results for rotation.
The rotation model cannot be used for all datasets, for example datasets with a lot of top-down images or images of round objects.
Therefore, we modified our network to predict the scale of an image.
In order for our scale model to work, the used dataset has to have a defined scale, that is, all images are taken from the same distance.
We see the same patterns that show the pretext task learns some useful features.
However, we also see that for most experiments the gap with supervised is bigger for our scale model than for our rotation model. 
Most promising is that our scale performs as good as supervised learning for our dataset with a low amount of labelled data.
### Requirements

This project is developed with tensorflow version 2.3.1.

### Datasets

We use CIFAR-10 for RotNet and EuroSAT for ScaleNet.

- If you run an experiment with the EuroSAT dataset, you should download this from
  http://madm.dfki.de/downloads.
    - You can find it under the header Datasets for Machine Learning
    - We use the rgb version
- The CIFAR-10 dataset is loaded from within tensorflow.

### Configuration file

To run an experiment you have to provide a configuration file.

- We provide standard configuration files for most experiments shown in the thesis.
- For both CIFAR-10 and EuroSAT an example configuration file is provided with the usual settings,
  and more explainations on what information is needed for each experiment as instruction to make your own
  configuration.
- The configuration files are expected to be in the ./config directory.
- If you use one of our configuration files, you must adapt the directories to match your own pc paths.

### Code structure

The implementation is module based. This means we can easily add something extra,
for example an experiment, a dataset, a network or a transformation.
A few of the more important files are discussed here.

#### experiments.py

This file contains the implementation of all the different experiments.

- parameter_experiment() is the base function that goes over all the hyperparameters.
- training_val_model() translates the configuration file settings to the corresponding training function.

#### TransformNet.py

This class inherits from tf.keras.Model and trains the pretext-task.
It contains the base for our transformation models.
Both RotNet and Scalenet inherit from TransformNet.

#### DownstreamModel.py

This class also inherits from tf.keras.Model and trains the downstream task.