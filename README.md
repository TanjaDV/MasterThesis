# This repository contains the code for the thesis project of Tanja

### Abstract thesis

todo: add abstract

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