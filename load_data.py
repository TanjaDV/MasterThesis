# import packages
import numpy as np
from datetime import datetime
import os
import yaml
from sklearn import model_selection
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


def load_config_file(config_file):
    # read config file settings
    config_path = os.path.join("config\\", config_file)
    with open(config_path) as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def initialize_results_directory(config):
    today = datetime.now()
    date = today.strftime("%Y-%m-%d_%H-%M")
    dir_main_experiment = config['dir_main_results'] + date + "_" + config['pretext_task'] + "_" + \
                          config['dataset'] + "_" + config['experiment'] + "\\"
    if not os.path.exists(dir_main_experiment):
        os.makedirs(dir_main_experiment)
    return dir_main_experiment


def save_config(config, dir_main_experiment):
    with open(dir_main_experiment + "config.yaml", 'a') as file:
        yaml.dump(config, file)
    return


def load_data(config):
    """
    Calls the load function corresponding to the dataset given in configuration
    dataset_mode: Determines whether to use the test set or a validation set
    """
    if 'dataset_mode' in config.keys():
        dataset_mode = config['dataset_mode']
    else:
        dataset_mode = 'validation'

    if config['dataset'] == 'cifar10':
        train_data, val_data = load_data_cifar10(config=config, dataset_mode=dataset_mode)
    elif config['dataset'] == 'eurosat':
        train_data, val_data = load_data_eurosat(config=config, dataset_mode=dataset_mode)
    else:
        raise ValueError("dataset '{}' is not recognized".format(config['dataset']))

    # save data to config file
    config['pretext']['train_data'] = train_data
    config['pretext']['val_data'] = val_data

    config['downstream']['train_data'] = train_data
    config['downstream']['val_data'] = val_data
    return


def load_data_cifar10(config, dataset_mode):
    """
    Load and normalize data from cifar10 dataset, and split in a train, val and testset
    """
    # load dataset
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

    # normalize data
    train_normalized = train_data.astype('float32') / 255
    test_normalized = test_data.astype('float32') / 255

    # store dataset information to config file
    store_dataset_info_to_config(config=config, data=train_normalized, labels=train_labels)

    if dataset_mode == 'test':
        return (train_normalized, train_labels), (test_normalized, test_labels)
    elif dataset_mode == 'validation':
        return split_train_val(config=config, data=train_normalized, labels=train_labels)
    else:
        raise ValueError("dataset mode '{}' is not recognized".format(config['dataset_mode']))


def load_data_eurosat(config, dataset_mode):
    path = config['dir_dataset']

    label = 0
    data_list = []
    labels_list = []

    for directory in os.listdir(path):
        path_dir = os.path.join(path, directory)
        if not os.path.isdir(path_dir):
            continue
        for file in os.listdir(path_dir):
            path_file = os.path.join(path_dir, file)
            if not os.path.isfile(path_file):
                continue
            labels_list.append(label)
            img = plt.imread(path_file)
            data_list.append(img)

        label += 1

    # convert to array
    data_array = np.array(data_list)
    labels_array = np.array(labels_list)

    # normalize data
    data_normalized = data_array.astype('float32') / 255

    # store dataset information to config file
    store_dataset_info_to_config(config=config, data=data_normalized, labels=labels_array)

    if dataset_mode == 'validation':
        return split_train_val(config=config, data=data_normalized, labels=labels_array)
    elif dataset_mode == 'cross-validation':
        config['cross-val-data'] = (data_normalized, labels_array)
        return (None, None), (None, None)
    else:
        raise ValueError("dataset mode '{}' is not recognized".format(config['dataset_mode']))


def split_train_val(config, data, labels, val_split=None):
    """
    Split the given data in a training and validation set
    """
    if val_split is None:
        val_split = config['val_split']
    data_train, data_val, labels_train, labels_val = model_selection.train_test_split(
        data,
        labels,
        test_size=val_split,
        random_state=3,
        shuffle=True,
        stratify=labels
    )
    return (data_train, labels_train), (data_val, labels_val)


def store_dataset_info_to_config(config, data, labels):
    """
    Add dataset information to config
    """
    config['num_channels'] = data.shape[-1]
    config['num_classes'] = np.amax(labels) + 1
    config['original_img_size'] = data.shape[1]
    return
