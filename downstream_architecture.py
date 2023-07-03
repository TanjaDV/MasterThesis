# import packages
import tensorflow as tf


def create_downstream_network(pretext_model, config, dir_experiment):
    """
    This function reads the config file and calls the corresponding function to create the downstream architecture
    """
    if 'network_downstream' not in config.keys():
        config['network_downstream'] = '2 layers'

    if config['network_downstream'] == '2 layers':
        network = create_network_2layers(pretext_model, config)
    elif config['network_downstream'] == '1 layer':
        network = create_network_1layer(pretext_model, config)
    else:
        raise ValueError("dataset '{}' is not recognized".format(config['dataset']))

    # save model architecture
    with open(dir_experiment + 'downstream_model_summary.txt', 'w') as f:
        network.summary(print_fn=lambda y: f.write(y + '\n'))

    return network


def take_pretext_layers(pretext_model, config):
    """
    This function reads the config file and calls the corresponding function to create the pretext architecture
    """
    if 'network_pretext' not in config.keys():
        config['network_pretext'] = 'nin_big'

    if config['network_pretext'] == 'nin_big':
        layers = tf.keras.layers.Flatten()(pretext_model.layers[24].output)  # end of block 2
    elif config['network_pretext'] == 'nin_small':
        layers = tf.keras.layers.Flatten()(pretext_model.layers[24].output)  # end of block 2
    elif config['network_pretext'] == 'simple_conv':
        layers = tf.keras.layers.Flatten()(pretext_model.layers[3].output)
    elif config['network_pretext'] == 'alexnet':
        layers = tf.keras.layers.Flatten()(pretext_model.layers[10].output)
    else:
        raise ValueError("pretext network '{}' is not recognized".format(config['network_pretext']))

    return layers


def create_network_2layers(pretext_model, config):
    """
    This function takes the pretext architecture and adds 2 dense layers
    """
    # compute layer size
    num_units = min(config['num_classes'] * 20, 2048)

    # take pretext layers
    x = take_pretext_layers(pretext_model, config)

    # add the classification layers
    x = tf.keras.layers.Dense(num_units, kernel_initializer=tf.keras.initializers.he_normal)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=0.00001)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Dense(num_units, kernel_initializer=tf.keras.initializers.he_normal)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=0.00001)(x)
    x = tf.keras.layers.Activation('relu')(x)

    outputs = tf.keras.layers.Dense(config['num_classes'], activation='linear')(x)
    nonlinearclassifier = tf.keras.Model(pretext_model.input, outputs)

    return nonlinearclassifier


def create_network_1layer(pretext_model, config):
    """
    This function takes the pretext architecture and adds 1 dense layer
    """
    # take pretext layers
    x = take_pretext_layers(pretext_model, config)

    # add the classification layer
    x = tf.keras.layers.Dense(100, kernel_initializer=tf.keras.initializers.he_normal)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=0.00001)(x)
    x = tf.keras.layers.Activation('relu')(x)

    outputs = tf.keras.layers.Dense(config['num_classes'], activation='linear')(x)
    nonlinearclassifier = tf.keras.Model(pretext_model.input, outputs)

    return nonlinearclassifier
