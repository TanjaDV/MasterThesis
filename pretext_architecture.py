# import packages
import tensorflow as tf


def create_pretext_network(config, num_classes, dir_experiment):
    """
    This function reads the config file and calls the corresponding function to create the architecture
    """
    input_shape = (config['resized_img_size'], config['resized_img_size'], config['num_channels'])

    if 'network_pretext' not in config.keys():
        config['network_pretext'] = 'nin_big'

    if config['network_pretext'] == 'nin_big':
        network = create_nin4(input_shape, num_classes)
    elif config['network_pretext'] == 'nin_small':
        network = create_nin3(input_shape, num_classes)
    elif config['network_pretext'] == 'simple_conv':
        network = create_simple_conv(input_shape, num_classes)
    elif config['network_pretext'] == 'alexnet':
        network = create_alexnet(input_shape, num_classes)
    else:
        raise ValueError("pretext network name '{}' is not recognized".format(config['network_pretext']))

    # save model architecture
    if dir_experiment is not None:
        with open(dir_experiment + 'pretext_model_summary.txt', 'w') as f:
            network.summary(print_fn=lambda x: f.write(x + '\n'))

    return network


def create_alexnet(input_shape, num_classes):
    """
    This function creates the architecture for alexnet
    """
    inputs = tf.keras.Input(shape=input_shape, name="inputs")

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=11, strides=4, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters=192, kernel_size=11, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=4096, activation='relu', name="dense_4096_1")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(units=4096, activation='relu', name="dense_4096_2")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax', name="dense_2")(x)
    alexnet = tf.keras.Model(inputs, outputs, name='pretext_model')

    return alexnet


def create_simple_conv(input_shape, num_classes):
    """
    This function creates the architecture for a small convolutional network
    """
    inputs = tf.keras.Input(shape=input_shape, name="inputs")

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=11, strides=4, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=11, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)

    # x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=8, activation='relu', name="dense_4096_1")(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax', name="dense_2")(x)
    simple_conv = tf.keras.Model(inputs, outputs, name='pretext_model')

    return simple_conv


def create_nin4(input_shape, num_classes):
    """
    This function creates network in netowrk with four blocks
    """
    inputs = tf.keras.Input(shape=input_shape)

    # block 1
    data = tf.keras.layers.ZeroPadding2D(padding=2)(inputs)
    data = tf.keras.layers.Conv2D(filters=192, kernel_size=5, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, input_shape=input_shape, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=160, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(data)
    data = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(data)

    # block 2
    data = tf.keras.layers.ZeroPadding2D(padding=2)(data)
    data = tf.keras.layers.Conv2D(filters=192, kernel_size=5, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.ZeroPadding2D(padding=1)(data)
    data = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2)(data)

    # block 3
    data = tf.keras.layers.ZeroPadding2D(padding=1)(data)
    data = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    # block 4
    data = tf.keras.layers.ZeroPadding2D(padding=1)(data)
    data = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    # output layers
    data = tf.keras.layers.GlobalAveragePooling2D()(data)
    outputs = tf.keras.layers.Dense(num_classes, activation='linear', )(data)
    nin = tf.keras.Model(inputs, outputs, name='pretext_model')

    return nin


def create_nin3(input_shape, num_classes):
    """
    This function creates network in network with three blocks
    """
    inputs = tf.keras.Input(shape=input_shape)

    # block 1
    data = tf.keras.layers.ZeroPadding2D(padding=2)(inputs)
    data = tf.keras.layers.Conv2D(filters=192, kernel_size=5, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, input_shape=input_shape, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=160, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(data)
    data = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(data)

    # block 2
    data = tf.keras.layers.ZeroPadding2D(padding=2)(data)
    data = tf.keras.layers.Conv2D(filters=192, kernel_size=5, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.ZeroPadding2D(padding=1)(data)
    data = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2)(data)

    # block 3
    data = tf.keras.layers.ZeroPadding2D(padding=1)(data)
    data = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_normal, )(data)
    data = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.00001, )(data)
    data = tf.keras.layers.Activation('relu')(data)

    # output layers
    data = tf.keras.layers.GlobalAveragePooling2D()(data)
    outputs = tf.keras.layers.Dense(num_classes, activation='linear', )(data)
    nin = tf.keras.Model(inputs, outputs, name='pretext_model')

    return nin
