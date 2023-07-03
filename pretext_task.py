# import packages
import tensorflow as tf

# import my functions
from RotNet import RotNet
from ScaleNet import ScaleNet
from pretext_architecture import create_pretext_network
from my_callbacks import RecordMetrics, learning_rate_scheduler


def create_pretext_model(config, learning_rate, momentum, weight_decay, dir_experiment=None):
    """
    This function creates and compiles the pretext model
    """
    # create model
    if config['pretext_task'] == 'rotation':
        network = create_pretext_network(config, num_classes=4, dir_experiment=dir_experiment)
        model = RotNet(network, weight_decay, num_classes=4)
    elif config['pretext_task'] == 'scale':
        network = create_pretext_network(config, num_classes=4, dir_experiment=dir_experiment)
        model = ScaleNet(network, weight_decay, resized_img_size=config['resized_img_size'], num_classes=4)
    else:
        raise ValueError("pretext task '{}' is not recognized".format(config['pretext_task']))

    # set optimizer and loss
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # compile model
    model.compile(optimizer=optimizer, loss=scce)

    return model


def train_pretext_model(model, config, dir_settings, title="pretext_history"):
    """
    This function trains the pretext model
    """
    if 'save_pretext_weights' not in config.keys():
        config['save_pretext_weights'] = 'best'

    dir_settings.set_dir_weights(config['save_pretext_weights'])
    if config['save_pretext_weights'] == 'best':
        filepath = dir_settings.dir_weights + "pretext_weights.hdf5"
        save_best_only = True
    elif config['save_pretext_weights'] == 'all':
        filepath = dir_settings.dir_weights + "pretext_weights.{epoch}.hdf5"
        save_best_only = False
    else:
        raise ValueError("save_pretext_weights mode '{}' is not recognized".format(config['save_pretext_weights']))

    history = model.fit(
        x=config['pretext']['train_data'][0],
        y=config['pretext']['train_data'][1],
        validation_data=config['pretext']['val_data'],
        epochs=config['pretext_epochs'],
        batch_size=config['batch_size'],
        verbose=1,
        callbacks=[
            RecordMetrics(dir_settings.dir_results, title),
            tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.002, mode='min', patience=35, verbose=0),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                verbose=0,
                save_best_only=save_best_only,
                save_weights_only=True,
                save_freq='epoch')
        ]
    )
    return history
