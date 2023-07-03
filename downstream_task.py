# import packages
import tensorflow as tf

# import my functions
from my_callbacks import RecordMetrics, learning_rate_scheduler
from downstream_architecture import create_downstream_network
from DownstreamModel import DownstreamModel


def create_downstream_model(model, config, learning_rate, momentum, weight_decay, dir_experiment, trainable=False):
    """
    This function creates and compiles the downstream task model
    """
    # Take pretext model and make it untrainable
    pretext_model = model.get_layer('pretext_model')
    if not trainable:
        pretext_model.trainable = False

    downstream_network = create_downstream_network(pretext_model, config, dir_experiment)
    downstream_model = DownstreamModel(downstream_network, config['num_classes'],
                                       config['resized_img_size'], weight_decay)

    # set optimizer and loss
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # compile model
    downstream_model.compile(optimizer=optimizer, loss=scce,
                             metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')])
    return downstream_model


def train_downstream_model(model, config, dir_settings, title="downstream_history"):
    """
    This function trains the downstream task model
    """
    dir_settings.set_dir_weights('best')
    filepath = dir_settings.dir_weights + "downstream_weights.hdf5"

    history = model.fit(
        x=config['downstream']['train_data'][0],
        y=config['downstream']['train_data'][1],
        validation_data=config['downstream']['val_data'],
        epochs=config['downstream_epochs'],
        batch_size=config['batch_size'],
        verbose=1,
        callbacks=[
            RecordMetrics(dir_settings.dir_results, title),
            tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler),
            tf.keras.callbacks.TerminateOnNaN(),
            # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.002, mode='min', patience=35, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                save_weights_only=True,
                save_freq='epoch')
        ]
    )
    return history
