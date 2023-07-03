# import packages
import tensorflow as tf


class RecordMetrics(tf.keras.callbacks.Callback):
    """
    This class is used to record the loss and accuracy per epoch
    """

    def __init__(self, dir_results, title):
        super().__init__()
        self.history_file = dir_results + title + ".txt"

    def on_epoch_end(self, epoch, logs=None):
        with open(self.history_file, 'a') as f:
            f.write("Epoch {}:  loss = {:1.4f},  acc = {:1.4f},  val_loss = {:1.4f},  val_acc = {:1.4f} \n"
                    .format(epoch, logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc']))


class RecordMetricsEvaluation(tf.keras.callbacks.Callback):
    def __init__(self, dir_results, title):
        super().__init__()
        self.history_file = dir_results + title + ".txt"

    def on_test_end(self, logs=None):
        with open(self.history_file, 'a') as f:
            f.write("loss = {:1.4f},  acc = {:1.4f} \n".format(logs['loss'], logs['acc']))


def learning_rate_scheduler(epoch, lr):
    """
    This function is used to reduce the learning rate at set epochs
    """
    if epoch == 30 or epoch == 60 or epoch == 80:
        return lr / 5
    else:
        return lr
