# import packages
from abc import ABC, abstractmethod
import tensorflow as tf


class TransformNet(tf.keras.Model):
    """
    This model is our base class, it applies the transformation given in prepare_data
    """

    def __init__(self, network, weight_decay, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = network
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.train_loss_tracker = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss_tracker = tf.keras.metrics.Mean(name='test_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_acc')

    def call(self, inputs, *args, **kwargs):
        return self.network(inputs, *args, **kwargs)

    @property
    def metrics(self):
        # The metrics listed here are reset automatically at the start of each epoch
        # and at the start of evaluate()
        return [self.test_loss_tracker, self.train_loss_tracker, self.train_accuracy, self.test_accuracy]

    def train_step(self, input_data):
        # unpack and transform data
        train_data, train_labels = self.prepare_data(input_data, shuffle=True)

        # track gradients for back-propagation
        with tf.GradientTape() as tape:
            y_pred = self(train_data, training=True)
            loss = self.compiled_loss(train_labels, y_pred, regularization_losses=self.losses)
            # add weighted l2 loss (weight decay)
            l2loss = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])
            loss = loss + self.weight_decay * l2loss

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # update loss tracker
        self.train_loss_tracker.update_state(loss)

        # compute accuracy, not used for training
        train_cat_labels = tf.one_hot(train_labels, self.num_classes)
        self.train_accuracy.update_state(train_cat_labels, y_pred)

        return {'loss': self.train_loss_tracker.result(), 'acc': self.train_accuracy.result()}

    def predict_step(self, input_data):
        # unpack and transform data
        input_data = input_data[0]
        data, labels = self.prepare_data(input_data, shuffle=False)

        # predict the transformation
        y_pred = self.network(data, training=False)

        # reshape to one-dimensional arrays
        y_pred_1d = tf.argmax(y_pred, axis=1)
        labels_1d = tf.reshape(labels, shape=(tf.shape(labels)[0],))

        return {'y_pred': y_pred_1d, 'labels': labels_1d}

    def test_step(self, input_data):
        # unpack and transform data
        test_data, test_labels = self.prepare_data(input_data, shuffle=False)

        # predict the transformation
        y_pred = self.network(test_data, training=False)

        # Updates stateful loss metrics.
        loss = self.compiled_loss(test_labels, y_pred, regularization_losses=self.losses)
        # add weighted l2 loss (weight decay)
        l2loss = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])
        loss = loss + self.weight_decay * l2loss

        # update loss tracker
        self.test_loss_tracker.update_state(loss)

        # compute accuracy, not used for training
        test_cat_labels = tf.one_hot(test_labels, self.num_classes)
        self.test_accuracy.update_state(test_cat_labels, y_pred)

        return {'loss': self.test_loss_tracker.result(), 'acc': self.test_accuracy.result()}

    @abstractmethod
    def prepare_data(self, input_data, shuffle):
        pass
