# import packages
import tensorflow as tf


class DownstreamModel(tf.keras.Model):
    def __init__(self, network, num_classes, resized_img_size, weight_decay, *args, **kwargs):
        super(DownstreamModel, self).__init__(*args, **kwargs)
        self.network = network
        self.num_classes = num_classes
        self.resized_img_size = resized_img_size
        self.weight_decay = weight_decay
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
        train_data, train_labels = self.__prepare_data(input_data)

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
        # unpack and rescale data
        input_data = input_data[0]
        data, labels = self.__prepare_data(input_data)

        # predict the class
        y_pred = self(data, training=False)

        # reshape to one-dimensional arrays
        y_pred_1d = tf.argmax(y_pred, axis=1)

        return {'y_pred': y_pred_1d, 'labels': labels}

    def test_step(self, input_data):
        # unpack and rescale data
        test_data, test_labels = self.__prepare_data(input_data)

        # predict the class
        y_pred = self(test_data, training=False)

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

    def __prepare_data(self, input_data):
        data, labels = input_data
        labels_1d = tf.reshape(labels, shape=(tf.shape(labels)[0],))

        # rescale images
        data_rescaled = tf.image.resize(data, [self.resized_img_size, self.resized_img_size])

        return data_rescaled, labels_1d
