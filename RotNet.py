# import packages
import tensorflow as tf
from TransformNet import TransformNet


class RotNet(TransformNet):
    """
    This model applies rotation, it inherits from the TransformNet class
    """

    def __init__(self, network, weight_decay, num_classes, *args, **kwargs):
        super().__init__(network, weight_decay, num_classes, *args, **kwargs)

    def prepare_data(self, input_data, shuffle):
        # unpack data
        data = input_data[0]
        batch_size = tf.shape(data)[0]

        # rotated x and concatenate the four (counter-clockwise) rotations
        rotated90 = tf.image.rot90(data, k=1)
        rotated180 = tf.image.rot90(data, k=2)
        rotated270 = tf.image.rot90(data, k=3)
        rotated_data = tf.concat([data, rotated90, rotated180, rotated270], axis=0)

        # define labels
        labels = tf.concat([tf.zeros(batch_size, dtype=tf.int32),
                            tf.ones(batch_size, dtype=tf.int32),
                            2 * tf.ones(batch_size, dtype=tf.int32),
                            3 * tf.ones(batch_size, dtype=tf.int32)], axis=0)

        # shuffle data and labels
        if shuffle:
            indices = tf.range(start=0, limit=batch_size * self.num_classes, dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            shuffled_data = tf.gather(rotated_data, shuffled_indices)
            shuffled_labels = tf.gather(labels, shuffled_indices)
            return shuffled_data, shuffled_labels
        else:
            return rotated_data, labels
