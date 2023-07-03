# import packages
import tensorflow as tf
from TransformNet import TransformNet


class ScaleNet(TransformNet):
    """
    This model applies scale, it inherits from TransformNet
    """

    def __init__(self, network, weight_decay, num_classes, resized_img_size, *args, **kwargs):
        super().__init__(network, weight_decay, num_classes, *args, **kwargs)
        self.img_size = resized_img_size

    def prepare_data(self, input_data, shuffle):
        # unpack data
        data = input_data[0]
        batch_size = tf.shape(data)[0]

        # scale data
        scale_1 = self.scale_and_resize_data(data, 1)
        scale_2 = self.scale_and_resize_data(data, 0.75)
        scale_3 = self.scale_and_resize_data(data, 0.5)
        scale_4 = self.scale_and_resize_data(data, 0.25)

        # scale_1 = self.scale_and_resize_data(data, 1)
        # scale_2 = self.scale_and_resize_data(data, 0.83)
        # scale_3 = self.scale_and_resize_data(data, 0.67)
        # scale_4 = self.scale_and_resize_data(data, 0.5)

        # scale_1 = self.scale_and_resize_data(data, 1)
        # scale_2 = self.scale_and_resize_data(data, 0.5)

        scaled_data = tf.concat([scale_1, scale_2, scale_3, scale_4], axis=0)

        # define labels
        labels = tf.concat([tf.zeros(batch_size, dtype=tf.int32),
                            tf.ones(batch_size, dtype=tf.int32),
                            2 * tf.ones(batch_size, dtype=tf.int32),
                            3 * tf.ones(batch_size, dtype=tf.int32)], axis=0)

        # scaled_data = tf.concat([scale_1, scale_2], axis=0)
        #
        # # define labels
        # labels = tf.concat([tf.zeros(batch_size, dtype=tf.int32),
        #                     tf.ones(batch_size, dtype=tf.int32)], axis=0)

        # shuffle data and labels
        if shuffle:
            indices = tf.range(start=0, limit=batch_size * self.num_classes, dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            shuffled_data = tf.gather(scaled_data, shuffled_indices)
            shuffled_labels = tf.gather(labels, shuffled_indices)
            return shuffled_data, shuffled_labels
        else:
            return scaled_data, labels

    def scale_and_resize_data(self, x, scale_factor):
        im_width, im_height = x.shape[2], x.shape[1]
        crop_width, crop_height = int(im_width * scale_factor), int(im_height * scale_factor)
        x_begin, y_begin = int(im_width / 2 - crop_width / 2), int(im_height / 2 - crop_height / 2)
        cropped_im = x[:, x_begin:x_begin + crop_width, y_begin:y_begin + crop_height, :]
        scaled_im = tf.image.resize(cropped_im, [self.img_size, self.img_size])
        return scaled_im
