import os
from core.config import cfg, root
os.chdir(root)
import numpy as np
import tensorflow as tf
import core.utils as utils


NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
NUM_POSES = len(utils.read_class_names(cfg.YOLO.PERSON_POSES_CLASSES))
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

class Decode:
    def __init__(self, i):
        self.i = i

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
                contains (x, y, w, h, score, probability)
        """
        conv_output = inputs
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, -1))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[self.i]
        pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[self.i]) * STRIDES[self.i]
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def decode(convlayer, i):
    return Decode(i=i).call(convlayer)