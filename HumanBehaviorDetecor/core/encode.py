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
class Encode:
    '''
    not used in this project,it is the reverse transformathon of the Decode.I made it just for test.
    '''

    def __init__(self,i):
        self.i = i

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self,label):
        label_raw_dxdy = label[:, :, :, :, 0:2]
        label_raw_dwdh = label[:, :, :, :, 2:4]
        label_raw_conf = label[:, :, :, :, 4:5]
        label_raw_prob = label[:, :, :, :, 5:]

        label_conf = self.reverse_sigmoid(label_raw_conf)
        label_prob = self.reverse_sigmoid(label_raw_prob)

        label_shape = tf.shape(label)
        batch_size = label_shape[0]
        output_size = label_shape[1]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        label_dxdy = self.reverse_sigmoid(label_raw_dxdy / STRIDES[i] - xy_grid)
        label_dwdh = tf.math.log(label_raw_dwdh / (STRIDES[i] * ANCHORS[i]))

        conv_output = tf.concat([label_dxdy, label_dwdh, label_conf, label_prob], axis=-1)
        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3 * (NUM_CLASS + NUM_POSES + 5)))
        return conv_output


    @classmethod
    def reverse_sigmoid(cls, x):
        x = tf.clip_by_value(x, 1e-10, 1)
        return tf.math.log((x) / (1 - x))