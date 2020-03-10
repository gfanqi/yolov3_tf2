import os
from core.config import cfg, root
os.chdir(root)
import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras import Model
import core.utils as utils
import core.backbone as backbone
from core import dataset, common

NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
NUM_POSES = len(utils.read_class_names(cfg.YOLO.PERSON_POSES_CLASSES))
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

def YOLOv3(for_annother_use=False):
    input_layer = tf.keras.layers.Input(shape=[416,416,3])
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    sconv1 = conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    sconv2 = conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    sconv3 = conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    mconv1 = conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    mconv2 = conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    mconv3 = conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    lconv1 = conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    lconv2 = conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    lconv3 = conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    if for_annother_use:
        convs_sbbox = tf.concat([lconv1, lconv2, lconv3], axis=-1, name="lfeature")
        convs_mbbox = tf.concat([mconv1, mconv2, mconv3], axis=-1, name="mfeature")
        convs_lbbox = tf.concat([sconv1, sconv2, sconv3], axis=-1, name="sfeature")
        yolo = Model(input_layer,[conv_sbbox, conv_mbbox, conv_lbbox, convs_sbbox,convs_mbbox,convs_lbbox])
    #                              52,26,13                         52,26,13
        yolo.load_weights(filepath=tf.keras.utils.get_file('yolo3.h5',''))
        yolo.trainable = False
    else:
        # classic YOLOV3 model trained for coco datasets
        yolo = Model(input_layer, [conv_sbbox, conv_mbbox, conv_lbbox])
        yolo.load_weights(filepath=tf.keras.utils.get_file('yolo3.h5',''))
        yolo.trainable =False
    return yolo

