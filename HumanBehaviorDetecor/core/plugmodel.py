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


# ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
# STRIDES = np.array(cfg.YOLO.STRIDES)
# IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

def plugmodel():
    sbbox = tf.keras.layers.Input(shape=(52, 52, 768))
    mbbox = tf.keras.layers.Input(shape=(26, 26, 1536))
    lbbox = tf.keras.layers.Input(shape=(13, 13, 3072))
    conv_spose = common.convolutional(sbbox, (1, 1, 256, 256))
    conv_spose = common.convolutional(conv_spose, (1, 1, 256, 3 * (NUM_POSES)), activate=False, bn=False)

    conv_mpose = common.convolutional(mbbox, (1, 1, 512, 512))
    conv_mpose = common.convolutional(conv_mpose, (1, 1, 512, 3 * (NUM_POSES)), activate=False, bn=False)

    conv_lpose = common.convolutional(lbbox, (1, 1, 255, 1024))
    conv_lpose = common.convolutional(conv_lpose, (1, 1, 1024, 3 * NUM_POSES), activate=False, bn=False)

    return Model(inputs=[sbbox, mbbox, lbbox], outputs=[conv_spose, conv_mpose, conv_lpose])

# 插件网络的辅助函数
def auxi_plugmodel(input_layer, plug_model, YOLO):
    conv_sbbox, conv_mbbox, conv_lbbox, convs_sbbox,convs_mbbox,convs_lbbox = YOLO(input_layer)
    conv_lbbox = common.reshape(conv_lbbox, 5 + NUM_CLASS)
    conv_mbbox = common.reshape(conv_mbbox, 5 + NUM_CLASS)
    conv_sbbox = common.reshape(conv_sbbox, 5 + NUM_CLASS)

    conv_spose, conv_mpose, conv_lpose, = plug_model([convs_sbbox,convs_mbbox,convs_lbbox])
    conv_spose = common.reshape(conv_spose, NUM_POSES)
    conv_mpose = common.reshape(conv_mpose, NUM_POSES)
    conv_lpose = common.reshape(conv_lpose, NUM_POSES)

    final_conv_sbbox = tf.concat([conv_sbbox, conv_spose], axis=-1)
    final_conv_mbbox = tf.concat([conv_mbbox, conv_mpose], axis=-1)
    final_conv_lbbox = tf.concat([conv_lbbox, conv_lpose], axis=-1)

    return [final_conv_sbbox, final_conv_mbbox, final_conv_lbbox]
    #       52,52,3,5+NUM_CLASS+NUM_POSE, 26,26,3,5+NUM_CLASS+NUM_POSE,13,13,3,5+NUM_CLASS+NUM_POSE,
