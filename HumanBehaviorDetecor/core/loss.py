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


# tf API
def bbox_iou(boxes1, boxes2):
    '''
    交并比
    '''
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


# tf API
def bbox_giou(boxes1, boxes2):
    '''
    交除围起来的总面积
    '''

    # print(boxes1.shape,boxes2.shape)
    # 左上角，右下角坐标计算
    # x-0.5w ,y - 0.5h,x+0.5w ,y + 0.5h,
    # print(boxes1)
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    # iou_mask = tf.cast(union_area > 0.0,dtype=tf.float32)
    iou = (tf.exp(inter_area - union_area) - 1.0) / (np.e - 1)  # 归一化
    # tf.clip_by_value(,0.001,union_area)
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    fiou = (enclose_area - union_area) / enclose_area
    ################################ #
    #########!!!!!!!!!?????###########
    #########!!!!!$$$$$....###########
    #########!!!!!$$$$$....###########
    #########?????.........###########
    ################################## fiou 表示问号除于所有除了# 符号的面积
    giou = iou - fiou
    return giou


from functools import wraps, partial


def compute_loss(pred, conv, label, bboxes, i=0):
    '''

    :param pred: 经过了解码的yolo3输出
    :param conv: 没有经过解码的yolo3输出
    :param label:
    :param bboxes:
    :param i:
    :return:
    '''
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = 416
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, NUM_CLASS + NUM_POSES + 5))
    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_class_prob = conv[:, :, :, :, 5:5 + NUM_CLASS]
    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]
    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_class_prob = label[:, :, :, :, 5:5 + NUM_CLASS]

    # giou_loss 位置损失函数
    pred_xywh = tf.nn.relu(pred_xywh)
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    # input_size = tf.cast(input_size, tf.float32)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    # bbox_loss_scale 计算的是背景占整张图片面积的比例 input_size = 416
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou) * 10

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf))

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_class_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_class_prob,
                                                                             logits=conv_raw_class_prob)
    prob_class_loss = tf.reduce_mean(tf.reduce_sum(prob_class_loss, axis=[1, 2, 3, 4]))
    # plugmodelloss
    prob_person_pose_loss = compute_pose_loss(pred, conv, label) * 0.01
    # giou_loss, conf_loss, prob_class_loss=0,0,0
    # tf.print("=>giou_loss{}, conf_loss, prob_class_loss")
    return giou_loss, conf_loss, prob_class_loss, prob_person_pose_loss


def compute_pose_loss(pred, conv, label,bbbox=None,i=0):
    '''

    :param pred: 经过了解码的yolo3输出
    :param conv: 没有经过解码的yolo3输出
    :param label:
    :param bboxes:
    :param i:
    :return:
    '''
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, NUM_CLASS + NUM_POSES + 5))
    pred_person_prob = pred[:, :, :, :, 5:6]
    conv_raw_person_pose_prob = conv[:, :, :, :, 5 + NUM_CLASS:]
    is_person_prob = label[:, :, :, :, 5:6]  # 这一层是对是不是人进行标注, 是人：prob=1，不是人：prob=-0
    label_person_pose_prob = label[:, :, :, :, 5 + NUM_CLASS:]
    prob_person_pose_loss = is_person_prob * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_person_pose_prob,
                                                                                     logits=conv_raw_person_pose_prob) \
                            + tf.nn.relu(
        1 - is_person_prob - pred_person_prob) * tf.nn.sigmoid_cross_entropy_with_logits(
        labels=label_person_pose_prob, logits=conv_raw_person_pose_prob)
    # loss = 标签中是人的概率*交叉熵+不该预测为人的概率*交叉熵，这里的乘法是逐元素乘法。
    # is_person_prob 实际上扮演的是一个过滤器的角色，由于里边的乘法计算是逐元素的，因此能过滤掉没有被标注为人的网格点。
    # tf.nn.relu(1 - is_person_prob-pred_person_prob)同样是一个过滤器，使用relu 函数是为了保证最小不会有小于零的树数，
    # 小于零的数对于此项来说没有意义。由于经过coco数据集训练过的网络已经能够很好的给人分类，因此用
    # (1- is_person_prob-pred_person_prob)过滤掉交叉熵是人的数据，它们是背景。由于yolo算法的特性，除了中心点外，中心点的其他
    # 部分也会被预测为人
    prob_person_pose_loss = tf.reduce_mean(tf.reduce_sum(prob_person_pose_loss, axis=[1, 2, 3, 4]))
    return prob_person_pose_loss
