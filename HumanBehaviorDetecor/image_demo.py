import os
from core.config import root
os.chdir(root)
import tensorflow as tf
from core import dataset
from core.decode import decode
from core.plugmodel import plugmodel, auxi_plugmodel
from core.yolov3_ import YOLOv3

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import cv2
import numpy as np
import core.utils as utils

input_size = 416
flag = True

def use_model(original_image, label):
    global model, YOLO,flag
    if flag:
        YOLO = YOLOv3(True)
        # YOLO.load_weights('./saved_model/Yolo_epoch_25_loss_64.19.h5')
        model = plugmodel()
        model.load_weights('./saved_model/plug_model_epoch_0_loss_2.96.h5', by_name=True)
        flag =False
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    conv_tensors = auxi_plugmodel(image_data, model, YOLO)
    bbox_tensors = [decode(feature, k) for k, feature in enumerate(conv_tensors)]
    pred_bbox = bbox_tensors
    return pred_bbox

def show_label(original_image, label):
    return label

def show(pred_bbox):
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    image = utils.draw_bbox(original_image, bboxes, show_label=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('sample', image)
    cv2.waitKey(0)

for i in range(1, 200):
    original_image, label, _ = dataset.Dataset('train').get_sample(i)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    # conv_tensors = use_model(original_image, label, )
    # show(conv_tensors)
    c = show_label(original_image, label)
    show(c)

