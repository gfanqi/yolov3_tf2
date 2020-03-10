import collections
import logging
import os

import cv2
import random
import numpy as np
import tensorflow as tf

import core.utils as utils
from core.config import cfg, root

os.chdir(root)


class Dataset:
    """implement Dataset here"""

    def __init__(self, dataset_type):
        self.annot_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.poses_classes = utils.read_class_names(cfg.YOLO.PERSON_POSES_CLASSES)
        self.num_classes = len(self.classes)
        self.num_poses_classes = len(self.poses_classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self, dataset_type, shuffle=True):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        if shuffle:
            np.random.shuffle(annotations)
        return annotations

    def get_sample(self, index):
        self.train_input_size = random.choice(self.train_input_sizes)
        self.train_output_sizes = self.train_input_size // self.strides
        batch_label_sbbox = np.zeros((1, self.train_output_sizes[0], self.train_output_sizes[0],
                                      self.anchor_per_scale, 5 + self.num_classes + self.num_poses_classes),
                                     dtype=np.float32)
        batch_label_mbbox = np.zeros((1, self.train_output_sizes[1], self.train_output_sizes[1],
                                      self.anchor_per_scale, 5 + self.num_classes + self.num_poses_classes),
                                     dtype=np.float32)
        batch_label_lbbox = np.zeros((1, self.train_output_sizes[2], self.train_output_sizes[2],
                                      self.anchor_per_scale, 5 + self.num_classes + self.num_poses_classes),
                                     dtype=np.float32)

        batch_sbboxes = np.zeros((1, self.max_bbox_per_scale, 4), dtype=np.float32)
        batch_mbboxes = np.zeros((1, self.max_bbox_per_scale, 4), dtype=np.float32)
        batch_lbboxes = np.zeros((1, self.max_bbox_per_scale, 4), dtype=np.float32)

        num = 0
        annotation = self.load_annotations('test', shuffle=False)[index]
        image, bboxes = self.parse_annotation(annotation, get_samples=True)
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
            bboxes)

        batch_label_sbbox[num, :, :, :, :] = label_sbbox
        batch_label_mbbox[num, :, :, :, :] = label_mbbox
        batch_label_lbbox[num, :, :, :, :] = label_lbbox
        batch_sbboxes[num, :, :] = sbboxes
        batch_mbboxes[num, :, :] = mbboxes
        batch_lbboxes[num, :, :] = lbboxes
        batch_smaller_target = batch_label_sbbox
        batch_medium_target = batch_label_mbbox
        batch_larger_target = batch_label_lbbox
        return image, (batch_smaller_target, batch_medium_target, batch_larger_target), (
        batch_sbboxes, batch_mbboxes, batch_lbboxes)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            with tf.device('/cpu:0'):
                self.train_input_size = random.choice(self.train_input_sizes)
                self.train_output_sizes = self.train_input_size // self.strides

                batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3),
                                       dtype=np.float32)

                batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                              self.anchor_per_scale, 5 + self.num_classes + self.num_poses_classes),
                                             dtype=np.float32)
                batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                              self.anchor_per_scale, 5 + self.num_classes + self.num_poses_classes),
                                             dtype=np.float32)
                batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                              self.anchor_per_scale, 5 + self.num_classes + self.num_poses_classes),
                                             dtype=np.float32)

                batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
                batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
                batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

                num = 0
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index = 0
                        self.batch_count = 0
                    try:
                        annotation = self.annotations[index]
                    except Exception as e:
                        with open('log.txt', 'a+') as f:
                            f.write('index:' + str(index) + '\n', )
                            f.write('num_samples:' + str(self.num_samples) + '\n', )
                            logging.warning(msg='ran out of data')
                        # index = 0
                    image, bboxes = self.parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                        bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes

                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation, get_samples=False):

        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        org_image = cv2.imread(image_path)
        image = cv2.imread(image_path)

        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        if self.data_aug and not get_samples:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size],
                                               np.copy(bboxes))
        if get_samples:
            return org_image, bboxes
        else:
            return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes + self.num_poses_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_pose_ind = bbox[4]

            class_onehot = np.zeros(self.num_classes, dtype=np.float)
            class_onehot[0] = 1.0  # 人是coco数据集中第一个
            class_uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)

            pose_onehot = np.zeros(self.num_poses_classes, dtype=np.float)
            pose_onehot[bbox_pose_ind] = 1.0
            pose_uniform_distribution = np.full(self.num_poses_classes, 1.0 / self.num_poses_classes)
            deta = 0.01
            smooth_onehot = np.concatenate((class_onehot * (1 - deta) + deta * class_uniform_distribution,
                                            pose_onehot * (1 - deta) + deta * pose_uniform_distribution),
                                           axis=-1)

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs

    def __call__(self, *args, **kwargs):
        return self

    @property
    def shape(self):
        return self.__len__(),


if __name__ == '__main__':
    pass
    num = 0
