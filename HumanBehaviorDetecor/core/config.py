import os

from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# YOLO options
__C.YOLO = edict()
root = os.path.dirname(os.path.dirname(__file__))
# print(os.path.abspath("./data/classes/coco.names"))
# Set the class name
__C.YOLO.CLASSES = "./data/classes/coco.names"
__C.YOLO.PERSON_POSES_CLASSES = './data/classes/poses.names'
__C.YOLO.ANCHORS = "./data/anchors/basline_anchors.txt"
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = "./data/dataset/pose_train.txt"
__C.TRAIN.SAVED_MODEL_PATH = './saved_model/'
__C.TRAIN.BATCH_SIZE = 3
# __C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE = [416]
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LR_INIT = 1e-3
__C.TRAIN.LR_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 2
__C.TRAIN.EPOCHS = 20

# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = "./data/dataset/pose_test.txt"
__C.TEST.BATCH_SIZE = 2
__C.TEST.INPUT_SIZE = 544
__C.TEST.DATA_AUG = False
__C.TEST.DECTECTED_IMAGE_PATH = None
__C.TEST.SCORE_THRESHOLD = 0.3
__C.TEST.IOU_THRESHOLD = 0.45

# DATA_PARSE
__C.DATA = edict()
__C.DATA.PASCAL_VOC_ANNOTATION = r"F:\dataset\VOC\train\VOCdevkit\VOC2012\Annotations"
__C.DATA.PASCAL_VOC_IMAGE = r"F:\dataset\VOC\train\VOCdevkit\VOC2012\JPEGImages/"
__C.DATA.PASCAL_VOC_PERSONS = r'F:\dataset\VOC\train\VOCdevkit\VOC2012\ImageSets\Main\person_trainval.txt'
__C.DATA.PASCAL_VOC_ACTION = r'F:\dataset\VOC\train\VOCdevkit\VOC2012\ImageSets\Action'
