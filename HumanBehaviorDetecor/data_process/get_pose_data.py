import os
import shutil
import xml.dom.minidom as xdom
from pprint import pprint

from core.config import cfg, root
os.chdir(root)
import core.utils as utils
from utils.resize_image import ResizeWithPad

PERSON_POSES_CLASSES = utils.read_class_names(cfg.YOLO.PERSON_POSES_CLASSES)
name2label = dict(zip(PERSON_POSES_CLASSES.values(), PERSON_POSES_CLASSES.keys()))
print(PERSON_POSES_CLASSES)
print(name2label)

actions = set()
dirnames = os.listdir(cfg.DATA.PASCAL_VOC_ACTION)

for action in dirnames:
    action = action.split('_')
    if action[0].endswith('.txt'):
        continue
    actions.add(action[0])
# actions.discard('ridinghorse')
actions = sorted(actions)
with open(cfg.YOLO.PERSON_POSES_CLASSES, 'a+') as f:
    for action in actions:
        write_mess = action
        f.write(action)
        f.write('\n')
    f.write('fall')

# unlabeled = set()
annotation = {}
for action in actions:
    # annotation[action] = {}
    with open(os.path.join(cfg.DATA.PASCAL_VOC_ACTION, str(action) + '_trainval.txt')) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('  ', ' ')
            line = line.strip('\n')
            line = line.split(' ')
            if line[2] == '1':
                annotation[(line[0], line[1])] = name2label[action]
            else:
                if annotation.get(line[0], line[1]) is None:
                    annotation[(line[0], line[1])] = -1


a = list(zip(annotation.keys()))
import numpy as np
a = np.array(a)
a = np.squeeze(a)
key = list(a[:, 0])
num = list(a[:, 1])
value = list(annotation.values())

keys = dict()
for img in set(key):
    keys[img] = {}
for img in set(key):
    for i, j, k in list(zip(key, num, value)):
        if i == img:
            keys[img][j] = k
pprint(keys)

for img_name in keys:
    img_dir = os.path.join(cfg.DATA.PASCAL_VOC_IMAGE, img_name + '.jpg')
    img_anno_dir = os.path.join(cfg.DATA.PASCAL_VOC_ANNOTATION, img_name + '.xml')
    line = img_dir
    DOMTree = xdom.parse(img_anno_dir)
    annotation = DOMTree.documentElement
    obj = annotation.getElementsByTagName("object")
    for i, o in enumerate(obj):
        o_list = []
        obj_name = o.getElementsByTagName("name")[0].childNodes[0].data
        bndbox = o.getElementsByTagName("bndbox")
        for box in bndbox:
            xmin = box.getElementsByTagName("xmin")[0].childNodes[0].data
            ymin = box.getElementsByTagName("ymin")[0].childNodes[0].data
            xmax = box.getElementsByTagName("xmax")[0].childNodes[0].data
            ymax = box.getElementsByTagName("ymax")[0].childNodes[0].data


            try:
                str(keys[img_name][str(i + 1)])
                line += ' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(
                    keys[img_name][str(i + 1)])
                break
            except:
                pass

    with open(root + cfg.TRAIN.ANNOT_PATH, 'a+') as f:
        f.write(line + '\n')

for xml in os.listdir(root + '/data/dataset/anno'):
    DOMTree = xdom.parse(root + '/data/dataset/anno/' + xml)
    annotation = DOMTree.documentElement
    size = annotation.getElementsByTagName("size")
    path = annotation.getElementsByTagName('path')[0].childNodes[0].data
    line = path
    image_height = 0
    image_width = 0
    for s in size:
        image_height = s.getElementsByTagName("height")[0].childNodes[0].data
        image_width = s.getElementsByTagName("width")[0].childNodes[0].data

    obj = annotation.getElementsByTagName("object")
    for i, o in enumerate(obj):
        o_list = []
        obj_name = o.getElementsByTagName("name")[0].childNodes[0].data
        bndbox = o.getElementsByTagName("bndbox")
        for box in bndbox:
            xmin = box.getElementsByTagName("xmin")[0].childNodes[0].data
            ymin = box.getElementsByTagName("ymin")[0].childNodes[0].data
            xmax = box.getElementsByTagName("xmax")[0].childNodes[0].data
            ymax = box.getElementsByTagName("ymax")[0].childNodes[0].data
            try:
                line += ' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(
                    name2label[obj_name])
            except Exception as e:
                print(e)
    print(line)
    # if os.path.exists(root + cfg.TRAIN.ANNOT_PATH):
    #     shutil.rmtree(root + cfg.TRAIN.ANNOT_PATH)
    with open(root + cfg.TRAIN.ANNOT_PATH, 'a+') as f:
        f.write(line + '\n')
