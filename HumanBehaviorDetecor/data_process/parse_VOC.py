import os

from core.config import cfg, root

os.chdir(root)
import xml.dom.minidom as xdom

import core.utils as utils

PERSON_POSES_CLASSES = utils.read_class_names(root + cfg.YOLO.PERSON_POSES_CLASSES)
name2label = dict(zip(PERSON_POSES_CLASSES.values(), PERSON_POSES_CLASSES.keys()))
print(PERSON_POSES_CLASSES)
print(name2label)

# cfg.TRAIN.ANNOT_PATH
# 用于解析自己标注使用labelImg标注的数据
def parse_xml(dirname):
    with open(dirname, 'a+') as f:
        file_num = len(os.listdir(dirname))
        for num, xml in enumerate(os.listdir(dirname)):
            DOMTree = xdom.parse(os.path.join(dirname + xml))
            annotation = DOMTree.documentElement
            path = annotation.getElementsByTagName('path')[0].childNodes[0].data
            line = path
            obj = annotation.getElementsByTagName("object")
            for i, o in enumerate(obj):
                obj_name = o.getElementsByTagName("name")[0].childNodes[0].data
                bndbox = o.getElementsByTagName("bndbox")
                for box in bndbox:
                    xmin = box.getElementsByTagName("xmin")[0].childNodes[0].data
                    ymin = box.getElementsByTagName("ymin")[0].childNodes[0].data
                    xmax = box.getElementsByTagName("xmax")[0].childNodes[0].data
                    ymax = box.getElementsByTagName("ymax")[0].childNodes[0].data
                    line += ' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(
                        name2label[obj_name])
            print(line)
            f.write(line + '\n')
    print('done!')


def parse_action1():
    # 从文件夹名中得到分类信息,并写入文件
    actions = set()
    dirnames = os.listdir(cfg.DATA.PASCAL_VOC_ACTION)
    for action in dirnames:  # ['move_train.txt',...,fly_val.txt']
        action = action.split('_')
        if action[0].endswith('.txt'):
            continue
        actions.add(action[0])
    # actions.discard('ridinghorse')
    actions = sorted(actions)
    with open(cfg.YOLO.PERSON_POSES_CLASSES, 'w') as f:
        for action in actions:
            f.write(action)
            f.write('\n')
    return actions

def parse_action2(actions):
    # 处理动作分类及其与图片名称的对应关系返回一个字典。结构为{name:{index,action}}

    img_names = []
    indexes = []
    action_labels = []
    for action in actions:
        # annotation[action] = {}
        with open(os.path.join(cfg.DATA.PASCAL_VOC_ACTION, str(action) + '_trainval.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('  ', ' ')
                line = line.strip('\n')
                line = line.split(' ')
                if line[2] == '1':
                    img_names.append(line[0])
                    indexes.append(line[1])
                    action_labels.append(name2label[action])

    anno_dict = dict()
    print(set(img_names))
    for img in set(img_names):
        anno_dict[img] = {}
    for img in set(img_names):
        for i, j, k in list(zip(img_names, indexes, action_labels)):
            if i == img:
                anno_dict[img][j] = k
    return anno_dict

def parse_VOC_xlm_jpg(anno_dict):
    with open(cfg.TRAIN.ANNOT_PATH, 'w') as f:
        for img_name in anno_dict:
            img_dir = os.path.join(cfg.DATA.PASCAL_VOC_IMAGE, img_name + '.jpg')
            img_anno_dir = os.path.join(cfg.DATA.PASCAL_VOC_ANNOTATION, img_name + '.xml')
            line = img_dir
            DOMTree = xdom.parse(img_anno_dir)
            annotation = DOMTree.documentElement
            obj = annotation.getElementsByTagName("object")
            for i, o in enumerate(obj):
                bndbox = o.getElementsByTagName("bndbox")
                for box in bndbox:
                    xmin = box.getElementsByTagName("xmin")[0].childNodes[0].data
                    ymin = box.getElementsByTagName("ymin")[0].childNodes[0].data
                    xmax = box.getElementsByTagName("xmax")[0].childNodes[0].data
                    ymax = box.getElementsByTagName("ymax")[0].childNodes[0].data
                    try:
                        a = anno_dict[img_name][str(i + 1)]
                        # VOC数据中标记位置的数据不一定有动作这个标记
                        line += ' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(
                            anno_dict[img_name][str(i + 1)])
                    except Exception as e:
                        continue
            f.write(line + '\n')

if __name__ == '__main__':
    actions = parse_action1()
    anno_dict = parse_action2(actions)
    parse_VOC_xlm_jpg(anno_dict)
    # parse_xml(r'D:\dataset\fall_label/')
