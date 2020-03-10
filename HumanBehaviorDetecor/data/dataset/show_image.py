
import cv2
import numpy as np
from PIL import Image


label_txt = "D:\projects\HumanBehaviorDetecor\data\dataset\pose_train.txt"
ID = 0
image_infos = open(label_txt).readlines()
num = 0
for image_info_item in image_infos:
    num+=1
    # try:
    image_info = image_info_item.split()
    # print(image_i)
    image_path = image_info[0]
    image = cv2.imread(image_path)
    # print(image_path)
    # image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.colorChange(image,cv2.COLOR_BGR2RGB)
    for bbox in image_info[1:]:
        bbox = bbox.split(",")
        image = cv2.rectangle(image, (int(float(bbox[0])),
                                      int(float(bbox[1]))),
                                      (int(float(bbox[2])),
                                       int(float(bbox[3]))), (255, 0, 0), 2)
    cv2.imshow('show',image)
    cv2.waitKey(1)
    # except Exception as e:
    #     print(e)
    #     print(num)
    #     print(image_path)
    # image = cv2.imread(r'D:\dataset\fall\18.jpg')
    # cv2.imshow('fds',image)
    # cv2.waitKey(0)