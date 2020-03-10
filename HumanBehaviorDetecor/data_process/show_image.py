
import cv2
import numpy as np
from PIL import Image

import os

from core.config import root

os.chdir(root)

ID = 0
label_txt = "./data/dataset/pose_train.txt"
image_info = open(label_txt).readlines()[ID].split()

image_path = image_info[0]
image = cv2.imread(image_path)
# image = Image.fromarray(np.uint8(image))
# image.show()
for bbox in image_info[1:]:
    bbox = bbox.split(",")
    image = cv2.rectangle(image,(int(float(bbox[0])),
                                 int(float(bbox[1]))),
                                (int(float(bbox[2])),
                                 int(float(bbox[3]))), (255,0,0), 2)

image = Image.fromarray(np.uint8(image))
image.show()
