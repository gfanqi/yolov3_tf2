import cv2
import time
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.decode import decode
from core.plugmodel import plugmodel, auxi_plugmodel
from core.yolov3_ import YOLOv3

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)

video_path = "F:/美剧/罗马第二季/S02E01.CN.BluRay.HR-HDTV.AC3.1024X576.x264.mkv"
# video_path      = 0

input_size = 416


YOLO = YOLOv3(True)
model = plugmodel()

vid = cv2.VideoCapture(video_path)
fps = vid.get(cv2.CAP_PROP_FPS)
i = 0
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("No image!")
    i += 1
    if i > 500:
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        prev_time = time.time()
        feature_maps = auxi_plugmodel(image_data, model, YOLO)
        pred_bbox = [decode(fm, i) for i, fm in enumerate(feature_maps)]
        curr_time = time.time()
        exec_time = curr_time - prev_time
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame, bboxes)

        result = np.asarray(image)
        info = "time: %.2f ms" % (1000 * exec_time)

        cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        result = frame
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result", result)
    cv2.waitKey(int(1000 // fps))
    if cv2.waitKey(1) & 0xFF == ord('q'): break
