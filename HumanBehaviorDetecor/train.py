import json
import os

import shutil
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from core.decode import decode
from core.loss import compute_loss, compute_pose_loss
from core.plugmodel import plugmodel, auxi_plugmodel
from core.yolov3_ import YOLOv3
from core.dataset import Dataset
from core.config import cfg, root

os.chdir(root)

YOLO = YOLOv3(True)
# YOLO.load_weights('./saved_model/Yolo_epoch_28_loss_57.29.h5')
for yololayer in YOLO.layers[-100:0]:
    yololayer.trainable = True
model = plugmodel()
input_tensor = tf.keras.layers.Input([416, 416, 3])
optimizer = tf.keras.optimizers.Adam()
trainset = Dataset('train')
logdir = "./data/log"
if not os.path.exists(logdir): os.mkdir(logdir)
weights_path = './saved_model'
if not os.path.exists(weights_path): os.mkdir(weights_path)

steps_per_epoch = len(trainset)
# steps_per_epoch = len(trainset)

warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch
print("step_per_epoch %4d,warmup_steps %4d,total_steps %4d " % (steps_per_epoch, warmup_steps, total_steps))

epoch_loss = 0
if os.path.exists('./log/global_steps.json'):
    with open('./log/global_steps.json', 'r') as f:
        global_steps = json.load(f)['steps']
    model.load_weights('./saved_model/plug_model_epoch_5_loss_3.73.h5')
else:
    global_steps = 1

def train_step(image_data, target):
    global global_steps
    giou_loss = conf_loss = prob_loss = person_pose_loss = 0
    with tf.GradientTape() as tape,tf.GradientTape() as tape2:
        conv_tensors = auxi_plugmodel(image_data, model, YOLO)
        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            pred_tensor = decode(conv_tensor, i)
            output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)
        pred_result = output_tensors
        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_pose_loss(pred, conv, *target[i], i)
            # giou_loss += loss_items[0]
            # conf_loss += loss_items[1]
            # prob_loss += loss_items[2]
            person_pose_loss += loss_items
        total_loss = person_pose_loss + giou_loss + conf_loss + prob_loss
        # total_loss =  giou_loss + conf_loss + prob_loss
    # gradients1 = tape.gradient(total_loss, YOLO.trainable_variables)
    # optimizer.apply_gradients(zip(gradients1, YOLO.trainable_variables))
    gradients2 = tape2.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients2, model.trainable_variables))
    return total_loss, person_pose_loss,giou_loss, conf_loss, prob_loss




for image_data, target in trainset:
    epoch = int(global_steps // steps_per_epoch)
    # print(epoch)
    total_loss, person_pose_loss,giou_loss, conf_loss, prob_loss=train_step(image_data, target)
    if global_steps % 30 == 0:
        print(
            "=>epoch:%2d/%4d  STEP:%4d   lr:%.6f  person_pose_loss: %4.5f giou_loss: %4.5f conf_loss: %4.5f "
            "prob_loss  %4.5f " % (
                epoch, cfg.TRAIN.EPOCHS, global_steps, optimizer.lr.numpy(),
                 person_pose_loss,giou_loss, conf_loss,prob_loss))
        epoch_loss = 0
    if global_steps < warmup_steps:
        lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
    else:
        lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
            (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
        )
    optimizer.lr.assign(lr)
    global_steps += 1
    if global_steps % 100 == 0:
        model.save_weights("./saved_model/plug_model_epoch_{}_loss_{:.2f}.h5".format(epoch,total_loss))
        # YOLO.save_weights("./saved_model/Yolo_epoch_{}_loss_{:.2f}.h5".format(epoch,total_loss))
        with open('./log/global_steps.json', 'w') as f:
            json.dump({
                'steps': global_steps
            }, f)
        print('successfully saved weigths!')
    if epoch > cfg.TRAIN.EPOCHS:
        break
model.save_weights("./saved_model/saved_weights.h5")
with open('./log/global_steps.json', 'w') as f:
    json.dump({
        'steps': global_steps,
        'mode':'end'
    }, f)

