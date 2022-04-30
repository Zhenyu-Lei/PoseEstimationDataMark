from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
from itertools import count
import os
import shutil
import math
from turtle import width
import re

from PIL import Image
from requests import status_codes
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time

import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_shoulder',
    2: 'right_shoulder',
    3: 'left_elbow',
    4: 'right_elbow',
    5: 'left_wrist',
    6: 'right_wrist',
    7: 'left_hip',
    8: 'right_hip',
    9: 'left_knee',
    10: 'right_knee',
    11: 'left_ankle',
    12: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

NUM_KPTS = 17

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# 人体目标检测,模型。图像tensor，阈值
def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    # print(pred)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    # 获取类别，位置和置信度
    if not pred_score or max(pred_score) < threshold:
        return []
    # Get list of index with score greater than threshold
    # 置信度是从高到低排的，所以是把置信度大于阈值的框全部测出来
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]  # 置信度大于阈值才会被选出，选取其中最后一个index
    # print(pred_t)
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_classes = pred_classes[:pred_t + 1]

    person_boxes = []
    # 抽取出所有检测为人的框
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)
            break

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    # 仿射变换的变换矩阵
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    # warpAffine(输入图像，变换矩阵，输出图像大小，插值方法)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    # 先将input展开为tensor，然后三个通道分别normalize，很奇怪为什么选这样的方差和标准差
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    # model_input大小为[1, 3, 384, 288]
    model_input = transform(model_input).unsqueeze(0)
    # print(model_input.shape)
    # switch to evaluate mode
    pose_model.eval()
    # with表达式其实是try-finally
    with torch.no_grad():
        # compute output heatmap，进行估计
        output = pose_model(model_input)
        # output输出是什么？
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        preds = np.delete(preds, [1, 2, 3, 4], 1)

        return preds


# 转换方框到中心，缩放信息所需的姿态转换（盒子，图片宽度，图片长度）
def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5
    # center获取box的中心

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    # 调整box_height和box_width为期望的image的比例
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    # 获取box的中心和大小，并且将大小扩展到1.25倍
    return center, scale


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='demo/inference-config.yaml')
    parser.add_argument('--video', type=str)
    parser.add_argument('--webcam', action='store_true')
    parser.add_argument('--outputDir', type=str, default='./output/')
    parser.add_argument('--image', type=str)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--showFps', action='store_true')
    parser.add_argument('--inputFile', type=str)

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # cudnn related setting 配置相关信息，用于让GPU选择最高效的算法
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()  # 读取参数，同时更新config参数
    update_config(cfg, args)

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                     pretrained_backbone=False)  # 下载好预训练模型的faster_rcnn
    # https://pytorch.org/vision/0.8/models.html torch提供的模型
    box_model.load_state_dict(torch.load("myModels/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))
    box_model.to(CTX)  # 将模型放到GPU或CPU上
    box_model.eval()  # 使模型状态变为eval()状态

    # 根据配置文件构建网络
    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))  # TEST.MODEL_FILE是命令行给出的
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)  # 加载模型
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)  # 设置模型，放置在GPU上，设置为eval状态
    pose_model.to(CTX)
    pose_model.eval()

    csv_output_rows = []
    for root, dirs, files in os.walk(args.inputFile, topdown=False):
        for file in files:
            if not re.match("(.*).jpg", str(file)):
                continue
            image_bgr = cv2.imread(root + '/' + file)
            print(root + '/' + file)
            print(type(image_bgr))
            # estimate on the image
            last_time = time.time()
            image = image_bgr[:, :, [2, 1, 0]]

            theHeight = int(image_bgr.shape[0])
            theWidth = int(image_bgr.shape[1])

            input = []
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)
            input.append(img_tensor)

            # object detection box
            pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

            # pose estimation
            if len(pred_boxes) >= 1:
                for box in pred_boxes:
                    center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                    image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                    pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)

                    # csv获取所有关键点坐标
                    new_csv_row = [str(file)]
                    for coords in pose_preds:
                        # Draw each point on image
                        for coord in coords:
                            x_coord, y_coord = int(coord[0]), int(coord[1])
                            new_csv_row.extend([x_coord, y_coord])
                    new_csv_row.extend([theWidth, theHeight])
                    csv_output_rows.append(new_csv_row)
            else:
                new_csv_row = [str(file)]
                for i in range(0, 13):
                    # Draw each point on image
                    new_csv_row.extend([0, 0])
                new_csv_row.extend([theWidth, theHeight])
                csv_output_rows.append(new_csv_row)
    # print(file)
    # write csv
    csv_headers = ['FileName']
    for keypoint in COCO_KEYPOINT_INDEXES.values():
        csv_headers.extend([keypoint + '_x', keypoint + '_y'])
    csv_headers.extend(['width', 'height'])

    csv_output_filename = os.path.join('./images_baidu_clean1', 'dataPreMark.csv')
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)


if __name__ == '__main__':
    main()
