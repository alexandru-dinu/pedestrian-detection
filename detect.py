from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--batch_count', type=int, default=10, help="num batches to pass through")
parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--data_config_path', type=str, default='config/coco.data', help='path to data config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--detect_dir', type=str, default="detections", help='detections folder')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--use_cuda', action="store_true", help='whether to use cuda if available')
parser.add_argument('--shuffle', action="store_true")
opt = parser.parse_args()

for x in opt.__dict__:
    print("%25s: %s" % (x, opt.__dict__[x]))
print("-" * 80)

FONT = cv2.FONT_HERSHEY_TRIPLEX
COLORS = [tuple(255 * np.array(plt.get_cmap('tab20b')(i)[:-1])) for i in np.linspace(0, 1, 20)]

cuda = torch.cuda.is_available() and opt.use_cuda
os.makedirs(os.path.join("detections", opt.detect_dir), exist_ok=True)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
test_path = data_config['detect']  # path to the imgs txt file (can be a subset of val.txt)
num_classes = int(data_config['classes'])
names = data_config['names']

for x, y in data_config.items():
    print("%25s: %s" % (x, y))
print("-" * 80)

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)

model.cuda()
model.eval()

img_size = int(model.hyperparams['height'])

for x, y in model.hyperparams.items():
    print("%25s: %s" % (x, y))
print("-" * 80)

print("Model loading done")

# Get dataloader
dataset = ListDataset(test_path)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.n_cpu
)

classes = load_classes(names)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

print('\nPerforming object detection:')
prev_time = time.time()
for batch_i, (img_paths, input_imgs, _) in enumerate(dataloader):
    if batch_i == opt.batch_count:
        break

    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, num_classes, opt.conf_thres, opt.nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print('\nSaving images:\n')

# Iterate through images and save plot of detections
for img_idx, (path, detections) in enumerate(zip(imgs, img_detections)):
    print("\nImage: '%s'" % path)

    img = cv2.imread(path)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))

    # Image height and width after padding is removed
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        bbox_colors = random.sample(COLORS, len(unique_labels))

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # rescale coordinates to original dimensions
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x2 = ((x2 - pad_x // 2) / unpad_w) * img.shape[1]
            y2 = ((y2 - pad_y // 2) / unpad_h) * img.shape[0]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

            # draw bbox over image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)

            # add label
            cv2.putText(img, str(int(cls_pred)), (x1, y1 - 3), FONT, 1, (255, 255, 255), 2)

            print(
                '\t+ Coords: [%4d, %4d, %4d, %4d], Class: %s, ObjConf: %.5f, ClassProb: %.5f' % \
                (x1, y1, x2, y2, classes[int(cls_pred)], conf.item(), cls_conf.item())
            )

    # Save generated image with detections
    save_path = 'detections/%s/detect_%d.png' % (opt.detect_dir, img_idx)
    cv2.imwrite(save_path, img)
