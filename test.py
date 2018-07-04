from __future__ import division

import argparse

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *
from coco_conv import coco_to_udacity

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
parser.add_argument('--batch_count', type=int, default=10, help="num batches to pass through")
parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--data_config_path', type=str, default='config/coco.data', help='path to data config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.45, help='iou thresshold for non-maximum suppression')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--use_cuda', action="store_true", help='whether to use cuda if available')
parser.add_argument('--shuffle', action="store_true")
opt = parser.parse_args()

for x in opt.__dict__:
    print("%25s: %s" % (x, opt.__dict__[x]))
print("-" * 80)

cuda = torch.cuda.is_available() and opt.use_cuda

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
test_path = data_config['valid']
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

print("Dataset setup done")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

n_gt = 0
correct = 0

print('Compute mAP...')

outputs = []
targets = None
APs = []
for batch_i, (_, imgs, targets) in enumerate(dataloader):
    if batch_i == opt.batch_count:
        break

    imgs = Variable(imgs.type(Tensor))
    targets = targets.type(Tensor)

    with torch.no_grad():
        output = model(imgs)
        output = non_max_suppression(output, num_classes, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

    # Compute average precision for each sample
    for sample_i in range(targets.size(0)):
        correct = []

        # Get labels for sample where width is not zero (dummies)
        annotations = targets[sample_i, targets[sample_i, :, 3] != 0]
        # Extract detections
        detections = output[sample_i]

        if detections is None:
            # If there are no detections but there are annotations mask as zero AP
            if annotations.size(0) != 0:
                APs.append(0)
            continue

        # Get detections sorted by decreasing confidence scores
        detections = detections[np.argsort(-detections[:, 4])]

        # If no annotations add number of detections as incorrect
        if annotations.size(0) == 0:
            correct.extend([0 for _ in range(len(detections))])
        else:
            # Extract target boxes as (x1, y1, x2, y2)
            target_boxes = torch.FloatTensor(annotations[:, 1:].shape)
            target_boxes[:, 0] = (annotations[:, 1] - annotations[:, 3] / 2)
            target_boxes[:, 1] = (annotations[:, 2] - annotations[:, 4] / 2)
            target_boxes[:, 2] = (annotations[:, 1] + annotations[:, 3] / 2)
            target_boxes[:, 3] = (annotations[:, 2] + annotations[:, 4] / 2)
            target_boxes *= img_size

            detected = []
            for *pred_bbox, conf, obj_conf, obj_pred in detections:

                pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                # Compute iou with target boxes
                iou = bbox_iou(pred_bbox, target_boxes)
                # Extract index of largest overlap
                best_i = np.argmax(iou)
                # If overlap exceeds threshold and classification is correct mark as correct
                if iou[best_i] > opt.iou_thres and obj_pred == annotations[best_i, 0] and best_i not in detected:
                    correct.append(1)
                    detected.append(best_i)
                else:
                    correct.append(0)

        # Extract true and false positives
        true_positives = np.array(correct)
        false_positives = 1 - true_positives

        # Compute cumulative false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # Compute recall and precision at all ranks
        recall = true_positives / annotations.size(0) if annotations.size(0) else true_positives
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # Compute average precision
        AP = compute_ap(recall, precision)
        APs.append(AP)

        print("+ Sample [%d/%d] AP: %.4f (%.4f)" % (len(APs), len(dataset), AP, np.mean(APs)))

print("Mean Average Precision: %.4f" % np.mean(APs))
