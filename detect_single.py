from __future__ import division

import argparse

import cv2

from models import *
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--data_config_path', type=str, default='config/coco.data', help='path to data config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--use_cuda', action="store_true", help='whether to use cuda if available')
opt = parser.parse_args()

for x in opt.__dict__:
	print("%25s: %s" % (x, opt.__dict__[x]))
print("-" * 80)

FONT = cv2.FONT_HERSHEY_TRIPLEX
COLORS = [tuple(255 * np.array(plt.get_cmap('tab20')(i)[:-1])) for i in np.linspace(0, 1, 20)]

cuda = torch.cuda.is_available() and opt.use_cuda
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
num_classes = int(data_config['classes'])
classes = load_names(data_config['names'])

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
while True:
	test_path = input("Image path: ")
	path, img = SingleImage(test_path, img_size=img_size)[0]
	input_img = Variable(img.type(Tensor).unsqueeze(0))

	# Get detections
	with torch.no_grad():
		detections = model(input_img)
		detections = non_max_suppression(detections, num_classes, opt.conf_thres, opt.nms_thres)
		detections = detections[0]

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
			cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

			# add label
			cv2.putText(img, str(int(cls_pred)), (x1, y1 - 3), FONT, 1, (255, 255, 255), 1)

			print(
				'\t+ Coords: [%4d, %4d, %4d, %4d], Class: %s, ObjConf: %.5f, ClassProb: %.5f' % \
				(x1, y1, x2, y2, classes[int(cls_pred)], conf.item(), cls_conf.item())
			)

	# Save generated image with detections
	save_path = 'detections/single.png'
	cv2.imwrite(save_path, img)
