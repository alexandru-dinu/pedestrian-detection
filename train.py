from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=30, help='number of num_epochs')
parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--data_config_path', type=str, default='config/coco.data', help='path to data config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=1)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--use_cuda', action="store_true", help='whether to use cuda if available')
parser.add_argument('--shuffle', action="store_true")
opt = parser.parse_args()

for x in opt.__dict__:
	print("%25s: %s" % (x, opt.__dict__[x]))
print("-" * 80)

cuda = torch.cuda.is_available() and opt.use_cuda
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

os.makedirs('checkpoints', exist_ok=True)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config['train']
names = load_names(data_config['names'])

for x, y in data_config.items():
	print("%25s: %s" % (x, y))
print("-" * 80)

# Model loading
model = Darknet(opt.model_config_path)
# model.load_weights(opt.weights_path)
model.apply(weights_init_normal)

model.cuda()
model.train()

for x, y in model.hyperparams.items():
	print("%25s: %s" % (x, y))
print("-" * 80)
print("Model loading done")

# Get hyper parameters
learning_rate = float(model.hyperparams['learning_rate'])
momentum = float(model.hyperparams['momentum'])
decay = float(model.hyperparams['decay'])
burn_in = int(model.hyperparams['burn_in'])
batch_size = int(model.hyperparams['batch'])
img_size = int(model.hyperparams['height'])

# Get dataloader
dataset = ListDataset(train_path, img_size=img_size)
dataloader = torch.utils.data.DataLoader(
	dataset,
	batch_size=batch_size, shuffle=opt.shuffle, num_workers=opt.n_cpu)
print("Dataset setup done")

# Setup optimizer TODO
optimizer = optim.SGD(
	model.parameters(),
	lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay
)

# Perform training
for epoch in range(opt.num_epochs):
	for batch_i, (_, imgs, targets) in enumerate(dataloader):
		imgs = Variable(imgs.type(Tensor))
		targets = Variable(targets.type(Tensor), requires_grad=False)

		optimizer.zero_grad()

		loss = model(imgs, targets)

		loss.backward()
		optimizer.step()

		print(
			'[Epoch %2d/%2d, Batch %5d/%5d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
			(epoch, opt.num_epochs, batch_i, len(dataloader),
			 model.losses['x'], model.losses['y'], model.losses['w'],
			 model.losses['h'], model.losses['conf'], model.losses['cls'],
			 loss.item(), model.losses['recall']))

		model.seen += imgs.size(0)

	# TODO: validate model

	if epoch % opt.checkpoint_interval == 0:
		model.save_weights('%s/%d.weights' % (opt.checkpoint_dir, epoch))
