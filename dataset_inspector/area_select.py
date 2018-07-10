import argparse
from PIL import Image
import numpy as np

"""
Returns a subset of images (from --imgs_txt) where objects from classes --cls have ratios >= --w_thr, --h_thr
If w/h thr are not specified, then all objs from --cls are returned
"""

parser = argparse.ArgumentParser()
parser.add_argument("--imgs_txt", type=str, required=True, help="txt file with image paths")
parser.add_argument("--cls", nargs="+", type=int, required=True, help="target classes")
parser.add_argument("--names", type=str)
parser.add_argument("--w_thr", nargs="*", type=float, default=0, help="width threshold (0,1)")
parser.add_argument("--h_thr", nargs="*", type=float, default=0, help="height threshold (0,1)")
args = parser.parse_args()

names = [x.strip() for x in open(args.names, "r").readlines()]
num_classes = len(names)

images = [x.strip() for x in open(args.imgs_txt, "r").readlines()]
labels = [x.replace("png", "txt").replace("jpg", "txt").replace("images", "labels") for x in images]
num_images = len(labels)

w, h = Image.open(images[0]).size

# extract labels for each image as a list of [cls, cx, cy, pw, ph]
labels = {
	i: [
		list(map(float, x.strip().split(" "))) for x in open(labels[i], "r")
	] for i in range(num_images)
}

# separate labels per each class
# [class][image] = list of [class, cx, cy, pw, ph]
# class c in image i <=> [c][i] != []
labels_per_cls = {}
for cls in args.cls:
	labels_per_cls[cls] = {
		i: [x for x in labels[i] if x[0] == cls] for i in range(num_images)
	}

# convert to x1,y1,x2,y2
boxes = {}
for cls in args.cls:
	boxes[cls] = {
		i: [
			[x[0], (x[1] - x[3] / 2) * w, (x[2] - x[4] / 2) * h, (x[1] + x[3] / 2) * w, (x[2] + x[4] / 2) * h]
			for x in labels_per_cls[cls][i]
		] for i in range(num_images)
	}

norm_areas = {}
for cls in args.cls:
	norm_areas[cls] = {
		i: [
			((x[3] - x[1]) / w, (x[4] - x[2]) / h) for x in boxes[cls][i]
		] for i in range(num_images)
	}

selected_areas = {}
for cls in args.cls:
	selected_areas[cls] = {
		i: [
			x for x in norm_areas[cls][i] if x[0] >= args.w_thr and x[1] >= args.h_thr
		] for i in norm_areas[cls].keys()
	}

num_instances = {}
for cls in args.cls:
	num_instances[cls] = sum(map(len, selected_areas[cls].values()))

corr_imgs = {}
for cls in args.cls:
	corr_imgs[cls] = {
		i: selected_areas[cls][i] for i in range(num_images) if selected_areas[cls][i] != []
	}


def get_num_instances():
	for cls in args.cls:
		print("%20s: %6d" % (names[cls], num_instances[cls]))


def get_avg_areas():
	means = {cls: 0 for cls in range(num_classes)}

	for cls in args.cls:
		count = 0

		for i in range(num_images):
			objs = selected_areas[cls][i]
			if not objs: continue
			for o in objs:
				means[cls] += o[0] * o[1]
				count += 1

		means[cls] /= count
		print(
			"%20s: [ratio %f] [square %f]" %
			(names[cls], means[cls] * 100, np.sqrt(means[cls] * w * h))
		)


def get_correlated_imgs():
	for cls in args.cls:
		for i in corr_imgs[cls]:
			print(images[i])


if __name__ == "__main__":
	get_correlated_imgs()
