import argparse

from PIL import Image

"""
Returns a subset of images (from --imgs_txt) where objects from --cls
have ratios >= --w_thr, --h_thr
"""

parser = argparse.ArgumentParser()
parser.add_argument("--imgs_txt", type=str, required=True, help="txt file with image paths")
parser.add_argument("--cls", type=int, required=True, help="target class")
parser.add_argument("--w_thr", type=float, default=0, help="width threshold (0,1)")
parser.add_argument("--h_thr", type=float, default=0, help="height threshold (0,1)")
args = parser.parse_args()

images = [x.strip() for x in open(args.imgs_txt, "r").readlines()]
labels = [x.replace("png", "txt").replace("jpg", "txt").replace("images", "labels") for x in images]

w, h = Image.open(images[0]).size

labels = {
	i: [
		list(map(float, x.strip().split(" "))) for x in open(labels[i], "r")
	] for i in range(len(labels))
}

selected_cls = {
	i: [x for x in labels[i] if x[0] == args.cls] for i in range(len(labels))
}

boxes = {
	i: [
		[int(x[0]), (x[1] - x[3] / 2) * w, (x[2] - x[4] / 2) * h, (x[1] + x[3] / 2) * w, (x[2] + x[4] / 2) * h]
		for x in selected_cls[i]
	] for i in selected_cls.keys()
}

norm_areas = {
	i: [
		((x[3] - x[1]) / w, (x[4] - x[2]) / h) for x in boxes[i]
	]
	for i in boxes.keys()
}

selected_areas = {
	i: [
		x for x in norm_areas[i] if x[0] >= args.w_thr and x[1] >= args.h_thr
	] for i in norm_areas.keys()
}

corr_imgs = {
	i: selected_areas[i] for i in selected_areas.keys() if selected_areas[i] != []
}

for cimg in corr_imgs:
	print(images[cimg])
