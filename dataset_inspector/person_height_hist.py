import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--imgs_txt", type=str, required=True, help="txt file with image paths"
)
parser.add_argument("--cls", nargs="+", type=int, required=True, help="target classes")
parser.add_argument("--bins", type=int, required=True)
args = parser.parse_args()

images = [x.strip() for x in open(args.imgs_txt, "r").readlines()]
labels = [
    x.replace("png", "txt").replace("jpg", "txt").replace("images", "labels")
    for x in images
]
num_images = len(labels)

w, h = Image.open(images[0]).size

# extract labels for each image as a list of [cls, cx, cy, pw, ph]
labels = {
    i: [list(map(float, x.strip().split(" "))) for x in open(labels[i], "r")]
    for i in range(num_images)
}

# separate labels per each class
# [class][image] = list of [class, cx, cy, pw, ph]
# class c in image i <=> [c][i] != []
labels_per_cls = {}
for cls in args.cls:
    labels_per_cls[cls] = {
        i: [x for x in labels[i] if x[0] == cls] for i in range(num_images)
    }

areas = {}
for cls in args.cls:
    areas[cls] = {
        i: [(x[4] * h) for x in labels_per_cls[cls][i]] for i in range(num_images)
    }

ped_only = []
for cls in args.cls:
    for i in range(num_images):
        if not areas[cls][i]:
            continue
        ped_only.extend(areas[cls][i])


import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

ax = plt.subplots()
plt.hist(ped_only, bins=args.bins, range=[min(ped_only), 240], facecolor="blue")
plt.xticks(np.arange(0, 240, 20.0))
plt.xlabel("Pedestrian height (px)")
plt.ylabel("Number of instances")
plt.show()
