import json
import os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import code
import sys
import numpy as np

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

path = "/home/alex/workspace/thesis/datasets/udacity/train.txt"

lbls = [x.strip().replace("images", "labels").replace("jpg", "txt") for x in open(path, "rt")]

# get ped count
labels = {i: [list(map(float, x.strip().split(" "))) for x in open(lbls[i]).readlines()] for i in range(len(lbls))}

lls = ["biker", "car", "pedestrian", "trafficLight", "truck"]

# COUNT PED
# ped_only = [0] * len(labels)
# for i in range(len(labels)):
# 	for l in labels[i]:
# 		if l[0] == 2:
# 			ped_only[i] += 1
#
# ped_only = list(filter(lambda x: x > 0, ped_only))
#
# ax = plt.subplots()
# plt.hist(ped_only, bins=int(sys.argv[1]), range=[min(ped_only), max(ped_only)], facecolor='blue')
# plt.xticks(np.arange(0, max(ped_only), 1.0))
# # plt.xlabel("Pedestrian height (px)")
# # plt.ylabel("Number of instances")
# plt.show()

# NUM INSTANCES
inst = defaultdict(lambda: 0)
for i in range(len(labels)):
	for l in labels[i]:
		inst[lls[int(l[0])]] += 1

ax = plt.subplot()
values = [(a, inst[a]) for a in list(inst.keys())]
x, y = [a for (a, _) in values], [a for (_, a) in values]
ax.bar(range(len(y)), y)
ax.set_xticks(range(len(y)))
ax.set_xticklabels(x)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_title("Number of instances")
plt.tight_layout()
plt.show()
