import json
import os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# parser = argparse.ArgumentParser()
# parser.add_argument("--data_path", type=str, required=True)
# args = parser.parse_args()
#
# data_cfg = parse_data_config(args.data_path)

path = "/home/alex/workspace/thesis/data/bdd/labels/100k"
dirs = ["train", "val"]

# attributes = weather / scene / timeofday

count = {
    "weather": defaultdict(lambda: 0),
    "scene": defaultdict(lambda: 0),
    "timeofday": defaultdict(lambda: 0),
}

what = "/" + dirs[0]

for what in dirs:
    what = "/" + what
    for f in tqdm(os.listdir(path + what)):
        j = json.load(open(path + what + "/" + f, "rt"))
        w = j["attributes"]["weather"]
        s = j["attributes"]["scene"]
        t = j["attributes"]["timeofday"]
        count["weather"][w] += 1
        count["scene"][s] += 1
        count["timeofday"][t] += 1

# fp = open("./bdd_persons.txt", "wt")
# ccc = defaultdict(lambda: 0)
# for what in dirs:
# 	what = "/" + what
# 	for f in os.listdir(path + what):
# 		j = json.load(open(path + what + "/" + f, "rt"))
#
# 		obj = j['frames'][0]['objects']
# 		for x in obj:
# 			c = x['category']
# 			if c in ["bike", "bus", "car", "motor", "person", "rider", "traffic light", "traffic sign", "train",
# 					 "truck"]:
# 				ccc[c] += 1
# ped = sum([1 for x in obj if x['category'] == 'person'])

# f = f.split(".")[0]
# if ped >= 20:
# 	fp.write(
# 		path.replace("labels", "images") + "/" + what + "/" + f + ".jpg\n"
# 	)
# fp.close()

# ax = plt.subplot()
# values = [(a, ccc[a]) for a in list(ccc.keys())]
# x, y = [a for (a, _) in values], [a for (_, a) in values]
# ax.bar(range(len(y)), y)
# ax.set_xticks(range(len(y)))
# ax.set_xticklabels(x)
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# ax.set_title("Number of instances")
# plt.tight_layout()
# plt.show()

fig = plt.subplots(1, 3, figsize=(16, 6))
for i, t in enumerate(count.keys()):
    values = [(a, count[t][a]) for a in list(count[t].keys())]
    x, y = [a for (a, _) in values], [a for (_, a) in values]
    ax = plt.subplot(1, 3, i + 1)
    ax.bar(range(len(y)), y)
    ax.set_xticks(range(len(y)))
    ax.set_xticklabels(x)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(t)
plt.tight_layout()
plt.show()

for t in count.keys():
    s = "%10s: " % t
    for x, y in count[t].items():
        s += "[%15s %6d] " % (x, y)
    print(s)
