import argparse
import json
import os
from collections import defaultdict

from utils.parse_config import parse_data_config

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

data_cfg = parse_data_config(args.data_path)

path = "/home/alex/thesis-workspace/data/bdd/labels/100k"
dirs = ["train", "val"]

# attributes = weather / scene / timeofday

count = {
	"weather"  : defaultdict(lambda: 0),
	"scene"    : defaultdict(lambda: 0),
	"timeofday": defaultdict(lambda: 0)
}

for f in os.listdir(path + "/val"):
	j = json.load(open(path + "/val/" + f, "rt"))
	w = j["attributes"]["weather"]
	s = j["attributes"]["scene"]
	t = j["attributes"]["timeofday"]
	count["weather"][w] += 1
	count["scene"][s] += 1
	count["timeofday"][t] += 1

for t in count.keys():
	s = "%10s: " % t
	for x, y in count[t].items():
		s += "[%15s %6d] " % (x, y)

	print(s)
