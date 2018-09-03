import json
import os
from collections import defaultdict

# parser = argparse.ArgumentParser()
# parser.add_argument("--data_path", type=str, required=True)
# args = parser.parse_args()
#
# data_cfg = parse_data_config(args.data_path)

path = "/home/alex/workspace/thesis/data/bdd/labels/100k"
dirs = ["train", "val"]

# attributes = weather / scene / timeofday

count = {
	"weather"  : defaultdict(lambda: 0),
	"scene"    : defaultdict(lambda: 0),
	"timeofday": defaultdict(lambda: 0)
}

what = "/train"

# for f in os.listdir(path + what):
# 	j = json.load(open(path + what + "/" + f, "rt"))
# 	w = j["attributes"]["weather"]
# 	s = j["attributes"]["scene"]
# 	t = j["attributes"]["timeofday"]
# 	count["weather"][w] += 1
# 	count["scene"][s] += 1
# 	count["timeofday"][t] += 1

fp = open("./bdd_persons.txt", "wt")
for f in os.listdir(path + what):
	j = json.load(open(path + what + "/" + f, "rt"))

	obj = j['frames'][0]['objects']
	ped = len([1 for x in obj if x['category'] == 'person'])
	if ped >= 50:
		print(f)
fp.close()

for t in count.keys():
	s = "%10s: " % t
	for x, y in count[t].items():
		s += "[%15s %6d] " % (x, y)

	print(s)
