import pickle
import matplotlib.pyplot as plt
import argparse

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

parser = argparse.ArgumentParser()
parser.add_argument("--pickle", type=str, required=True)
parser.add_argument("--bins", type=int, required=True)
parser.add_argument("--fig", type=str, required=True)
args = parser.parse_args()

d = pickle.load(open(args.pickle, "rb"))

d = [x[1] for x in d]

l, h = min(d) - 5, max(d) + 5

plt.figure(figsize=(6, 5))
n, bins, patches = plt.hist(d, bins=args.bins, range=[l, h], facecolor="blue")
plt.axis([40, 110, 0, max(n)])
# plt.xticks(range(20, 180, 10))
# plt.yticks(range(0, int(max(n)) + 2, 250))
# plt.grid(True)
plt.xlabel("Color average")
plt.ylabel("Number of frames")
plt.tight_layout()
plt.savefig(args.fig)
