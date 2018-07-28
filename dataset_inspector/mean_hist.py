import pickle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pickle", type=str, required=True)
parser.add_argument("--bins", type=int, required=True)
parser.add_argument("--fig", type=str, required=True)
args = parser.parse_args()

d = pickle.load(open(args.pickle, "rb"))

d = [x[1] for x in d]

l, h = min(d) - 5, max(d) + 5

plt.figure(figsize=(12, 10))
n, bins, patches = plt.hist(d, bins=args.bins, range=[l, h], facecolor='green')
plt.axis([l, h, 0, max(n) + 2])
plt.xticks(range(0, int(max(d)), 10))
plt.yticks(range(0, int(max(n)) + 2, 250))
plt.grid(True)
plt.tight_layout()
plt.savefig(args.fig)
