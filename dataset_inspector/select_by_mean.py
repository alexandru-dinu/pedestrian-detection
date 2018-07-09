import argparse
import operator
import pickle

import cv2
from tqdm import tqdm

"""
Returns a subset of images (from --imgs_txt) where objects from --cls
have ratios >= --w_thr, --h_thr
"""

parser = argparse.ArgumentParser()
parser.add_argument("--imgs_txt", type=str, required=True, help="txt file with image paths")
parser.add_argument("--dict_name", type=str, required=True, help="exported dict name")
args = parser.parse_args()

images = [x.strip() for x in open(args.imgs_txt, "r").readlines()]
labels = [x.replace("png", "txt").replace("jpg", "txt").replace("images", "labels") for x in images]

means = {}
for x in tqdm(images):
	means[x] = cv2.imread(x).mean()
means = sorted(means.items(), key=operator.itemgetter(1))

pickle.dump(means, open(args.dict_name + ".pickle", "wb"))
