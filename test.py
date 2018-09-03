from __future__ import division

import argparse

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset_names_conv import *
from models import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *
import seaborn as sea

parser = argparse.ArgumentParser()
parser.add_argument('--batch_count', type=int, default=10, help="num batches to pass through")
parser.add_argument('--model_config_path', type=str, required=True, help='path to model config file')
parser.add_argument('--data_config_path', type=str, required=True, help='path to data config file')
parser.add_argument('--weights_path', type=str, required=True, help='path to weights file')
parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.45, help='iou thresshold for non-maximum suppression')
parser.add_argument('--area_thres', type=float, default=0.05, help='ignore objs with area < threshold')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--use_cuda', action="store_true", help='whether to use cuda if available')
parser.add_argument('--shuffle', action="store_true")
opt = parser.parse_args()

for x in opt.__dict__:
	print("%25s: %s" % (x, opt.__dict__[x]))
print("-" * 80)

cuda = torch.cuda.is_available() and opt.use_cuda
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Get data configuration
data_config = parse_data_config(opt.data_config_path)

# perform testing on these images
test_path = data_config['valid']

# matches the trained weights
num_train_names = int(data_config['classes'])
train_names = load_names(data_config['names'])

# needed if testing on a different dataset
names_conv_needed = bool(data_config.get("names_conv", False))
if names_conv_needed:
	valid_names = load_names(data_config["valid_names"])
	names_conv_dict = eval(data_config["conv_dict"])
else:
	valid_names = train_names
	names_conv_dict = None

assert num_train_names == len(train_names)

for x, y in data_config.items():
	print("%25s: %s" % (x, y))
print("-" * 80)

# Model loading
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)
model.cuda()
model.eval()

batch_size = int(model.hyperparams['batch'])
img_size = int(model.hyperparams['height'])

for x, y in model.hyperparams.items():
	print("%25s: %s" % (x, y))
print("-" * 80)
print("Model loading done")

# Dataset setup
dataset = ListDataset(test_path, img_size=img_size)
dataloader = torch.utils.data.DataLoader(
	dataset,
	batch_size=batch_size, shuffle=opt.shuffle, num_workers=opt.n_cpu
)

real_img_w, real_img_h = dataset.get_real_img_shape()
real_img_ratio = max(real_img_w, real_img_h) / min(real_img_w, real_img_h)
print("Dataset setup done")

print('Compute mAP...')

# stats-keeping
outputs, APs = [], []
targets = None

all_scores = {
	cls: {x: 0 for x in ["tp", "fp", "fn"]} for cls in valid_names
}

all_instances = {
	x: 0 for x in range(len(valid_names))
}

errors = {
	cls: {
		# error sources of FPs
		'fp': {y: 0 for y in ['no_gt', 'iou', 'cls', 'mult']},
		# error sources of FNs: will append areas of FNs
		'fn': []
	}
	for cls in valid_names
}

confusion = np.zeros((len(valid_names), len(valid_names)))


def do_name_conversion(det):
	for n in range(num_train_names):
		if det.size(0) == 0: break

		# there exists a class conversion for class n from X to Y
		if names_conv_dict[n] != -1:
			aux = det[det[..., -1] == n]
			if aux.size(0) == 0: continue
			aux[..., -1] = names_conv_dict[n]
			det[det[..., -1] == n] = aux
		# no class conversion for class n, remove detections
		else:
			det = det[det[..., -1] != n]

	return det


def save_confusion_matrix(cm):
	labels = valid_names

	ocm = np.copy(cm)

	for i in range(len(valid_names)):
		if all_instances[i] == 0:
			cm[i, :] = 0  # TODO
		else:
			cm[i, :] /= all_instances[i]

	fig, ax = plt.subplots(figsize=(6, 6))
	im = ax.imshow(cm)

	fig.colorbar(im, ax=ax)

	ax.set_xticks(np.arange(len(labels)))
	ax.set_yticks(np.arange(len(labels)))
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# for i in range(len(valid_names)):
	# 	for j in range(len(valid_names)):
	# 		s = "" if ocm[i, j] == 0 else round(ocm[i, j] / all_instances[i], 3)
	# 		ax.text(j, i, s, ha="center", va="center", color="m", fontsize='smaller')
	# print(ocm)
	# print(all_instances)

	fig.tight_layout()
	plt.savefig("cm.png")


# perform testing
for batch_i, (paths, imgs, targets) in enumerate(dataloader):
	out_str = ""

	if batch_i == opt.batch_count:
		break

	imgs = Variable(imgs.type(Tensor))
	targets = targets.type(Tensor)

	with torch.no_grad():
		output = model(imgs)
		output = non_max_suppression(output, num_train_names, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

	# Compute average precision for each sample
	for sample_i in range(targets.size(0)):
		correct = []

		# Get labels for sample where width is not zero (dummies)
		# annotation = Nx5
		annotations = targets[sample_i, targets[sample_i, :, 3] != 0]

		# remove annotations with area < thr
		if annotations.size(0) > 0:
			areas = annotations[:, 3] * annotations[:, 4]
			annotations = annotations[areas >= opt.area_thres]

		annotation_count = annotations.size(0)

		# Extract detections
		detections = output[sample_i]

		if detections is None:
			# If there are no detections, but there are annotations with area >= thr, mark as zero AP
			if annotation_count != 0:
				for a in annotations:
					all_scores[valid_names[int(a[0])]]['fn'] += 1
					# append area of fn
					errors[valid_names[int(a[0])]]['fn'].append(a[3] * a[4] * real_img_ratio)
				APs.append(0)
			continue

		# Get detections sorted by decreasing confidence scores
		detections = detections[np.argsort(-detections[:, 4])]
		if names_conv_needed:
			detections = do_name_conversion(detections)

		# used to keep track of FNs
		current_tps_per_class = defaultdict(lambda: 0)
		num_instances = {x: 0 for x in range(len(valid_names))}
		if annotation_count > 0:
			for i in range(len(valid_names)):
				num_instances[i] = annotations[annotations[:, 0] == i].shape[0]
				all_instances[i] += num_instances[i]
		assert sum(num_instances.values()) == annotation_count

		# will contain each detected gt
		detected = []

		# If no annotations add number of detections as incorrect
		if annotation_count == 0:
			correct.extend([0 for _ in range(len(detections))])
			for d in detections:
				x1, y1, x2, y2 = Tensor(d[:4]).view(1, -1)[0]
				area = (x2 - x1) * (y2 - y1) / (img_size * img_size)
				if area < opt.area_thres: continue

				all_scores[valid_names[int(d[-1])]]['fp'] += 1
				errors[valid_names[int(d[-1])]]['fp']['no_gt'] += 1
		else:
			# Extract target boxes as (x1, y1, x2, y2)
			target_boxes = torch.FloatTensor(annotations[:, 1:].shape)
			target_boxes[:, 0] = (annotations[:, 1] - annotations[:, 3] / 2)
			target_boxes[:, 1] = (annotations[:, 2] - annotations[:, 4] / 2)
			target_boxes[:, 2] = (annotations[:, 1] + annotations[:, 3] / 2)
			target_boxes[:, 3] = (annotations[:, 2] + annotations[:, 4] / 2)
			target_boxes *= img_size

			for *pred_bbox, conf, obj_conf, obj_pred in detections:
				pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)

				# ignore detections with area < thr
				x1, y1, x2, y2 = pred_bbox[0]
				area = (x2 - x1) * (y2 - y1) / (img_size * img_size)
				if area < opt.area_thres: continue

				# Compute iou with target boxes
				iou = bbox_iou(pred_bbox, target_boxes)
				# Extract index of largest overlap
				best_i = np.argmax(iou)

				# if best_i not in detected:
				# 	x, y = int(annotations[best_i, 0].item()), int(obj_pred)
				# 	confusion[x, y] += 1

				# If overlap exceeds threshold and classification is correct mark as correct
				if iou[best_i] >= opt.iou_thres and obj_pred == annotations[best_i, 0] and best_i not in detected:
					correct.append(1)
					detected.append(best_i)
					all_scores[valid_names[int(obj_pred)]]['tp'] += 1
					current_tps_per_class[int(obj_pred)] += 1

					zz = int(obj_pred)
					confusion[zz, zz] += 1
				else:
					correct.append(0)
					all_scores[valid_names[int(obj_pred)]]['fp'] += 1
					errors[valid_names[int(obj_pred)]]['fp']['iou'] += int(iou[best_i] < opt.iou_thres)
					errors[valid_names[int(obj_pred)]]['fp']['cls'] += int(obj_pred != annotations[best_i, 0])
					errors[valid_names[int(obj_pred)]]['fp']['mult'] += int(best_i in detected)

					if obj_pred != annotations[best_i, 0]:
						zz, zx = int(annotations[best_i, 0].item()), int(obj_pred)
						confusion[zz, zx] += 1

		for cls in range(len(valid_names)):
			all_scores[valid_names[cls]]['fn'] += num_instances[cls] - current_tps_per_class[cls]

		for a in range(annotation_count):
			if a not in detected:
				ar = annotations[a, 3] * annotations[a, 4] * real_img_ratio
				cls = int(annotations[a, 0])
				errors[valid_names[cls]]['fn'].append(ar)

		# Extract TP/FP
		true_positives = np.array(correct)
		false_positives = 1 - true_positives

		# how many tp/fp/fn are in current sample
		current_tp, current_fp = true_positives[true_positives == 1].size, true_positives[true_positives == 0].size
		current_fn = annotation_count - current_tp

		# Compute cumulative false positives and true positives
		true_positives = np.cumsum(true_positives)
		false_positives = np.cumsum(false_positives)

		# Compute recall and precision at all ranks
		precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
		recall = true_positives / annotation_count if annotation_count else true_positives

		# Compute average precision
		AP = compute_ap(recall, precision)
		APs.append(AP)

		# print stats for current sample
		out_str += "+ Sample [%5d/%5d] Paths: %s\n" % (len(APs), len(dataset), paths)
		out_str += "+ AP: %.4f (%.4f)\n" % (AP, np.mean(APs))
		out_str += "+ [tp %2d] [fp %2d] [fn %2d] [all %2d]\n" % (current_tp, current_fp, current_fn, annotation_count)

		for cls, res in all_scores.items():
			tp, fp, fn = res['tp'], res['fp'], res['fn']
			err_fp, err_fn = errors[cls]['fp'], errors[cls]['fn']

			p = 1.0 if fp == 0 else tp / (tp + fp)
			r = 1.0 if fn == 0 else tp / (tp + fn)
			f1 = 0 if p + r == 0 else 2 * p * r / (p + r)

			stats_str = "\t%15s: [tp %5d] [fp %5d] [fn %5d] [p %.3f] [r %.3f] [f1 %.3f] " + \
						"[no_gt %5d] [iou %5d] [cls %5d] [mult %5d] [area %f: sqrt %4.6f]\n"
			out_str += stats_str % (
				cls, tp, fp, fn, p, r, f1,
				err_fp['no_gt'], err_fp['iou'], err_fp['cls'], err_fp['mult'],
				100.0 * np.mean(err_fn), np.sqrt(np.mean(err_fn)*real_img_w*real_img_h)
			)

	print(out_str)

save_confusion_matrix(confusion)
print("Confusion matrix done")

print("Mean Average Precision: %f" % np.mean(APs))
