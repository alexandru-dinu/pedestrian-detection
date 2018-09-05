from collections import defaultdict

coco_names_path = "cfgs/coco/coco.names"
coco_names = [
	x.strip() for x in open(coco_names_path, "r")
]

udacity_names = [
	"biker", "car", "pedestrian", "trafficLight", "truck"
]
crowdai_names = [
	"car", "pedestrian", "truck"
]
bdd_names = [
	"bike", "bus", "car", "motor", "person", "rider", "traffic light", "traffic sign", "train", "truck"
]
citypersons_names = [
	"fake", "pedestrian", "rider", "sitting", "unusual", "group"
]


def coco_to_udacity():
	conv = defaultdict(lambda: -1)

	# conv[coco_names.index("bicycle")] = udacity_names.index("biker")
	conv[coco_names.index("car")] = crowdai_names.index("car")
	conv[coco_names.index("person")] = crowdai_names.index("pedestrian")
	# conv[coco_names.index("traffic light")] = udacity_names.index("trafficLight")
	conv[coco_names.index("truck")] = crowdai_names.index("truck")

	return conv


def coco_to_bdd():
	conv = defaultdict(lambda: -1)
	# N/A: rider, traffic sign

	conv[coco_names.index("bicycle")] = bdd_names.index("bike")
	conv[coco_names.index("bus")] = bdd_names.index("bus")
	conv[coco_names.index("car")] = bdd_names.index("car")
	conv[coco_names.index("motorbike")] = bdd_names.index("motor")
	conv[coco_names.index("person")] = bdd_names.index("person")
	conv[coco_names.index("traffic light")] = bdd_names.index("traffic light")
	conv[coco_names.index("train")] = bdd_names.index("train")
	conv[coco_names.index("truck")] = bdd_names.index("truck")

	return conv


def coco_to_crowdai():
	conv = defaultdict(lambda: -1)

	conv[coco_names.index("car")] = crowdai_names.index("car")
	conv[coco_names.index("person")] = crowdai_names.index("pedestrian")
	conv[coco_names.index("truck")] = crowdai_names.index("truck")

	return conv


def coco_to_citypersons():
	conv = defaultdict(lambda: -1)

	conv[coco_names.index("person")] = citypersons_names.index("pedestrian")

	return conv


def bdd_to_crowdai():
	conv = defaultdict(lambda: -1)

	conv[bdd_names.index("person")] = crowdai_names.index("pedestrian")
	conv[bdd_names.index("car")] = crowdai_names.index("car")
	conv[bdd_names.index("truck")] = crowdai_names.index("truck")

	return conv


def citypersons_to_crowdai():
	conv = defaultdict(lambda: -1)

	conv[citypersons_names.index("pedestrian")] = crowdai_names.index("pedestrian")

	return conv


def udacity_to_crowdai():
	conv = defaultdict(lambda: -1)

	for n in udacity_names:
		if n in crowdai_names:
			conv[udacity_names.index(n)] = crowdai_names.index(n)

	return conv
