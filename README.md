# PyTorch-YOLOv3

This is my **main** YOLOv3 workspace.

## Table of Contents
* [Paper](#paper)
* [Algorithm description](#algorithm-description)
* [Datasets](#datasets)
* [Credits](#credits)


## Paper
### YOLOv3: An Incremental Improvement

[Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf),  [Darknet Implementation](https://github.com/pjreddie/darknet)

_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5 AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.



## Algorithm description
### Input
- `w x h` image reshaped to `608 x 608`
- subdivisions = number of mini-batches a batch is split in
- images per mini-batch = `batch_size / subdivisions`, get sent to GPU

### Output
- multi-scale detection is used
- `strides = img.size / scales` -- `[32, 16, 8] = 416 / [13, 26, 52]`
- vanilla output: `scale x scale x (3 * (5 + C))` tensor
- the output is actually merged as `batch_size x (3 * (sum(scale^2))) x (5 + C)`

### Idea
- split the input image into an `S x S` grid
- each grid cell is assigned a number of anchors (3) of various sizes, depending on the scale
(see [Anchors](#anchors))
- YOLO doesn't predict the absolute coordinates of the bounding box's center,
instead, it predicts offsets, which are:
    - relative to the top left corner of the grid cell which is predicting the object
    - normalised by the dimension of the cell from the feature map (`scale x scale`)
- example: on a 13x13 grid, if a prediction of cell (6,3) is (0.4, 0.7), it means that
on the 13x13 feature map we get (6.4, 3.7)
- bounding box format: `[x, y, w, h, conf, classes]`

### Anchors
- using multi-scale detection, for vanilla YOLOv3 we have:
    - `scale = 13 x 13, stride = 32`, anchors = `116,90,  156,198,  373,326`
    - `scale = 26 x 26, stride = 16`, anchors = `30,61,  62,45,  59,119`
    - `scale = 52 x 52, stride = 8`, anchors = `10,13,  16,30,  33,23`
- all anchors are `w,h` boxes (from `416 x 416`)

### Non-maximum suppression
```
convert predicted bboxes from `[cx, cy, pw, ph]` to `[x1, y1, x2, y2]`

for each prediction in current batch
    filter out obj_conf < conf_thr (1)
    get classes (prob, class) with highest probability for each detection with (1) satisfied
    get all unique predicted classes for current prediction

    for each unique predicted class c
        det_cls = all detections for current prediction which have class == c
        sort det_cls in descending order by obj_conf
        get detection with highest confidence (det_cls[0]) and save as max detection

        compute ious between max detection and rest of detections for current class,
        in order to allow other objects with same class to be detected
```

### Building targets
- at a given scale (s), we have an `s x s` grid of cells
- to each cell is assigned a number of 3 anchors (depending on `s`: the larger the scale, the larger the anchors' area)
- for a ground truth, a target is constructed by finding the cell which contains the center of the ground truth bounding box;
next, find the best overlapping anchor (one of the 3 default anchors ar this scale) and say
`mask[b, a, cy, cx] = 1`, i.e.: for batch `b`, the cell `cy, cx` is responsible for predicting the given ground truth,
and the best overlapping anchor's index is `a`
- network predicts `tx, ty, tw, th`; the final prediction is `bx, by, bw, bh` where the conversion formulas are:
`bx = sigma(tx) + cx, by = sigma(ty) + cy, bw = pw * exp(tw), bh = ph * exp(th)`

## Datasets
- converted labels format: `<frame_id>.txt : [class rx ry rw rh]`
	- `rx, ry` are the center `x, y` coordinates,
    - `rw, rh` are the width and height
    - all of them are normalized to image width / height
- `(train | test | val).txt` file containing pairs `(<image_i>.txt, <labels_i>.txt)`
- each raw dataset provides its custom label format -> needs conversion to yolo format
- write `<name>_dataset_prepare.py` script that converts and generates the needed `dataset_txts`

## Credits
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

- [@eriklindernoren](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- [@ayooshkathuria](https://github.com/ayooshkathuria/pytorch-yolo-v3).
