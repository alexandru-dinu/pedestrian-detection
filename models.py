from __future__ import division

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.parse_config import *
from utils.utils import build_targets


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(
                scale_factor=int(module_def["stride"]), mode="nearest"
            )
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module("route_%d" % i, EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_dim = img_dim

        # from paper
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

        self.mse_loss = nn.MSELoss().cuda()
        self.bce_loss = nn.BCELoss().cuda()

    def forward(self, x, targets=None):
        # x.size == bs x (3*(5+C)) x scale x scale
        bs, _, scale, _ = x.size()

        stride = self.img_dim / scale
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        prediction = (
            x.view(bs, self.num_anchors, self.bbox_attrs, scale, scale)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = (
            torch.linspace(0, scale - 1, scale)
            .repeat(scale, 1)
            .repeat(bs * self.num_anchors, 1, 1)
            .view(x.shape)
            .type(FloatTensor)
        )
        grid_y = (
            torch.linspace(0, scale - 1, scale)
            .repeat(scale, 1)
            .t()
            .repeat(bs * self.num_anchors, 1, 1)
            .view(y.shape)
            .type(FloatTensor)
        )
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, scale * scale).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, scale * scale).view(h.shape)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                targets=targets.cpu().data,
                anchors=scaled_anchors,
                num_classes=self.num_classes,
                scale=scale,
            )

            nProposals = int((conf > 0.25).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1

            # Handle masks
            mask = Variable(mask.type(FloatTensor))
            cls_mask = Variable(
                mask.unsqueeze(-1)
                .repeat(1, 1, 1, 1, self.num_classes)
                .type(FloatTensor)
            )
            conf_mask = Variable(conf_mask.type(FloatTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(FloatTensor), requires_grad=False)

            # Mask outputs to ignore non-existing objects
            loss_x = self.lambda_coord * self.mse_loss(x * mask, tx * mask) / 2
            loss_y = self.lambda_coord * self.mse_loss(y * mask, ty * mask) / 2
            loss_w = self.lambda_coord * self.mse_loss(w * mask, tw * mask) / 2
            loss_h = self.lambda_coord * self.mse_loss(h * mask, th * mask) / 2

            loss_conf = self.bce_loss(
                conf * conf_mask, tconf * conf_mask
            ) + self.lambda_noobj * self.bce_loss(
                conf * (1 - conf_mask), tconf * (1 - mask)
            )

            loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask)

            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(bs, -1, 4) * stride,
                    conf.view(bs, -1, 1),
                    pred_cls.view(bs, -1, self.num_classes),
                ),
                dim=-1,
            )

            return output.data


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, freeze_point=-1):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)

        self.img_size = int(self.hyperparams["width"])
        self.freeze_point = freeze_point

        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])

        self.losses = None
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall"]

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []

        self.losses = defaultdict(float)
        layer_outputs = []

        for i, (module_def, module) in enumerate(
            zip(self.module_defs, self.module_list)
        ):
            if module_def["type"] in ["convolutional", "upsample"]:
                x = module(x)

            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)

            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        self.losses["recall"] /= 3
        return sum(output) if is_training else torch.cat(output, 1)

    def freeze_layers(self):
        print("Freeze layers\t[%3d, %3d)" % (0, self.freeze_point))

        for i in range(len(self.module_list)):
            for param in self.module_list[i].parameters():
                param.requires_grad = i >= self.freeze_point

    def init_layers(self):
        print("Init layers\t[%3d, %3d]" % (self.freeze_point, len(self.module_list)))

        for i, seq in enumerate(self.module_list[self.freeze_point :]):
            for j, m in enumerate(seq):
                classname = m.__class__.__name__
                if classname.find("Conv") != -1:
                    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
                    print("init %4s [%3d/%3d]" % ("conv", self.freeze_point + i, j))
                elif classname.find("BatchNorm2d") != -1:
                    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                    torch.nn.init.constant_(m.bias.data, 0.0)
                    print("init %4s [%3d/%3d]" % ("bn", self.freeze_point + i, j))

    def debug(self):
        for i, seq in enumerate(self.module_list):
            for j, m in enumerate(seq):
                if "YOLO" in m.__class__.__name__:
                    print("YOLO @ [%3d/%3d], num_classes = %3d" % (i, j, m.num_classes))

    def load_weights(self, weights_path, upto=-1):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        fp = open(weights_path, "rb")

        header = np.fromfile(
            fp, dtype=np.int32, count=5
        )  # First five are header values
        self.header_info = header
        self.seen = header[3]

        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(
            zip(self.module_defs, self.module_list)
        ):
            if i == upto:
                break

            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.bias
                    )
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.weight
                    )
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_mean
                    )
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_var
                    )
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        conv_layer.bias
                    )
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b

                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(
                    conv_layer.weight
                )
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_weights(self, weights_path, cutoff=0):
        # num of layers to save
        nl = cutoff if cutoff != 0 else len(self.module_list)
        modules = zip(self.module_defs[:nl], self.module_list[:nl])

        fp = open(weights_path, "wb")

        # Attach the header at the top of the file
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        for i, (module_def, module) in enumerate(modules):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]

                # If batch norm, save bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]

                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)

                # save conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)

                # save conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

    def __str__(self):
        s = "Darknet [%5s] (img_size: %3d, fp: %3d)" % (
            self.mode,
            self.img_size,
            self.freeze_point,
        )

        return s
