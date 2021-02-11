from __future__ import division

import argparse
import timeit

import torch.optim as optim
from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=30, help="number of num_epochs")
parser.add_argument(
    "--model_config_path",
    type=str,
    default="config/yolov3.cfg",
    help="path to model config file",
)
parser.add_argument(
    "--data_config_path",
    type=str,
    default="config/coco.data",
    help="path to data config file",
)
parser.add_argument(
    "--weights_path",
    type=str,
    default="weights/yolov3.weights",
    help="path to weights file",
)
parser.add_argument(
    "--freeze_point", type=int, default=-1, help="-1 to load all weights"
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=0,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument("--avg_interval", type=int, default=1)
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument(
    "--use_cuda", action="store_true", help="whether to use cuda if available"
)
parser.add_argument("--shuffle", action="store_true")
opt = parser.parse_args()

for x in opt.__dict__:
    print("%25s: %s" % (x, opt.__dict__[x]))
print("-" * 80)

cuda = torch.cuda.is_available() and opt.use_cuda
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

os.makedirs("checkpoints", exist_ok=True)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]
names = load_names(data_config["names"])

for x, y in data_config.items():
    print("%25s: %s" % (x, y))
print("-" * 80)

# Model loading
model = Darknet(config_path=opt.model_config_path, freeze_point=opt.freeze_point)
model.load_weights(opt.weights_path, upto=model.freeze_point)
model.freeze_layers()
model.init_layers()
model.cuda()
model.train()
model.debug()

for x, y in model.hyperparams.items():
    print("%25s: %s" % (x, y))
print("-" * 80)
print("Model loading done")

# Get hyper parameters
learning_rate = float(model.hyperparams["learning_rate"])
momentum = float(model.hyperparams["momentum"])
decay = float(model.hyperparams["decay"])
burn_in = int(model.hyperparams["burn_in"])
batch_size = int(model.hyperparams["batch"])
subdivisions = int(model.hyperparams["subdivisions"])
img_size = int(model.hyperparams["height"])
sub_batch_size = batch_size // subdivisions

assert batch_size % subdivisions == 0, "Wrong bs/sd config"

# Get dataloader
dataset = ListDataset(train_path, img_size=img_size)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=opt.shuffle, num_workers=opt.n_cpu
)
print("Dataset setup done")

# Setup optimizer TODO
learnable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.SGD(
    learnable_params,
    lr=learning_rate,
    momentum=momentum,
    dampening=0,
    weight_decay=decay,
)

# stats-keeping
avg_losses = {x: 0 for x in model.loss_names}
avg_total = 0

# Perform training
print("Starting training")

for epoch in range(opt.num_epochs):

    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        start = timeit.default_timer()

        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()
        loss = 0

        for i in range(subdivisions):
            sub_imgs = imgs[i * sub_batch_size : (i + 1) * sub_batch_size]
            sub_targets = targets[i * sub_batch_size : (i + 1) * sub_batch_size]
            loss += model(sub_imgs, sub_targets)

        loss /= subdivisions

        loss.backward()
        optimizer.step()

        avg_total += loss.item()
        for x in model.loss_names:
            avg_losses[x] += model.losses[x]

        if batch_i % opt.avg_interval == 0:
            print(
                (
                    "[Epoch %2d/%2d, Batch %5d/%5d] "
                    + "[Avg Losses: total %f, x %f, y %f, w %f, h %f, conf %f, cls %f, recall: %.5f]"
                )
                % (
                    epoch,
                    opt.num_epochs,
                    batch_i,
                    len(dataloader),
                    avg_total / opt.avg_interval,
                    *[avg_losses[x] / opt.avg_interval for x in model.loss_names],
                )
            )

            avg_losses = {x: 0 for x in model.loss_names}
            avg_total = 0

        # print(
        # 	("+ [Epoch %2d/%2d, Batch %5d/%5d] " +
        # 	 "[Losses: total %f, x %f, y %f, w %f, h %f, conf %f, cls %f, recall: %.5f] " +
        # 	 "[Took %2.6f]"
        # 	 ) % (
        # 		epoch, opt.num_epochs, batch_i, len(dataloader),
        # 		loss.item(), *[model.losses[x] for x in model.loss_names],
        # 		timeit.default_timer() - start
        # 	)
        # )

        model.seen += imgs.size(0)

    # TODO: validate model

    if epoch % opt.checkpoint_interval == 0:
        print("Saving weights for [epoch %2d]" % epoch)
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
