import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import dataloaders as dl

import models.resnet_orig as resnet
from models.models import LeNetMadry
from swag import data, utils, losses
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dir",
    type=str,
    default='pretrained_models',
    help="training directory (default: None)",
)

parser.add_argument(
    "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
)
parser.add_argument(
    "--use_test",
    dest="use_test",
    default=True,
    action="store_false",
    help="use test dataset instead of validation (default: True)",
)
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)

parser.add_argument(
    "--resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to resume training from (default: None)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=150,
    metavar="N",
    help="number of epochs to train (default: 150)",
)
parser.add_argument(
    "--save_freq",
    type=int,
    default=150,
    metavar="N",
    help="save frequency (default: 150)",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=5,
    metavar="N",
    help="evaluation frequency (default: 5)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.1,
    metavar="LR",
    help="initial learning rate (default: 0.1)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)"
)

parser.add_argument("--swa", action="store_true", help="swa usage flag (default: off)")
parser.add_argument(
    "--swa_start",
    type=float,
    default=101,
    metavar="N",
    help="SWA start epoch number (default: 101)",
)
parser.add_argument(
    "--swa_lr", type=float, default=0.01, metavar="LR", help="SWA LR (default: 0.01)"
)
parser.add_argument(
    "--swa_c_epochs",
    type=int,
    default=1,
    metavar="N",
    help="SWA model collection frequency/cycle length in epochs (default: 1)",
)
parser.add_argument(
    "--max_num_models",
    type=int,
    default=20,
    help="maximum number of SWAG models to save",
)

parser.add_argument(
    "--swa_resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to restor SWA from (default: None)",
)
parser.add_argument(
    "--loss",
    type=str,
    default="CE",
    help="loss to use for training model (default: Cross-entropy)",
)

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument("--no_schedule", action="store_true", help="store schedule")

args = parser.parse_args()

args.device = None
args.cov_mat = True  # !!!!!!!!!!!!!

use_cuda = torch.cuda.is_available()

if use_cuda:
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

print("Preparing directory %s" % args.dir)
args.dir = args.dir + f'/{args.dataset}'
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, "command.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed(args.seed)

model_cfg = None

print("Loading dataset %s" % (args.dataset))
loaders = {
    "train": dl.datasets_dict[args.dataset](train=True, batch_size=args.batch_size),
    "test": dl.datasets_dict[args.dataset](train=False, batch_size=args.batch_size),
}
num_classes = 100 if args.dataset == 'CIFAR100' else 10

print("Preparing model")

def get_model():
    return LeNetMadry() if args.dataset == 'MNIST' else resnet.ResNet18(num_classes=num_classes)

model = get_model()
model.to(args.device)


if args.swa:
    print("SWAG training")
    swag_model = SWAG(
        get_model(),
        no_cov_mat=False,
        max_num_models=args.max_num_models,
        num_classes=num_classes,
    )
    swag_model.to(args.device)
else:
    print("SGD training")


def schedule(epoch):
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if epoch < 50:
        factor = 1.0
    elif epoch < 75:
        factor = 0.1
    elif epoch < 90:
        factor = 0.01
    elif epoch < args.swa_start:
        factor = 0.001
    else:
        factor = lr_ratio
    return args.lr_init * factor


# use a slightly modified loss function that allows input of model
if args.loss == "CE":
    criterion = losses.cross_entropy
    # criterion = F.cross_entropy
elif args.loss == "adv_CE":
    criterion = losses.adversarial_cross_entropy

if args.dataset == 'MNIST':
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
else:
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd
    )

start_epoch = 0
if args.resume is not None:
    print("Resume training from %s" % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

if args.swa and args.swa_resume is not None:
    checkpoint = torch.load(args.swa_resume)
    swag_model = SWAG(
        get_model(),
        no_cov_mat=False,
        max_num_models=args.max_num_models,
        loading=True,
        num_classes=num_classes
    )
    swag_model.to(args.device)
    swag_model.load_state_dict(checkpoint["state_dict"])

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time", "mem_usage"]
if args.swa:
    columns = columns[:-2] + ["swa_te_loss", "swa_te_acc"] + columns[-2:]
    swag_res = {"loss": None, "accuracy": None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict(),
)

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.0

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    if not args.no_schedule:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    if (args.swa and (epoch + 1) > args.swa_start) and args.cov_mat:
        train_res = utils.train_epoch(loaders["train"], model, criterion, optimizer, cuda=use_cuda)
    else:
        train_res = utils.train_epoch(loaders["train"], model, criterion, optimizer, cuda=use_cuda)

    if (
        epoch == 0
        or epoch % args.eval_freq == args.eval_freq - 1
        or epoch == args.epochs - 1
    ):
        test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda)
    else:
        test_res = {"loss": None, "accuracy": None}

    if (
        args.swa
        and (epoch + 1) > args.swa_start
        and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0
    ):
        # sgd_preds, sgd_targets = utils.predictions(loaders["test"], model)
        sgd_res = utils.predict(loaders["test"], model)
        sgd_preds = sgd_res["predictions"]
        sgd_targets = sgd_res["targets"]
        print("updating sgd_ens")
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            # TODO: rewrite in a numerically stable way
            sgd_ens_preds = sgd_ens_preds * n_ensembled / (
                n_ensembled + 1
            ) + sgd_preds / (n_ensembled + 1)
        n_ensembled += 1
        swag_model.collect_model(model)
        if (
            epoch == 0
            or epoch % args.eval_freq == args.eval_freq - 1
            or epoch == args.epochs - 1
        ):
            swag_model.sample(0.0)
            utils.bn_update(loaders["train"], swag_model)
            swag_res = utils.eval(loaders["test"], swag_model, criterion)
        else:
            swag_res = {"loss": None, "accuracy": None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        if args.swa:
            utils.save_checkpoint(
                args.dir, epoch + 1, name="swag", state_dict=swag_model.state_dict()
            )

    time_ep = time.time() - time_ep

    if use_cuda:
        memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

    values = [
        epoch + 1,
        lr,
        train_res["loss"],
        train_res["accuracy"],
        test_res["loss"],
        test_res["accuracy"],
        time_ep,
        memory_usage,
    ]
    if args.swa:
        values = values[:-2] + [swag_res["loss"], swag_res["accuracy"]] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    if args.swa and args.epochs > args.swa_start:
        utils.save_checkpoint(
            args.dir, args.epochs, name="swag", state_dict=swag_model.state_dict()
        )

if args.swa:
    np.savez(
        os.path.join(args.dir, "sgd_ens_preds.npz"),
        predictions=sgd_ens_preds,
        targets=sgd_targets,
    )
