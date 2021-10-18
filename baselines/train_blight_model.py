import torch
from models.models import LeNetMadry
from models import resnet_orig as resnet
from models import densenet
from models import gp, dkl
from laplace import llla, kfla, dla
import laplace.util as lutil
from util.evaluation import *
from util.tables import *
import util.dataloaders as dl
from util.misc import *
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import os, sys
import math
import gpytorch
from gpytorch import kernels, likelihoods


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Pick one \\{"MNIST", "CIFAR10", "SVHN", "CIFAR100"\\}', default='MNIST')
parser.add_argument('--kernel', help='Pick one \\{"RBF", "DSCS"\\}', default='DSCS')
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

img_mean, img_std = None, None

if args.dataset == 'MNIST':
    train_loader = dl.MNIST(train=True)
    val_loader, test_loader = dl.MNIST(train=False, augm_flag=False, val_size=2000)
elif args.dataset == 'CIFAR10':
    train_loader = dl.CIFAR10(train=True, mean=img_mean, std=img_std)
    val_loader, test_loader = dl.CIFAR10(train=False, augm_flag=False, val_size=2000, mean=img_mean, std=img_std)
elif args.dataset == 'SVHN':
    train_loader = dl.SVHN(train=True, mean=img_mean, std=img_std)
    val_loader, test_loader = dl.SVHN(train=False, augm_flag=False, val_size=2000, mean=img_mean, std=img_std)
elif args.dataset == 'CIFAR100':
    train_loader = dl.CIFAR100(train=True, mean=img_mean, std=img_std)
    val_loader, test_loader = dl.CIFAR100(train=False, augm_flag=False, val_size=2000, mean=img_mean, std=img_std)

targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
targets_val = torch.cat([y for x, y in val_loader], dim=0).numpy()

data_shape = [1, 28, 28] if args.dataset == 'MNIST' else [3, 32, 32]
num_classes = 100 if args.dataset == 'CIFAR100' else 10

print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

means = np.load('means.npy', allow_pickle=True).item()[args.dataset]
means = [torch.tensor(m, device='cuda').float() for m in means]
stds = np.load('stds.npy', allow_pickle=True).item()[args.dataset]
stds = [torch.tensor(s, device='cuda').float() for s in stds]

print(len(means), len(stds))

# Obtain inducing points
n_inducing = 64
inducing_points = []
stop = n_inducing//dl.train_batch_size

for x, y in train_loader:
    inducing_points.append(x)

    if len(inducing_points) >= stop:
        break

inducing_points = torch.cat(inducing_points, dim=0)
inducing_points = inducing_points[:n_inducing, :].cuda()

# Obtain the pre-trained base model
base_model = LeNetMadry() if args.dataset == 'MNIST' else resnet.ResNet18(num_classes=num_classes)
base_model.cuda()
base_model.load_state_dict(torch.load(f'./pretrained_models/{args.dataset}_plain.pt'))
base_model.eval()

# Don't train the base model
for p in base_model.parameters():
    p.requires_grad = False

model = gp.GPResidual(base_model, inducing_points, num_classes, data_shape, args.kernel).cuda()
likelihood = likelihoods.SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False).cuda()

gp_hypers = list(filter(lambda p: p.requires_grad, model.parameters()))

opt = torch.optim.SGD([
    {'params': gp_hypers},
    {'params': likelihood.parameters()},
], lr=0.1, weight_decay=0)

mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

pbar = trange(20)

for epoch in pbar:
    train_loss = 0
    n = 0

    model.train()
    likelihood.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.long().cuda()

        opt.zero_grad()
        output = model(data.flatten(1))
        # output = model(data)
        loss = -mll(output, target)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        n += 1

    train_loss /= n

    # Validation accuracy
    # -------------------
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(20):
        pred_val = []

        for data, target in val_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = likelihood(model(data.flatten(1)))  # This gives us 20 samples from the predictive distribution
            # output = likelihood(model(data))
            pred_val_ = output.probs.mean(0).cpu().numpy()  # Taking the mean over all of the sample we've drawn
            pred_val.append(pred_val_)

        pred_val = np.concatenate(pred_val, 0)
        acc_val = np.mean(np.argmax(pred_val, 1) == targets_val)*100

    pbar.set_description(f'[Epoch: {epoch+1}; train_loss: {train_loss:.4f}; val_acc: {acc_val:.1f}]')

torch.save({'model': model.state_dict(), 'likelihood': likelihood.state_dict()}, f'./pretrained_models/{args.dataset}_blight_{args.kernel}.pt')

state = torch.load(f'./pretrained_models/{args.dataset}_blight_{args.kernel}.pt')
model.load_state_dict(state['model'])
likelihood.load_state_dict(state['likelihood'])
model.eval()
likelihood.eval()


print()

# Test (in-distribution)
# ----------------------
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.num_likelihood_samples(20):
    py_in = []

    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output = likelihood(model(data.flatten(1)))  # This gives us 20 samples from the predictive distribution
        py_in_ = output.probs.mean(0).cpu().numpy()  # Taking the mean over all of the sample we've drawn
        py_in.append(py_in_)

    py_in = np.concatenate(py_in, 0)

acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
mmc = np.maximum(py_in, 1-py_in).mean()*100
print(f'[In, MAP] Accuracy: {acc_in:.3f}; MMC: {mmc:.3f}')
