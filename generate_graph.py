import torch
import numpy as np
from models import models, resnet_orig
from laplace import llla
from util.evaluation import *
import util.dataloaders as dl
from util.misc import *
from math import *
from tqdm import tqdm, trange
import argparse
import pickle
import os, sys
from tqdm import tqdm, trange
import torch.utils.data as data_utils
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib


sns.set_style('white')

parser = argparse.ArgumentParser()
parser.add_argument('--randseed', type=int, default=1)
parser.add_argument('--dataset', default='MNIST')
parser.add_argument('--no_rgpr', action='store_true', default=False)
args = parser.parse_args()

assert args.dataset in ['MNIST', 'CIFAR10', 'SVHN', 'CIFAR100']

torch.cuda.set_device(0)
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_classes = 100 if args.dataset == 'CIFAR100' else 10

train_loader = dl.datasets_dict[args.dataset](train=True, augm_flag=False)
val_loader, test_loader = dl.datasets_dict[args.dataset](train=False, val_size=2000)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

# For normalizing the input and activations when evaluating the DSCS kernel
means = np.load('means.npy', allow_pickle=True).item()[args.dataset]
means = [torch.tensor(m, device='cuda').float() for m in means]
stds = np.load('stds.npy', allow_pickle=True).item()[args.dataset]
stds = [torch.tensor(s, device='cuda').float() for s in stds]

# Handle zeros in stds
for i in range(len(stds)):
    stds[i][stds[i] == 0] = 1


def load_model():
    if args.dataset == 'MNIST':
        model = models.LeNetMadry().cuda()
    else:
        model = resnet_orig.ResNet18(num_classes=num_classes).cuda()

    model.load_state_dict(torch.load(f'./pretrained_models/{args.dataset}_plain.pt'))
    model.eval()

    params = list(np.load(f'./pretrained_models/{args.dataset}_plain_llla.npy', allow_pickle=True))

    return model, params


def evaluate(alpha=1, rgpr=True):
    model, params = load_model()
    n_params = 4 if args.dataset == 'MNIST' else 5
    kernel_vars = {
        'MNIST': [1.0567e-04, 2.3090e-04, 2.6643e-05, 1.2558e-02],
        'CIFAR10': [1.0081e+01, 1.0264e-03, 6.6690e-04, 1.8025e-03, 1.5980e-03],
        'SVHN': [0.0034, 0.0109, 0.0016, 0.0014, 0.0078],
        'CIFAR100': [2.6449e+01, 4.3422e-03, 1.0548e-03, 3.6053e-03, 7.6463e-03]
    }  # From eval_OOD.py
    kernel_vars = kernel_vars[args.dataset]

    py = llla.predict(test_loader, model, *params, delta=alpha, n_samples=1000, rgpr=rgpr, kernel_vars=kernel_vars, means=means, stds=stds, n=1000)
    conf = get_confidence(py.cpu().numpy())

    return conf.mean(), conf.std()


num_eval = 100
# alpha_max = 5000 if args.dataset == 'MNIST' else 100
alpha_max = 100
alphas = np.linspace(1, alpha_max+0.1, num_eval)
mean = np.zeros(num_eval)
std = np.zeros(num_eval)

for i, alpha in tqdm(enumerate(alphas)):
    m, s = evaluate(alpha, rgpr=not args.no_rgpr)
    mean[i] = m
    std[i] = s

plt.axhline(1/num_classes, lw=3, ls='-', c='k')
plt.fill_between(alphas, np.maximum(1/num_classes, mean-std), np.minimum(1, mean+std), color='r', alpha=0.15)
plt.plot(alphas, mean, lw=3, c='r')

plt.xlim(1, alpha_max)
plt.ylim(0, 1.05)
plt.xlabel(r'$\alpha$')
plt.ylabel('Confidence')

tikzplotlib.save(f'notebooks/figs/conf_lll{"_rgpr" if not args.no_rgpr else ""}_{args.dataset.lower()}.tex')
plt.savefig(f'notebooks/figs/conf_lll{"_rgpr" if not args.no_rgpr else ""}_{args.dataset.lower()}.pdf')
