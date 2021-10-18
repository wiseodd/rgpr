import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from math import *
from models.models import LeNetMadry
from models import resnet_orig
from tqdm.auto import tqdm, trange
from util import dataloaders as dl
from util import evaluation as ev
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Pick one \\{"MNIST", "CIFAR10", "SVHN", "CIFAR100"\\}', default='MNIST')
parser.add_argument('--train', default=False, action='store_true')
args = parser.parse_args()


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if args.dataset == 'MNIST':
    train_loader = dl.MNIST(train=True, augm_flag=True)
    val_loader, test_loader = dl.MNIST(train=False, augm_flag=False, val_size=2000)
elif args.dataset == 'CIFAR10':
    train_loader = dl.CIFAR10(train=True, augm_flag=True)
    val_loader, test_loader = dl.CIFAR10(train=False, augm_flag=False,  val_size=2000)
elif args.dataset == 'SVHN':
    train_loader = dl.SVHN(train=True, augm_flag=True)
    val_loader, test_loader = dl.SVHN(train=False, augm_flag=False,  val_size=2000)
elif args.dataset == 'CIFAR100':
    train_loader = dl.CIFAR100(train=True, augm_flag=True)
    val_loader, test_loader = dl.CIFAR100(train=False, augm_flag=False,  val_size=2000)

targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
targets_val = torch.cat([y for x, y in val_loader], dim=0).numpy()


@torch.no_grad()
def predict(dataloader, models):
    py = []

    for x, y in dataloader:
        x = x.cuda()

        py_ = 0
        for model in models:
            py_ += 1/len(models)*torch.softmax(model(x), 1)

        py.append(py_)

    return torch.cat(py, dim=0)


K = 5  # DE's num of components
n_classes = 100 if args.dataset == 'CIFAR100' else 10
lam = 5e-4

if args.dataset == 'MNIST':
    models = [LeNetMadry().cuda() for _ in range(K)]
    opts = [torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=lam) for m in models]
else:
    models = [resnet_orig.ResNet18(num_classes=n_classes).cuda() for _ in range(K)]
    opts = [torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=lam) for m in models]


if args.train:
    pbar = trange(100)

    for epoch in pbar:
        if epoch+1 in [50, 75, 90]:
            for opt in opts:
                for group in opt.param_groups:
                    group['lr'] *= .1

        train_loss= 0
        n = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.long().cuda()

            for k in range(K):
                output = models[k](data).squeeze()
                loss = F.cross_entropy(output, target)
                opts[k].zero_grad()
                loss.backward()
                opts[k].step()

            train_loss += loss.item()
            n += 1

        train_loss /= n

        pred_val = predict(val_loader, models).cpu().numpy()
        acc_val = np.mean(np.argmax(pred_val, 1) == targets_val)*100

        pbar.set_description(f'[Epoch: {epoch+1}; val: {acc_val:.1f}]')

    torch.save([m.state_dict() for m in models], f'./pretrained_models/{args.dataset}_de.pt')

# Test
state_dicts = torch.load(f'./pretrained_models/{args.dataset}_de.pt')
for k in range(K):
    models[k].load_state_dict(state_dicts[k])
    models[k].eval()

print()

# Test
ood_loader = dl.FMNIST(train=False)

py_in = predict(test_loader, models).cpu().numpy()
py_out = predict(ood_loader, models).cpu().numpy()

acc = np.mean(np.argmax(py_in, 1) == targets)*100
mmc = np.max(py_in).mean()
aur = ev.get_auroc(py_in, py_out)
aupr = ev.get_aupr(py_in, py_out)
fpr95, _ = ev.get_fpr95(py_in, py_out)

print(f'Accuracy: {acc:.3f}; MMC: {mmc:.3f}; AUR: {aur:.3f}; AUPR: {aupr:.3f}; FPR@95: {fpr95:.3f}')
