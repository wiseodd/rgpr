import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import torch
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from models import models, resnet_orig, dkl, gp
from laplace import llla, kfla
import laplace.util as lutil
from swag.posteriors import SWAG
from util.evaluation import *
from util.tables import *
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
from gpytorch import likelihoods
from sklearn.preprocessing import StandardScaler

from pycalib.calibration_methods import TemperatureScaling


parser = argparse.ArgumentParser()
parser.add_argument('--randseed', type=int, default=9999)
parser.add_argument('--dataset', default='MNIST')
parser.add_argument('--faraway', action='store_true', default=False)
parser.add_argument('--compute_hessian', action='store_true', default=False)
parser.add_argument('--optimize_hyper', action='store_true', default=False)
parser.add_argument('--lam', type=float, default=1)
parser.add_argument('--ood_dset', default='smooth', choices=['smooth', 'imagenet'])
parser.add_argument('--dont_save', action='store_true', default=False)
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

ood_loaders = {'uniform': dl.UniformNoise(args.dataset), 'smooth': dl.Noise(args.dataset),
               'imagenet': dl.ImageNet32(dataset=args.dataset)}
ood_loader = ood_loaders[args.ood_dset]

data_shape = [1, 28, 28] if args.dataset == 'MNIST' else [3, 32, 32]

ood_noise_names = ['UniformNoise', 'Noise']

ood_test_names = {
    'MNIST': ['EMNIST', 'KMNIST', 'FMNIST', 'GrayCIFAR10'],
    'CIFAR10': ['SVHN', 'LSUN', 'CIFAR100', 'FMNIST3D'],
    'SVHN': ['CIFAR10', 'LSUN', 'CIFAR100', 'FMNIST3D'],
    'CIFAR100': ['SVHN', 'LSUN', 'CIFAR10', 'FMNIST3D'],
}

if args.faraway:
    ood_names = ['FarAway']
    ood_test_loaders = {'FarAway': dl.UniformNoise(args.dataset, delta=2000)}
else:
    ood_names = ood_test_names[args.dataset] + ood_noise_names
    ood_test_loaders = {}

    for ood_name in ood_test_names[args.dataset]:
        ood_test_loaders[ood_name] = dl.datasets_dict[ood_name](train=False)

    for ood_name in ood_noise_names:
        ood_test_loaders[ood_name] = dl.datasets_dict[ood_name](dataset=args.dataset, train=False, size=2000)

method_types = ['MAP', 'LLL', 'KFL', 'SWAG', 'SVDKL', 'BNO', 'DE', 'Temp']
method_types_rgpr = [m + '-RGPR-loglik' for m in method_types if m not in ('MAP', 'BNO', 'DE', 'Temp')]
method_types_rgpr += [m + '-RGPR-ood' for m in method_types if m not in ('MAP', 'BNO', 'DE', 'Temp')]

# For normalizing the input and activations when evaluating the DSCS kernel
means = np.load('means.npy', allow_pickle=True).item()[args.dataset]
means = [torch.tensor(m, device='cuda').float() for m in means]
stds = np.load('stds.npy', allow_pickle=True).item()[args.dataset]
stds = [torch.tensor(s, device='cuda').float() for s in stds]
# stds = [torch.tensor(s, device='cuda').float() * 100 for s in stds]
# stds = [torch.ones_like(torch.tensor(s), device='cuda').float() * 5 for s in stds]


# Handle zeros in stds
for i in range(len(stds)):
    stds[i][stds[i] == 0] = 1

# Optimal hyperparams
hyperparams = {
    'smooth': {
        'ood': {
            'MNIST': [1.7384e-05, 1.6409e-06, 1.3555e-07, 2.5206e-03],
            'SVHN': [8.2850e+00, 6.2021e-03, 9.1418e-03, 4.7633e-03, 1.3424e-02],
            'CIFAR10': [4.6957e+01, 8.4602e-04, 1.3050e-03, 5.9322e-03, 1.9222e-03],
            'CIFAR100': [2.6372e+01, 2.8527e-03, 8.7588e-04, 4.5595e-03, 2.5490e-01]
        },
        'loglik': {
            'MNIST': [3.3939e-08, 5.4485e-07, 1.1377e-07, 2.3509e-03],
            'SVHN': [9.3995e-04, 1.3767e-04, 1.1347e-04, 2.2835e-04, 3.9480e-05],
            'CIFAR10': [0.0036, 0.0005, 0.0008, 0.0018, 0.0028],
            'CIFAR100': [0.0094, 0.0093, 0.0019, 0.0049, 0.0144]
        }
    },
    'imagenet': {
        'ood': {
            'MNIST': [3.5457e-08, 5.9255e-07, 1.1685e-07, 2.4544e-03],
            'SVHN': [1.1849e-03, 1.3038e-01, 3.5909e-04, 3.8309e-04, 8.2367e-05],
            'CIFAR10': [0.0236, 0.9079, 0.0030, 0.0049, 0.0053],
            'CIFAR100': [0.0152, 0.9533, 0.0051, 0.0094, 0.2049]
        },
        'loglik': {  # Same as above
            'MNIST': [3.3939e-08, 5.4485e-07, 1.1377e-07, 2.3509e-03],
            'SVHN': [9.3995e-04, 1.3767e-04, 1.1347e-04, 2.2835e-04, 3.9480e-05],
            'CIFAR10': [0.0036, 0.0005, 0.0008, 0.0018, 0.0028],
            'CIFAR100': [0.0094, 0.0093, 0.0019, 0.0049, 0.0144]
        }
    }
}

tab_mmc = {mt: {} for mt in method_types + method_types_rgpr}
tab_aur = {mt: {} for mt in method_types + method_types_rgpr}
tab_aupr = {mt: {} for mt in method_types + method_types_rgpr}
tab_fpr95 = {mt: {} for mt in method_types + method_types_rgpr}
tab_acc = {}
tab_cal = {}


def load_model(type):
    assert type in method_types, 'Invalid model type'

    if args.dataset == 'MNIST':
        if type == 'DE':
            model = [models.LeNetMadry().cuda() for _ in range(5)]
        else:
            model = models.LeNetMadry().cuda()
    else:
        if type == 'DE':
            model = [resnet_orig.ResNet18(num_classes=num_classes).cuda() for _ in range(5)]
        else:
            model = resnet_orig.ResNet18(num_classes=num_classes).cuda()

    if type == 'SVDKL':
        model_dkl, likelihood = dkl.get_dkl_model(dataset=args.dataset)
        model_dkl.cuda()
        likelihood.cuda()

        state = torch.load(f'./pretrained_models/{args.dataset}_dkl.pt')
        model_dkl.load_state_dict(state['model'])
        likelihood.load_state_dict(state['likelihood'])
        model_dkl.eval()
        likelihood.eval()

        return model_dkl, likelihood

    if type == 'BNO':
        model_bno = gp.GPResidual(model, torch.zeros(64, np.prod(data_shape)), num_classes, data_shape, 'DCSC')
        likelihood = likelihoods.SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
        model_bno.cuda()
        likelihood.cuda()

        state = torch.load(f'./pretrained_models/{args.dataset}_blight_DSCS.pt')
        model_bno.load_state_dict(state['model'])
        likelihood.load_state_dict(state['likelihood'])
        model_bno.eval()
        likelihood.eval()

        return model_bno, likelihood

    if type == 'DE':
        state_dicts = torch.load(f'./pretrained_models/{args.dataset}_de.pt')
        for k in range(5):
            model[k].load_state_dict(state_dicts[k])
            model[k].eval()
    else:
        model.load_state_dict(torch.load(f'./pretrained_models/{args.dataset}_plain.pt'))
        model.eval()

    return model


def predict_(test_loader, model, model_name, params=None, rgpr=False, kernel_var=1, base_model=None, T=1):
    assert model_name in method_types

    if model_name == 'LLL':
        py = llla.predict(test_loader, model, *params, rgpr=rgpr, kernel_vars=kernel_var, means=means, stds=stds)
    elif model_name == 'SVDKL':
        model_dkl, likelihood = model
        py = predict_dkl(test_loader, model_dkl, likelihood, rgpr=rgpr, kernel_vars=kernel_var, means=means, stds=stds)
    elif model_name == 'BNO':
        model_bno, likelihood = model
        py = predict_blight(test_loader, model_bno, likelihood)
    elif model_name == 'SWAG':
        # SWAG is severly underconfident on MNIST and CIFAR100 with 0.5 scale
        if args.dataset == 'MNIST':
            scale = 0.15
        elif args.dataset == 'CIFAR100':
            scale = 0.25
        else:
            scale = 0.5

        py = lutil.predict2(
            test_loader, model, n_classes=num_classes, scale=scale,
            train_loader=train_loader, rgpr=rgpr,
            kernel_vars=kernel_var, base_model=base_model, means=means, stds=stds
        )
    elif model_name == 'KFL':
        py = lutil.predict(
            test_loader, model, n_classes=num_classes, rgpr=rgpr,
            kernel_vars=kernel_var, base_model=base_model, means=means, stds=stds
        )
    elif model_name == 'DE':
        py = predict_ensemble(test_loader, model)
    else:  # MAP
        py = predict(test_loader, model, T=T)

    return py.cpu().numpy()


def get_best_kernel_var(val_loader, model, model_name, loss_type, params=None, base_model=None, lam=1):
    targets_in = []
    for _, y in val_loader:
        targets_in.append(y.cuda())
    targets_in = torch.cat(targets_in)

    n_params = 4 if args.dataset == 'MNIST' else 5

    # # Targetting specific layers
    # inits = torch.randn(n_params, device='cuda')
    # inits.data[:-1] = -15
    # log_kernel_vars = torch.nn.Parameter(inits)
    # log_kernel_vars.register_hook(lambda grad: grad * torch.tensor([0, 0, 0, 0, 1]).float().cuda())
    # kernel_vars = torch.exp(log_kernel_vars).detach()

    log_kernel_vars = torch.nn.Parameter(torch.randn(n_params, device='cuda'))
    print(kernel_vars)

    n_epochs = 50
    opt = torch.optim.Adam([log_kernel_vars], lr=1e-1)
    # scd = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs*len(val_loader))
    scd = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    pbar = trange(n_epochs)

    for it in pbar:
        loss_epoch = 0
        n_batches = 0

        if args.ood_dset == 'imagenet':
            ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))

        for (x_in, y_in), (x_out, y_out) in zip(val_loader, ood_loader):
            x_in, y_in = x_in.cuda(), y_in.cuda()
            x_out, y_out = x_out.cuda(), y_out.cuda()

            kernel_vars = torch.exp(log_kernel_vars)

            py_in = llla.predict_mb(x_in, y_in, model, *params, kernel_vars=kernel_vars,
                                    rgpr=True, means=means, stds=stds)

            # OE-like
            loss_in = F.nll_loss(torch.log(py_in + 1e-8), y_in)

            if loss_type == 'ood':
                py_out = llla.predict_mb(x_out, y_out, model, *params, kernel_vars=kernel_vars,
                                         rgpr=True, means=means, stds=stds)
                loss_out = torch.log(py_out + 1e-8).mean()
            else:
                loss_out = 0

            loss = loss_in - lam*loss_out

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_epoch += loss.item()
            n_batches += 1

        scd.step(loss_epoch/n_batches)
        pbar.set_description(f'[Loss: {loss_epoch/n_batches:.3f}; {kernel_vars.cpu().detach().numpy()}]')

    return torch.exp(log_kernel_vars).detach()


def evaluate(model_name, rgpr=False, loss_type='ood'):
    assert model_name in method_types
    assert loss_type in ['ood', 'loglik']

    model_ = load_model(model_name)
    params = None
    kernel_var = 1
    rgpr_str = '-RGPR' if rgpr else ''

    if rgpr and not args.faraway:
        rgpr_str += f'-{loss_type}'

    model_str = model_name + rgpr_str
    base_model = load_model(model_name)

    if model_name == 'LLL':
        model = model_
        model.eval()

        if args.compute_hessian:
            # Compute LLL Hessians
            hessians = llla.get_hessian(model, train_loader)

            # Majority from https://github.com/wiseodd/last_layer_laplace
            var0 = {'MNIST': 657.9332, 'CIFAR10': 41.3636, 'CIFAR100': 1, 'SVHN': 566.0909}

            # var0 = torch.tensor(var0[args.dataset]).float().cuda()
            M_W, M_b, U, V, B = llla.estimate_variance(var0, hessians)
            np.save(f'./pretrained_models/{args.dataset}_plain_llla.npy', [M_W, M_b, U, V, B])

            print(f'LLLA saved: pretrained_models/{args.dataset}_plain_llla.npy')
            sys.exit()

        # Load LLL Hessians
        params = list(np.load(f'./pretrained_models/{args.dataset}_plain_llla.npy', allow_pickle=True))
    elif model_name == 'KFL':
        model = kfla.KFLA(model_)
        model.load_state_dict(torch.load(f'./pretrained_models/{args.dataset}_plain_kfla.pt'))
        model.eval()
    elif model_name == 'SWAG':
        model = SWAG(
            model_,
            no_cov_mat=False,
            max_num_models=20,
            num_classes=num_classes
        ).cuda()

        chkpt = torch.load(f'./pretrained_models/swag/{args.dataset}_swag.pt')
        model.load_state_dict(chkpt['state_dict'])
        model.eval()
    else:
        model = model_

    if rgpr:
        if args.optimize_hyper:
            kernel_var = get_best_kernel_var(val_loader, model, model_name, loss_type, params=params, base_model=base_model, lam=args.lam)
            print(args.dataset, loss_type, args.ood_dset, kernel_var)
            sys.exit(0)
        else:
            kernel_var = hyperparams[args.ood_dset][loss_type][args.dataset]

            if args.faraway:
                kernel_var = [1e-10] * len(kernel_var)

    if model_name == 'Temp':
        X = predict_logit(val_loader, model).cpu().numpy()
        y = torch.cat([y for _, y in val_loader], dim=0).numpy()
        T = TemperatureScaling().fit(X, y).T
    else:
        T = 1

    py_in = predict_(test_loader, model, model_name, params=params, rgpr=rgpr, kernel_var=kernel_var, base_model=base_model, T=T)
    acc_in = np.mean(np.argmax(py_in, 1) == targets)
    conf_in = get_confidence(py_in)
    mmc = conf_in.mean()
    ece, mce = get_calib(py_in, targets)
    tab_mmc[model_str][args.dataset] = mmc
    tab_aur[model_str][args.dataset] = None
    tab_aupr[model_str][args.dataset] = None
    tab_fpr95[model_str][args.dataset] = None
    tab_acc[model_str] = acc_in
    tab_cal[model_str] = ece
    nll = -dist.Categorical(torch.tensor(py_in).float()).log_prob(torch.tensor(targets).long()).mean()
    print(f'[In, {model_str}] Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; NLL: {nll:.3f}; MMC: {mmc:.3f}')

    for ood_name, ood_test_loader in ood_test_loaders.items():
        py_out = predict_(ood_test_loader, model, model_name, params=params, rgpr=rgpr, kernel_var=kernel_var, base_model=base_model, T=T)

        conf = get_confidence(py_out)
        mmc = conf.mean()
        aur = get_auroc(py_in, py_out)
        aupr = get_aupr(py_in, py_out)
        fpr95, _ = get_fpr95(py_in, py_out)

        tab_mmc[model_str][ood_name] = mmc
        tab_aur[model_str][ood_name] = aur
        tab_aupr[model_str][ood_name] = aupr
        tab_fpr95[model_str][ood_name] = fpr95

        print(f'[Out-{ood_name}, {model_str}] MMC: {mmc:.3f}; AUR: {aur:.3f}; AUPR: {aupr:.3f}; FPR@95: {fpr95:.3f}')


if not args.faraway:
    if args.ood_dset != 'imagenet':
        evaluate('MAP')
        print()
        evaluate('Temp')
        print()
        evaluate('BNO')
        print()
        evaluate('DE')
        print()
        evaluate('LLL')
        print()
        evaluate('LLL', rgpr=True, loss_type='loglik')
        print()

    evaluate('LLL', rgpr=True, loss_type='ood')
    print()
else:
    evaluate('KFL')
    print()
    evaluate('SWAG')
    print()
    evaluate('SVDKL')
    print()
    evaluate('KFL', rgpr=True)
    print()
    evaluate('SWAG', rgpr=True)
    print()
    evaluate('SVDKL', rgpr=True)
    print()


if not args.dont_save:
    # Save dict
    dir_name = f'results/'

    if args.faraway:
        dir_name += 'faraway/'
    else:
        dir_name += 'hyperopt/'

        if args.ood_dset == 'imagenet':
            dir_name += 'imagenet/'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(f'{dir_name}/tab_mmc_{args.dataset.lower()}_{args.randseed}.pkl', 'wb') as f:
        pickle.dump(tab_mmc, f)

    with open(f'{dir_name}/tab_aur_{args.dataset.lower()}_{args.randseed}.pkl', 'wb') as f:
        pickle.dump(tab_aur, f)

    with open(f'{dir_name}/tab_aupr_{args.dataset.lower()}_{args.randseed}.pkl', 'wb') as f:
        pickle.dump(tab_aupr, f)

    with open(f'{dir_name}/tab_fpr95_{args.dataset.lower()}_{args.randseed}.pkl', 'wb') as f:
        pickle.dump(tab_fpr95, f)

    if not args.faraway:
        with open(f'{dir_name}/tab_acc_{args.dataset.lower()}_{args.randseed}.pkl', 'wb') as f:
            pickle.dump(tab_acc, f)

        with open(f'{dir_name}/tab_cal_{args.dataset.lower()}_{args.randseed}.pkl', 'wb') as f:
            pickle.dump(tab_cal, f)
