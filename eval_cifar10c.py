import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import torch
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

from pycalib.calibration_methods import TemperatureScaling


parser = argparse.ArgumentParser()
parser.add_argument('--randseed', type=int, default=9999)
args = parser.parse_args()

torch.cuda.set_device(0)
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

val_loader, in_loader = dl.CIFAR10(train=False, augm_flag=False, val_size=2000)
num_classes = 10

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


method_types = ['MAP', 'Temp', 'LLL', 'BNO', 'DE']

# For normalizing the input and activations when evaluating the DSCS kernel
means = np.load('means.npy', allow_pickle=True).item()['CIFAR10']
means = [torch.tensor(m, device='cuda').float() for m in means]
stds = np.load('stds.npy', allow_pickle=True).item()['CIFAR10']
stds = [torch.tensor(s, device='cuda').float() for s in stds]

# Handle zeros in stds
for i in range(len(stds)):
    stds[i][stds[i] == 0] = 1

# Optimal hyperparams
hyperparams = {
    'LLL-RGPR-loglik': [0.0036, 0.0005, 0.0008, 0.0018, 0.0028],
    'LLL-RGPR-ood': [4.6957e+01, 8.4602e-04, 1.3050e-03, 5.9322e-03, 1.9222e-03],
    'LLL': None,
    'BNO': None,
    'DE': None
}

tab_acc = defaultdict(list)
tab_brier = defaultdict(list)
tab_mmc = defaultdict(list)
tab_ece = defaultdict(list)
tab_nll = defaultdict(list)


def load_model(type):
    assert type in method_types, 'Invalid model type'

    if type == 'DE':
        model = [resnet_orig.ResNet18(num_classes=num_classes).cuda() for _ in range(5)]
    else:
        model = resnet_orig.ResNet18(num_classes=num_classes).cuda()

    if type == 'BNO':
        data_shape = [3, 32, 32]
        model_bno = gp.GPResidual(model, torch.zeros(64, np.prod(data_shape)), num_classes, data_shape, 'DCSC')
        likelihood = likelihoods.SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
        model_bno.cuda()
        likelihood.cuda()

        state = torch.load(f'./pretrained_models/CIFAR10_blight_DSCS.pt')
        model_bno.load_state_dict(state['model'])
        likelihood.load_state_dict(state['likelihood'])
        model_bno.eval()
        likelihood.eval()

        return model_bno, likelihood
    elif type == 'DE':
        state_dicts = torch.load(f'./pretrained_models/CIFAR10_de.pt')
        for k in range(5):
            model[k].load_state_dict(state_dicts[k])
            model[k].eval()
    else:
        model.load_state_dict(torch.load(f'./pretrained_models/CIFAR10_plain.pt'))
        model.eval()

    return model


def predict_(test_loader, model, model_name, params=None, rgpr=False, kernel_var=1, base_model=None, T=1):
    assert model_name in method_types

    if model_name == 'LLL':
        py = llla.predict(test_loader, model, *params, rgpr=rgpr, kernel_vars=kernel_var, means=means, stds=stds)
    elif model_name == 'DE':
        py = predict_ensemble(test_loader, model)
    elif model_name == 'BNO':
        model_bno, likelihood = model
        py = predict_blight(test_loader, model_bno, likelihood)
    else:  # MAP
        py = predict(test_loader, model, T=T)

    return py.cpu().numpy()


def evaluate(model_name, out_loader, out_targets, rgpr=False, loss_type='ood'):
    assert loss_type in ['loglik', 'ood']
    assert model_name in method_types

    targets_onehot = get_one_hot(out_targets, num_classes)

    model_ = load_model(model_name)
    params = None
    rgpr_str = f'-RGPR-{loss_type}' if rgpr else ''
    model_str = model_name + rgpr_str
    base_model = load_model(model_name)
    kernel_var = hyperparams[model_str] if rgpr else None

    if model_name == 'LLL':
        model = model_
        model.eval()
        params = list(np.load(f'./pretrained_models/CIFAR10_plain_llla.npy', allow_pickle=True))
    else:
        model = model_

    if model_name == 'Temp':
        X = predict_logit(val_loader, model).cpu().numpy()
        y = torch.cat([y for _, y in val_loader], dim=0).numpy()
        T = TemperatureScaling().fit(X, y).T
    else:
        T = 1

    py = predict_(out_loader, model, model_name, params=params, rgpr=rgpr, kernel_var=kernel_var, base_model=base_model, T=T)

    acc = np.mean(np.argmax(py, 1) == out_targets)
    conf = get_confidence(py)
    mmc = conf.mean()
    brier = np.mean(np.linalg.norm(py - targets_onehot, ord=2, axis=1)**2)
    ece, _ = get_calib(py, out_targets)
    nll = get_nll(py, out_targets)

    tab_mmc[model_str].append(mmc)
    tab_acc[model_str].append(acc)
    tab_brier[model_str].append(brier)
    tab_ece[model_str].append(ece)
    tab_nll[model_str].append(nll)


for distortion in tqdm(dl.CorruptedCIFAR10Dataset.distortions):
    for severity in trange(1, 6, leave=False):
        out_loader = dl.CorruptedCIFAR10(distortion, severity)
        targets = torch.cat([y for x, y in out_loader], dim=0).numpy()

        # Vanilla
        evaluate('MAP', out_loader, targets)
        evaluate('Temp', out_loader, targets)
        evaluate('DE', out_loader, targets)
        evaluate('BNO', out_loader, targets)
        evaluate('LLL', out_loader, targets)

        # ReLU-GP
        evaluate('LLL', out_loader, targets, rgpr=True, loss_type='loglik')
        evaluate('LLL', out_loader, targets, rgpr=True, loss_type='ood')


# Save dict
dir_name = f'results/CIFAR10C'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

with open(f'{dir_name}/tab_mmc.pkl', 'wb') as f:
    pickle.dump(tab_mmc, f)

with open(f'{dir_name}/tab_acc.pkl', 'wb') as f:
    pickle.dump(tab_acc, f)

with open(f'{dir_name}/tab_brier.pkl', 'wb') as f:
    pickle.dump(tab_brier, f)

with open(f'{dir_name}/tab_ece.pkl', 'wb') as f:
    pickle.dump(tab_ece, f)

with open(f'{dir_name}/tab_nll.pkl', 'wb') as f:
    pickle.dump(tab_nll, f)
