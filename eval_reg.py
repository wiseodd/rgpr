import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import regression, gp
from laplace import llla_binary, kfla
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
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--randseed', type=int, default=9999)
args = parser.parse_args()

torch.cuda.set_device(0)
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

root_dir = 'data/UCI_datasets'
dset_names = ['boston_housing', 'concrete', 'energy', 'wine']

save_dir = f'pretrained_models/regression/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def read_datasets(dset_names):
    X_trains , X_tests = {}, {}
    y_trains, y_tests = {}, {}

    for name in dset_names:
        dataset = pd.read_csv(f'{root_dir}/{name}.txt', sep=r'\s+', header=None, engine='python').astype('float32').to_numpy()
        X, y = dataset[:, :-1], dataset[:, -1]

        # Train-val-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        X_trains[name] = torch.from_numpy(X_train).float()
        X_tests[name] = torch.from_numpy(X_test).float()
        y_trains[name] = torch.from_numpy(y_train).float().float()
        y_tests[name] = torch.from_numpy(y_test).float()

    return X_trains, X_tests, y_trains, y_tests

X_trains, X_tests, y_trains, y_tests = read_datasets(dset_names)

method_types = ['LLL', 'KFL', 'SWAG', 'SVGP']
method_types_rgpr = [m + '-RGPR' for m in method_types]


def load_model(type, input_dim, dataset='boston_housing', inducing_points=None):
    assert type in method_types, 'Invalid model type'

    model = regression.RegressionNet(input_dim).cuda()

    if type == 'SWAG':
        model = SWAG(
            model,
            no_cov_mat=False,
            max_num_models=20,
        ).cuda()

    if type == 'SVGP':
        model = gp.GPModel(inducing_points).cuda()
        state = torch.load(f'{save_dir}/{dataset}_gp.pt')
        model.load_state_dict(state['model'])
        model.eval()

        return model

    model.load_state_dict(torch.load(f'{save_dir}/{dataset}{"_swag" if type == "SWAG" else ""}.pt'))
    model.eval()

    return model


def predict_(X_test, model, model_name, params=None, rgpr=False, kernel_var=1, base_model=None):
    assert model_name in method_types

    if model_name == 'LLL':
        mean, var = llla_binary.predict_reg(X_test, model, *params, rgpr=rgpr, kernel_var=kernel_var)
    elif model_name == 'SVGP':
        model_dkl = model
        mean, var = predict_dkl_reg(X_test, model, rgpr=rgpr, kernel_var=kernel_var)
    else:
        mean, var = lutil.predict2_reg(
            X_test, model, scale=0.5 if model_name == 'SWAG' else 1,
            rgpr=rgpr, kernel_var=kernel_var, base_model=base_model
        )

    return mean.cpu(), var.cpu()


def evaluate(model_name, train_loader, test_loader, dataset, tab_rmse, tab_std, rgpr=False):
    assert model_name in method_types

    X_in, y_in = test_loader.dataset.tensors[0], test_loader.dataset.tensors[1]

    alpha = 2000
    faraway_test = alpha*torch.randn(1000, X_test.shape[1], device='cuda')

    inducing_points = None
    if model_name == 'SVGP':
        _, input_dim = train_loader.dataset.tensors[0].shape
        inducing_points = torch.randn(50, input_dim)

    model_ = load_model(model_name, X_train.shape[1], dataset, inducing_points)
    var0 = torch.tensor(1/(1e-1*len(train_loader.dataset)))
    params = None

    kernel_var = 0.001

    if model_name == 'LLL':
        model = model_
        model.eval()
        hessians = llla_binary.get_hessian(model, train_loader, nn.MSELoss())
        params = llla_binary.estimate_variance(var0, hessians)
    elif model_name == 'KFL':
        model = kfla.KFLA(model_)
        model.get_hessian(train_loader, regression=True)
        model.estimate_variance(var0)
        model.eval()
    else:
        model = model_

    rgpr_str = '-RGPR' if rgpr else ''
    model_str = model_name + rgpr_str

    mean, var = predict_(X_in, model, model_name, params, rgpr, kernel_var, model_)
    rmse_in = F.mse_loss(mean, y_in.squeeze().cpu()).item()
    mean_std_in = var.sqrt().mean().item()

    mean, var = predict_(faraway_test, model, model_name, params, rgpr, kernel_var, model_)
    mean_std_out = var.sqrt().mean().item()

    tab_rmse[model_str] = rmse_in
    tab_mean_std[model_str][k] = mean_std_in
    tab_mean_std[model_str]['FarAway'] = mean_std_out

    return tab_rmse, tab_mean_std


for k in dset_names:
    X_train, X_test = X_trains[k].cuda(), X_tests[k].cuda()
    y_train, y_test = y_trains[k].unsqueeze(-1).cuda(), y_tests[k].unsqueeze(-1).cuda()

    X_mean = X_train.mean(0, keepdims=True)
    X_std = X_train.std(0, keepdims=True)
    X_train = (X_train - X_mean)/X_std
    X_test = (X_test - X_mean)/X_std

    train_set = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_set, batch_size=20)

    test_set = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size=20)

    tab_rmse = {mt: {} for mt in method_types + method_types_rgpr}
    tab_mean_std = {mt: {} for mt in method_types + method_types_rgpr}

    # Vanilla
    tabs = evaluate('LLL', train_loader, test_loader, k, tab_rmse, tab_mean_std)
    tabs = evaluate('KFL', train_loader, test_loader, k, *tabs)
    tabs = evaluate('SWAG', train_loader, test_loader, k, *tabs)
    tabs = evaluate('SVGP', train_loader, test_loader, k, *tabs)

    # ReLU-GP
    tabs = evaluate('LLL', train_loader, test_loader, k, *tabs, rgpr=True)
    tabs = evaluate('KFL', train_loader, test_loader, k, *tabs, rgpr=True)
    tabs = evaluate('SWAG', train_loader, test_loader, k, *tabs, rgpr=True)
    tabs = evaluate('SVGP', train_loader, test_loader, k, *tabs, rgpr=True)

    tab_rmse, tab_mean_std = tabs

    # Save dict
    dir_name = f'results/regression/'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(f'{dir_name}/tab_rmse_{k}_{args.randseed}.pkl', 'wb') as f:
        pickle.dump(tab_rmse, f)

    with open(f'{dir_name}/tab_mean_std_{k}_{args.randseed}.pkl', 'wb') as f:
        pickle.dump(tab_mean_std, f)
