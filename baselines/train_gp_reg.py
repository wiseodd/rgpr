import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import torch
import torch.nn.functional as F
from models.gp import GPModel
from util.evaluation import *
from util.tables import *
import util.dataloaders as dl
from util.misc import *
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import os
import math
import gpytorch
import pandas as pd
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--randseed', type=int, default=9999)
args = parser.parse_args()

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
batch_size = 20
print()

for k in dset_names:
    print(k)

    X_train, X_test = X_trains[k].cuda(), X_tests[k].cuda()
    y_train, y_test = y_trains[k].cuda(), y_tests[k].cuda()

    X_mean = X_train.mean(0, keepdims=True)
    X_std = X_train.std(0, keepdims=True)
    X_train = (X_train - X_mean)/X_std
    X_test = (X_test - X_mean)/X_std

    model = GPModel(inducing_points=X_train[:50, :]).cuda()
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()

    opt = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=1e-2)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train.shape[0])

    pbar = trange(100)

    for epoch in pbar:
        model.train()
        likelihood.train()

        for i in range(0, X_train.shape[0], batch_size):
            x = X_train[i:i+batch_size]
            y = y_train[i:i+batch_size]

            output = model(x)
            loss = -mll(output, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        with torch.no_grad():
            if epoch % 10 == 0:
                model.eval()
                likelihood.eval()

                out = model(X_test).mean
                test_rmse = torch.sqrt(F.mse_loss(out, y_test))
                pbar.set_description(f'Train loss: {loss.item():.3f}; Test RMSE: {test_rmse:.3f}')


    torch.save({'model': model.state_dict(), 'likelihood': likelihood.state_dict()}, f'{save_dir}/{k}_gp.pt')

    print()
