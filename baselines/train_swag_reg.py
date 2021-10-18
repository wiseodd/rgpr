import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.regression import RegressionNet
from sklearn import datasets
from math import *
import pandas as pd
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import os
from swag.posteriors import SWAG


np.random.seed(9999)
torch.manual_seed(9999)
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

    model = RegressionNet(X_train.shape[1]).cuda()
    swag_model = SWAG(
        RegressionNet(X_train.shape[1]).cuda(),
        no_cov_mat=False,
        max_num_models=20,
    ).cuda()

    opt = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-1)
    pbar = trange(100)

    for epoch in pbar:
        for i in range(0, X_train.shape[0], batch_size):
            x = X_train[i:i+batch_size]
            y = y_train[i:i+batch_size]

            out = model(x).squeeze()
            loss = F.mse_loss(out, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        with torch.no_grad():
            if epoch % 10 == 0:
                out_test = model(X_test).squeeze()
                test_rmse = torch.sqrt(F.mse_loss(out_test, y_test))
                pbar.set_description(f'Train loss: {loss.item():.3f}; Test RMSE: {test_rmse:.3f}')


    print('SWAG Training')
    pbar = trange(50)

    for epoch in pbar:
        for i in range(0, X_train.shape[0], batch_size):
            x = X_train[i:i+batch_size]
            y = y_train[i:i+batch_size]

            out = model(x).squeeze()
            loss = F.mse_loss(out, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        swag_model.collect_model(model)

        with torch.no_grad():
            if epoch % 10 == 0:
                swag_model.sample(0.0)
                out_test = swag_model(X_test).squeeze()
                test_rmse = torch.sqrt(F.mse_loss(out_test, y_test))
                pbar.set_description(f'Train loss: {loss.item():.3f}; Test RMSE: {test_rmse:.3f}')

    torch.save(swag_model.state_dict(), f'{save_dir}/{k}_swag.pt')

    print()

