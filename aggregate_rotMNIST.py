import numpy as np
import pickle
import os, sys, argparse
import matplotlib
import matplotlib.cm as cm
from math import *
import tikzplotlib
import tqdm
import seaborn as sns; sns.set_style('whitegrid')
sns.set_palette('tab10')

matplotlib.rcParams['figure.figsize'] = (11,8)
matplotlib.rcParams['font.size'] = 30
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams['font.serif'] = 'Times'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['lines.linewidth'] = 1.0
plt = matplotlib.pyplot


path = f'./results/rotMNIST'
_, _, filenames = next(os.walk(path))

method_types = ['MAP', 'Temp', 'DE', 'BNO', 'LLL', 'LLL-RGPR-loglik', 'LLL-RGPR-ood']
method2str = {'MAP': 'MAP', 'Temp': 'Temp. Scaling', 'DE': 'Deep Ens.', 'BNO': 'GP-DSCS',
    'DKL': 'SVDKL', 'DKL-RGPR': 'SVDKL-RGPR',
    'LLL': 'LLL', 'LLL-RGPR-ood': 'LLL-RGPR-OOD', 'LLL-RGPR-loglik': 'LLL-RGPR-LL'
}
x = list(range(0, 181, 15))


def load(str='MMC'):
    out = []
    for fname in [fname for fname in filenames if f'_{str.lower()}_' in fname]:
        with open(f'{path}/{fname}', 'rb') as f:
            out.append(pickle.load(f))
    return out


def aggregate(lst):
    d_mean = {}
    d_std = {}

    for k in lst[0].keys():
        samples = np.array([l[k] for l in lst])
        d_mean[k] = np.mean(samples, axis=0)
        d_std[k] = np.std(samples, axis=0)

    return d_mean, d_std


def plot(means, stds, name, legend=False, plot_var=False):
    plt.figure()

    for method in method_types:
        m, s = means[method], stds[method]
        plt.plot(x, m, lw=3, label=method2str[method], alpha=0.75)

        if plot_var:
            plt.fill_between(x, m-s, m+s, interpolate=True, alpha=0.1)

    plt.xticks(range(0, 181, 30), rotation=45)
    plt.xlim(0, 180)
    plt.ylim(bottom=0)

    if name in ['mmc', 'acc', 'aur']:
        plt.ylim(top=1)

    if legend:
        plt.legend(loc='lower right')

    tikzplotlib.save(f'notebooks/figs/rotMNIST_{name}.tex')
    plt.savefig(f'notebooks/figs/rotMNIST_{name}.pdf', bbox_inches='tight')


metrics = ['acc', 'nll', 'ece', 'brier', 'mmc']

for metric in tqdm.tqdm(metrics):
    means, stds = aggregate(load(metric))
    plot(means, stds, metric, legend=True, plot_var=True)
