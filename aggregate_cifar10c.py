import numpy as np
import pandas as pd
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


path = f'results/CIFAR10C'

method_types = ['MAP', 'Temp', 'DE', 'BNO', 'LLL', 'LLL-RGPR-loglik', 'LLL-RGPR-ood']
method2str = {'MAP': 'MAP', 'Temp': 'Temp. Scaling', 'DE': 'Deep Ens.', 'BNO': 'GP-DSCS',
    'DKL': 'SVDKL', 'DKL-RGPR': 'SVDKL-RGPR',
    'LLL': 'LLL', 'LLL-RGPR-ood': 'LLL-RGPR-OOD', 'LLL-RGPR-loglik': 'LLL-RGPR-LL'
}
metric2str = {'acc': 'Acc.', 'mmc': 'MMC', 'ece': 'ECE', 'brier': 'Brier', 'nll': 'NLL'}


def load(metric):
    with open(f'{path}/tab_{metric}.pkl', 'rb') as f:
        return pickle.load(f)


data = {'Method': [], 'Metric': [], 'Values': []}
metrics = ['NLL', 'Brier', 'ECE', 'MMC', 'Acc']

for method in method_types:
    for metric in metrics:
        vals = load(metric.lower())[method]
        for val in vals:
            data['Method'].append(method2str[method])
            data['Metric'].append(metric)
            data['Values'].append(val)

df = pd.DataFrame(data)

# Normalize each metric
for metric in metrics:
    vals = df[df.Metric == metric]['Values'].values
    vals = (vals - vals.min()) / (vals.max() - vals.min())
    df.loc[df.Metric == metric, 'Values'] = vals

sns.catplot(
    data=df, x='Metric', y='Values', hue='Method',
    kind='bar',
    # ci=None
)

tikzplotlib.save(f'notebooks/figs/cifar10c.tex')
plt.show()
