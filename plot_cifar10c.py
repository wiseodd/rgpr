import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
import os, sys, argparse
import seaborn as sns
import pickle

sns.set_style('whitegrid')
sns.set_palette('colorblind')


path = f'results/CIFAR10C'

methods = ['MAP', 'Temp', 'DE', 'BNO', 'LLL', 'LLL-RGPR-loglik', 'LLL-RGPR-ood']
method2str = {'MAP': 'MAP', 'Temp': 'Temp. Scaling', 'DE': 'Deep Ens.', 'BNO': 'GP-DSCS',
    'DKL': 'SVDKL', 'DKL-RGPR': 'SVDKL-RGPR',
    'LLL': 'LLL', 'LLL-RGPR-ood': 'LLL-RGPR-OOD', 'LLL-RGPR-loglik': 'LLL-RGPR-LL'
}
metric2str = {'acc': 'Acc.', 'mmc': 'MMC', 'ece': 'ECE', 'brier': 'Brier', 'nll': 'NLL'}

# palette = {
#     'MAP': '#0173B2', 'DE': '#CC78BC', 'LA': '#ECE133', 'LA-LULA': '#029E73'
# }


def plot(metric='ece'):
    metric_str = metric2str[metric]
    data = {'Method': [], 'Severity': [], metric_str: []}

    for method in methods:
        # vals = np.load(f'{path}/{metric}s.npy', allow_pickle=True).item()
        with open(f'{path}/tab_{metric}.pkl', 'rb') as f:
            vals = pickle.load(f)[method]
            vals = np.reshape(vals, (19, 5))  # 19 corr. types, 5 intensities

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                data['Method'].append(method)
                data['Severity'].append(j+1)
                data[metric_str].append(vals[i][j])


    df = pd.DataFrame(data)

    df_filtered = df[df['Method'].isin(methods)]

    print(df_filtered)

    # sns.boxplot(
    #     data=df_filtered, x='Severity', y=metric_str, hue='Method', fliersize=0, width=0.5,
    #     # palette=palette
    # )

    # dir_name = f'figs/CIFAR10C'
    # if not os.path.exists(dir_name):
    #     os.makedirs(dir_name)

    # tikzplotlib.save(f'{dir_name}/cifar10c_{metric}.tex')
    # plt.savefig(f'{dir_name}/cifar10c_{metric}.pdf', bbox_inches='tight')
    # plt.savefig(f'{dir_name}/cifar10c_{metric}.png', bbox_inches='tight')
    # plt.close()


plot(metric='nll')
plot(metric='ece')
plot(metric='brier')
plot(metric='mmc')
plot(metric='acc')
