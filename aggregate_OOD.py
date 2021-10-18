import numpy as np
import pickle
import os, sys, argparse
from util.tables import *
from collections import defaultdict, namedtuple
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--mean_only', default=False, action='store_true')
parser.add_argument('--metrics', default='mmc_fpr95', choices=['mmc_fpr95', 'aur_aupr', 'acc_cal'])
parser.add_argument('--ood_dset', default='smooth', choices=['smooth', 'imagenet'])
args = parser.parse_args()


path = f'./results/hyperopt{"/imagenet" if args.ood_dset == "imagenet" else ""}'
_, _, filenames = next(os.walk(path))

if args.ood_dset != 'imagenet':
    method_types = ['MAP', 'Temp', 'DE', 'BNO', 'LLL', 'LLL-RGPR-loglik', 'LLL-RGPR-ood']
else:
    method_types = ['LLL-RGPR-ood']

method2str = {'MAP': 'MAP', 'Temp': 'Temp. Scaling', 'DE': 'Deep Ens.', 'BNO': 'GP-DSCS',
    'DKL': 'SVDKL', 'DKL-RGPR': 'SVDKL-RGPR',
    'LLL': 'LLL', 'LLL-RGPR-ood': 'LLL-RGPR-OOD', 'LLL-RGPR-loglik': 'LLL-RGPR-LL'
}
metric2str = {'fpr95': 'FPR@95', 'mmc': 'MMC', 'aur': 'AUROC', 'aupr': 'AUPRC'}
datasets = ['MNIST', 'CIFAR10', 'SVHN', 'CIFAR100']

TEXTBF = '\\textbf'


def get_dfs(dset, type='mmc', return_dicts=False):
    def cond(fname, str):
        return f'_{dset.lower()}_' in fname and str in fname

    temps = []

    fnames = [fname for fname in filenames if cond(fname, f'_{type}_')]

    for fname in fnames:
        with open(f'{path}/{fname}', 'rb') as f:
            d = pickle.load(f)

            for k in list(d.keys()):
                if not d[k]:  # d[k] is an empty dict
                    del d[k]

            # print(fname)
            # print(d);input()

            if return_dicts:
                temps.append(d)
            else:
                temps.append(pd.DataFrame(d))

    if return_dicts:
        return temps

    df = pd.concat(temps, ignore_index=False)
    df = df[(m for m in method_types)]
    df_mean = df.groupby(df.index).mean() * 100
    df_std = df.groupby(df.index).sem() * 100

    return df_mean, df_std


def get_str(test_dset, method_type, df_mean, df_std, bold=True):
    try:
        mean = df_mean[method_type][test_dset]
        std = df_std[method_type][test_dset]
    except KeyError:
        mean, std = np.NaN, np.NaN

    mean = round(mean, 1)

    if not np.isnan(mean):
        mean_str = f'\\textbf{{{mean:.1f}}}' if bold else f'{mean:.1f}'
        str = f'{mean_str}'

        if method_type not in ['MAP', 'DE']:
            str += f'$\\pm${std:.1f}'
    else:
        str = '-'

    return str


if args.metrics != 'acc_cal':
    if args.mean_only:
        metrics = args.metrics.split('_')
        vals = {m: {metrics[0]: [], metrics[1]: []} for m in method_types}

        for dset in datasets:
            for metric in metrics:
                df, _ = get_dfs(dset, type=metric)

                if metric == 'mmc':
                    df = df.drop(index=dset)

                for method in method_types:
                    vals[method][metric].append(f'{df[method].mean():.1f}')

        print()
        for i, metric in enumerate(metrics):
            print(f'\\textbf{{{metric2str[metric]}}} $\\downarrow$ \\\\')

            for method in method_types:
                if method == 'LLL-RGPR':
                    print('\\midrule')
                print(f'{method2str[method]} & {" & ".join(vals[method][metric])} \\\\')

            if i < len(metrics)-1:
                print('\n\\midrule\n\\midrule\n')
    else:
        values = {dset: defaultdict(list) for dset in datasets}

        for dset in datasets:
            metric1, metric2 = args.metrics.split('_')

            df1_mean, df1_std = get_dfs(dset, type=metric1)
            df2_mean, df2_std = get_dfs(dset, type=metric2)

            for test_dset in df1_mean.index:
                str = []

                vals1 = df1_mean.loc[test_dset].round(1).to_numpy()

                try:
                    vals2 = df2_mean.loc[test_dset].round(1).to_numpy()
                except KeyError:
                    vals2 = np.array([np.NaN]*len(vals1))

                best1 = vals1.min() if metric1 == 'mmc' else vals1.max()
                idx_best1 = vals1.argmin() if metric1 == 'mmc' else vals1.argmax()

                best2 = vals2.min() if metric2 == 'fpr95' else vals2.max()
                idx_best2 = vals2.argmin() if metric2 == 'fpr95' else vals2.argmax()

                # With error bars to test significance --- for bolding values
                best1_bar = df1_std.loc[test_dset][idx_best1].round(1)

                try:
                    best2_bar = df2_std.loc[test_dset][idx_best2].round(1)
                except KeyError:
                    best2_bar = np.array([np.NaN]*len(vals1))

                # print(max_aur, max_aur_bar)

                for method_type in method_types:
                    if metric1 == 'mmc':
                        # * is not significant if against o if: ---(---o-*-)---
                        bold = df1_mean[method_type][test_dset].round(1) <= round(best1 + best1_bar, 1)
                    else:
                        # * is not significant if against o if: ---(-*-o---)---
                        bold = df1_mean[method_type][test_dset].round(1) >= round(best1 - best1_bar, 1)
                    str1 = get_str(test_dset, method_type, df1_mean, df1_std, bold=False if test_dset == dset else bold)

                    try:
                        if metric2 == 'fpr95':
                            # * is not significant if against o if: ---(---o-*-)---
                            bold = df2_mean[method_type][test_dset].round(1) <= round(best2 + best2_bar, 1)
                        else:
                            # * is not significant if against o if: ---(-*-o---)---
                            bold = df2_mean[method_type][test_dset].round(1) >= round(best2 - best2_bar, 1)
                    except KeyError:
                        bold = [False]*len(vals1)

                    str2 = get_str(test_dset, method_type, df2_mean, df2_std, bold=False if test_dset == dset else bold)

                    str.append(str1)
                    str.append(str2)

                values[dset][test_dset] = str

        print()


        ood_noise_names = ['UniformNoise']
        # ood_noise_names = []
        ood_test_names = {
            'MNIST': ['EMNIST', 'KMNIST', 'FMNIST', 'GrayCIFAR10'],
            'CIFAR10': ['SVHN', 'LSUN', 'CIFAR100', 'FMNIST3D'],
            'SVHN': ['CIFAR10', 'LSUN', 'CIFAR100', 'FMNIST3D'],
            'CIFAR100': ['SVHN', 'LSUN', 'CIFAR10', 'FMNIST3D'],
        }


        for i, dset in enumerate(datasets):
            print(f'\\textbf{{{dset}}} & {" & ".join(values[dset][dset])} \\\\')

            for ood_dset in ood_test_names[dset] + ood_noise_names:
                print(f'{ood_dset} & {" & ".join(values[dset][ood_dset])} \\\\')

            if i < len(datasets)-1:
                print()
                print('\\midrule')
                print()

    print()

else:
    # Accuracy & Calibration
    # ----------------------

    for method_type in method_types:
        if method_type == 'LLL-RGPR-loglik':
            print(r'\midrule')

        str = method2str[method_type] + ' '

        for dset in datasets:
            df_acc = pd.DataFrame(get_dfs(dset, 'acc', return_dicts=True)).mean()
            str += f'& {df_acc[method_type]*100:.1f} '

        print(str + r'\\')

    print()
    print(r'\midrule')
    print(r'\midrule')
    print()

    for method_type in method_types:
        if method_type == 'LLL-RGPR-loglik':
            print(r'\midrule')

        str = method2str[method_type] + ' '

        for dset in datasets:
            df_acc = pd.DataFrame(get_dfs(dset, 'cal', return_dicts=True)).mean()
            str += f'& {df_acc[method_type]:.1f} '

        print(str + r'\\')
