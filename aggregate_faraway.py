import numpy as np
import pickle
import os, sys, argparse
from util.tables import *
from collections import defaultdict
import pandas as pd


path = f'./results/faraway'
_, _, filenames = next(os.walk(path))

# method_types = ['BNO', 'LLL', 'KFL', 'SWAG', 'SVDKL']
method_types = ['KFL', 'SWAG', 'SVDKL']
rgpr_suffix = '-RGPR'
# datasets = ['MNIST', 'CIFAR10', 'SVHN', 'CIFAR100']
datasets = ['CIFAR10', 'SVHN', 'CIFAR100']
faraway = 'FarAway'

TEXTBF = '\\textbf'

values = defaultdict(list)

for dset in datasets:
    mmcs = []; aurs = []

    def cond(fname, str):
        return f'_{dset.lower()}_' in fname and str in fname

    # ========================== MMC ====================================

    for fname in [fname for fname in filenames if cond(fname, '_mmc_')]:
        with open(f'{path}/{fname}', 'rb') as f:
            d = pickle.load(f)
            mmcs.append(pd.DataFrame(d))

    df_mmc = pd.concat(mmcs, ignore_index=False)
    df_mmc_mean = df_mmc.groupby(df_mmc.index).mean() * 100
    df_mmc_std = df_mmc.groupby(df_mmc.index).sem() * 100

    for method_type in method_types:
        mean_vanilla = df_mmc_mean[method_type]["FarAway"]
        mean_rgp = df_mmc_mean[method_type+rgpr_suffix]["FarAway"]

        std_vanilla = df_mmc_std[method_type]["FarAway"]
        std_rgp = df_mmc_std[method_type+rgpr_suffix]["FarAway"]

        bold_vanilla = mean_vanilla <= mean_rgp
        bold_rgp = mean_rgp <= mean_vanilla

        mean_vanilla_str = f'\\textbf{{{mean_vanilla:.1f}}}' if bold_vanilla else f'{mean_vanilla:.1f}'
        mean_rgp_str = f'\\textbf{{{mean_rgp:.1f}}}' if bold_rgp else f'{mean_rgp:.1f}'

        str_vanilla = f'{mean_vanilla_str}$\\pm${std_vanilla:.1f}'
        str_rgp = f'{mean_rgp_str}$\\pm${std_rgp:.1f}'

        values[method_type].append(str_vanilla)
        values[method_type+rgpr_suffix].append(str_rgp)

    # ========================== AUR ====================================

    for fname in [fname for fname in filenames if cond(fname, '_aur_')]:
        with open(f'{path}/{fname}', 'rb') as f:
            d = pickle.load(f)
            aurs.append(pd.DataFrame(d))

    df_aur = pd.concat(aurs, ignore_index=False)
    df_aur_mean = df_aur.groupby(df_aur.index).mean() * 100
    df_aur_std = df_aur.groupby(df_aur.index).sem() * 100

    for method_type in method_types:
        mean_vanilla = df_aur_mean[method_type]["FarAway"]
        mean_rgp = df_aur_mean[method_type+rgpr_suffix]["FarAway"]

        std_vanilla = df_aur_std[method_type]["FarAway"]
        std_rgp = df_aur_std[method_type+rgpr_suffix]["FarAway"]

        bold_vanilla = mean_vanilla >= mean_rgp
        bold_rgp = mean_rgp >= mean_vanilla

        mean_vanilla_str = f'\\textbf{{{mean_vanilla:.1f}}}' if bold_vanilla else f'{mean_vanilla:.1f}'
        mean_rgp_str = f'\\textbf{{{mean_rgp:.1f}}}' if bold_rgp else f'{mean_rgp:.1f}'

        str_vanilla = f'{mean_vanilla_str}$\\pm${std_vanilla:.1f}'
        str_rgp = f'{mean_rgp_str}$\\pm${std_rgp:.1f}'

        values[method_type].append(str_vanilla)
        values[method_type+rgpr_suffix].append(str_rgp)

print()

for i, method_type in enumerate(values.keys()):
    if i % 2 == 0 and i > 0:
        print()
        print('\\midrule')
        print()

    latex_str = f'{method_type} & {" & ".join(values[method_type])} \\\\'
    print(latex_str)

print()


print()
print("==================================================================================")
print()


# ========================== MMC In ====================================
values = defaultdict(list)

for dset in datasets:
    mmcs = []

    def cond(fname, str):
        return f'_{dset.lower()}_' in fname and str in fname

    for fname in [fname for fname in filenames if cond(fname, '_mmc_')]:
        with open(f'{path}/{fname}', 'rb') as f:
            d = pickle.load(f)
            mmcs.append(pd.DataFrame(d))

    df_mmc = pd.concat(mmcs, ignore_index=False)
    df_mmc_mean = df_mmc.groupby(df_mmc.index).mean() * 100
    df_mmc_std = df_mmc.groupby(df_mmc.index).sem() * 100

    for method_type in method_types:
        mean_vanilla = round(df_mmc_mean[method_type][dset], 1)
        mean_rgp = round(df_mmc_mean[method_type+rgpr_suffix][dset], 1)
        std_vanilla = round(df_mmc_std[method_type][dset], 1)
        std_rgp = round(df_mmc_std[method_type+rgpr_suffix][dset], 1)

        std_vanilla = df_mmc_std[method_type][dset]
        std_rgp = df_mmc_std[method_type+rgpr_suffix][dset]

        # * not significant against o if: ------(-*-o---)-------
        bold_vanilla = mean_vanilla >= round(mean_rgp-std_rgp, 1)
        bold_rgp = mean_rgp >= round(mean_vanilla-std_vanilla, 1)

        str_vanilla = f'\\textbf{{{mean_vanilla:.1f}}}' if bold_vanilla else f'{mean_vanilla:.1f}'
        str_rgp = f'\\textbf{{{mean_rgp:.1f}}}' if bold_rgp else f'{mean_rgp:.1f}'

        str_vanilla += f'$\\pm${std_vanilla:.1f}'
        str_rgp += f'$\\pm${std_rgp:.1f}'

        values[method_type].append(str_vanilla)
        values[method_type+rgpr_suffix].append(str_rgp)

print()

for i, method_type in enumerate(values.keys()):
    if i % 2 == 0 and i > 0:
        print()
        print('\\midrule')
        print()

    latex_str = f'{method_type} & {" & ".join(values[method_type])} \\\\'
    print(latex_str)

print()
