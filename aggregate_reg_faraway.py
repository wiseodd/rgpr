import numpy as np
import pickle
import os, sys, argparse
from util.tables import *
from collections import defaultdict
import pandas as pd


path = f'./results/regression'
_, _, filenames = next(os.walk(path))

method_types = ['LLL', 'KFL', 'SWAG', 'SVGP']
rgpr_suffix = '-RGPR'
datasets = ['boston_housing', 'concrete', 'energy', 'wine']

FARAWAY = 'FarAway'
TEXTBF = '\\textbf'


# ========================== Error-bars ====================================
values = defaultdict(list)

for dset in datasets:
    stds = []

    def cond(fname, str):
        return f'_{dset.lower()}_' in fname and str in fname

    for fname in [fname for fname in filenames if cond(fname, '_std_')]:
        with open(f'{path}/{fname}', 'rb') as f:
            d = pickle.load(f)
            stds.append(pd.DataFrame(d))

    df_std = pd.concat(stds, ignore_index=False)
    df_std_mean = df_std.groupby(df_std.index).mean()

    for method_type in method_types:
        mean_vanilla = df_std_mean[method_type][dset]
        mean_rgp = df_std_mean[method_type+rgpr_suffix][dset]

        # bold_vanilla = mean_vanilla <= mean_rgp
        # bold_rgp = mean_rgp <= mean_vanilla
        bold_vanilla = False
        bold_rgp = False

        str_vanilla = f'\\textbf{{{mean_vanilla:.3f}}}' if bold_vanilla else f'{mean_vanilla:.3f}'
        str_rgp = f'\\textbf{{{mean_rgp:.3f}}}' if bold_rgp else f'{mean_rgp:.3f}'

        values[method_type].append(str_vanilla)
        values[method_type+rgpr_suffix].append(str_rgp)

    for method_type in method_types:
        mean_vanilla = df_std_mean[method_type][FARAWAY]
        mean_rgp = df_std_mean[method_type+rgpr_suffix][FARAWAY]

        bold_vanilla = mean_vanilla >= mean_rgp
        bold_rgp = mean_rgp >= mean_vanilla

        str_vanilla = f'\\textbf{{{mean_vanilla:.3f}}}' if bold_vanilla else f'{mean_vanilla:.3f}'
        str_rgp = f'\\textbf{{{mean_rgp:.3f}}}' if bold_rgp else f'{mean_rgp:.3f}'

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
print("==================================================================================")
print()


# ========================== RMSE ====================================
values = defaultdict(list)

for dset in datasets:
    rmses = []

    def cond(fname, str):
        return dset.lower() in fname and str in fname

    for fname in [fname for fname in filenames if cond(fname, '_rmse_')]:
        with open(f'{path}/{fname}', 'rb') as f:
            d = pickle.load(f)
            # print(d); input()
            rmses.append(pd.DataFrame([d]))

    df_rmse = pd.concat(rmses, ignore_index=False)
    df_rmse_mean = df_rmse.groupby(df_rmse.index).mean()
    df_rmse_std = df_rmse.groupby(df_rmse.index).std()

    for method_type in method_types:
        mean_vanilla = df_rmse_mean[method_type][0]
        mean_rgp = df_rmse_mean[method_type+rgpr_suffix][0]

        std_vanilla = df_rmse_std[method_type][0]
        std_rgp = df_rmse_std[method_type+rgpr_suffix][0]

        bold_vanilla = mean_vanilla <= round(mean_rgp+std_rgp, 1)
        bold_rgp = mean_rgp <= round(mean_vanilla+std_vanilla, 1)

        str_vanilla = f'\\textbf{{{mean_vanilla:.3f}}}' if bold_vanilla else f'{mean_vanilla:.3f}'
        str_rgp = f'\\textbf{{{mean_rgp:.3f}}}' if bold_rgp else f'{mean_rgp:.3f}'

        str_vanilla += f'$\pm${std_vanilla:.3f}'
        str_rgp += f'$\pm${std_rgp:.3f}'

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
