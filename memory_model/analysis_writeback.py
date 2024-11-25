import argparse
import json
import sys, os
import pandas as pd
import numpy as np

xlsx_file = sys.argv[1]
output_xlsx_file = sys.argv[2]

xlsx_file = 'memory_simulator_sector-base.xlsx'
output_xlsx_file = 'writeback.xlsx'

# Returns a DataFrame
df = pd.read_excel(xlsx_file, sheet_name="kernels")

df['dram_st_trans_caulate'] = df['l2_st_trans_sim'] * (1 - df['l2_hit_rate_st_sim'])

g = df.groupby(["bench", "app"], sort=False, as_index=False)
df_apps = g.mean(numeric_only=True)
df_apps['kernels'] = g.size()['size']


def get_MAE(y1, y2):
    y1 = np.array(y1.to_numpy())
    y2 = np.array(y2.to_numpy())
    non_zero_idxs = y1 != 0
    MAE = np.mean(np.abs(y1[non_zero_idxs] - y2[non_zero_idxs])/y1[non_zero_idxs])
    NRMSE = np.sqrt(np.mean((y1 - y2)**2)) / np.mean(y1)
    corr = np.corrcoef(y1, y2)[0, 1]
    print({"MAE": MAE, "NRMSE": NRMSE, "corr": corr})

get_MAE(df_apps['dram_st_trans_hw'], df_apps['dram_st_trans_caulate'])
get_MAE(df_apps['dram_st_trans_hw'], df_apps['dram_st_trans_sim'])

df_output = df_apps[['bench', 'app', 'kernels', 'dram_st_trans_caulate', 'dram_st_trans_hw', 'dram_st_trans_sim']]
df_output.to_excel(output_xlsx_file, sheet_name="apps", index=False, engine='xlsxwriter')
