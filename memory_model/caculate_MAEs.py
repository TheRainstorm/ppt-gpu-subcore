import argparse
import json
import sys, os
import pandas as pd
import numpy as np

# trick to import from parent directory
curr_dir = os.path.dirname(__file__)
par_dir = os.path.dirname(curr_dir)
sys.path.insert(0, os.path.abspath(par_dir))

cmp_list = ["l1_hit_rate", "l1_hit_rate_ld", "l1_hit_rate_st", "l2_hit_rate", "l2_hit_rate_ld", "l2_hit_rate_st",
            "l2_ld_trans","l2_st_trans","l2_tot_trans","dram_ld_trans","dram_st_trans","dram_tot_trans"]

def process_hw(hw_res):
    for app_arg, app_res in hw_res.items():
        for i, kernel_res in enumerate(app_res):
            # # "l1_hit_rate": "tex_cache_hit_rate",
            kernel_res['l1_hit_rate'] = kernel_res['global_hit_rate'] / 100
            kernel_res['l1_hit_rate_ld'] = kernel_res['global_hit_rate_ld'] / 100
            kernel_res['l1_hit_rate_st'] = kernel_res['global_hit_rate_st'] / 100
            kernel_res['l2_hit_rate'] = kernel_res['l2_tex_hit_rate'] / 100
            kernel_res['l2_hit_rate_ld'] = kernel_res['l2_tex_read_hit_rate'] / 100
            kernel_res['l2_hit_rate_st'] = kernel_res['l2_tex_write_hit_rate'] / 100
        
            kernel_res['gmem_tot_reqs'] = kernel_res['global_load_requests'] + kernel_res['global_store_requests']
            kernel_res['gmem_ld_reqs'] = kernel_res['global_load_requests']
            kernel_res['gmem_st_reqs'] = kernel_res['global_store_requests']
            kernel_res['gmem_tot_sectors'] = kernel_res['gld_transactions'] + kernel_res['gst_transactions']
            kernel_res['gmem_ld_sectors'] = kernel_res['gld_transactions']
            kernel_res['gmem_st_sectors'] = kernel_res['gst_transactions']
            kernel_res['gmem_ld_diverg'] = kernel_res['gld_transactions_per_request']
            kernel_res['l2_tot_trans'] = kernel_res['l2_read_transactions'] + kernel_res['l2_write_transactions']
            kernel_res['l2_ld_trans'] = kernel_res['l2_read_transactions']
            kernel_res['l2_st_trans'] = kernel_res['l2_write_transactions']
            kernel_res['dram_tot_trans'] = kernel_res['dram_read_transactions'] + kernel_res['dram_write_transactions']
            kernel_res['dram_ld_trans'] = kernel_res['dram_read_transactions']
            kernel_res['dram_st_trans'] = kernel_res['dram_write_transactions']
    return hw_res

def json2df(json_data):
    data = {}
    data['bench'] = []
    data['app'] = []
    data['kernel_id'] = []
    for key in cmp_list:
        data[f"{key}"] = []
    
    for app_arg, app_res in json_data.items():
        bench = suite_info['map'][app_arg][0]
        for i, kernel_res in enumerate(app_res):
            data['bench'].append(bench)
            data['app'].append(app_arg)
            data['kernel_id'].append(i)
            for key in cmp_list:
                data[f"{key}"].append(kernel_res[key])
    df = pd.DataFrame(data)
    return df

def get_summary(df):
    df_summary = pd.DataFrame(columns=["key", "MAE", "NRMSE", "corr"])
    for key in cmp_list:
        y1 = np.array(df[f"{key}_hw"].to_numpy())
        y2 = np.array(df[f"{key}_sim"].to_numpy())
        non_zero_idxs = y1 != 0
        MAE = np.mean(np.abs(y1[non_zero_idxs] - y2[non_zero_idxs])/y1[non_zero_idxs])
        NRMSE = np.sqrt(np.mean((y1 - y2)**2)) / np.mean(y1)
        corr = np.corrcoef(y1, y2)[0, 1]
        df_summary.loc[-1] = [key, MAE, NRMSE, corr]
        df_summary.index = df_summary.index + 1
    return df_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument("-S", "--sim_res",
                        required=True)
    parser.add_argument("-H", "--hw_res",
                        required=True)
    parser.add_argument("-o", "--output-file",
                        default="memory_res.xlsx")
    parser.add_argument("-B", "--benchmark_list",
                        help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for the benchmark suite names.",
                        default="")
    parser.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
    parser.add_argument("-c", "--limit_kernel_num",
                        type=int,
                        default=300)
    args = parser.parse_args()


    from scripts.common import *
    from scripts.draw.draw_1 import filter_res,truncate_kernel,find_common

    apps = gen_apps_from_suite_list(args.benchmark_list)
    app_and_arg_list = get_app_arg_list(apps)
    app_arg_filtered_list = filter_app_list(app_and_arg_list, args.app_filter)

    with open(args.hw_res, 'r') as f:
        hw_res = json.load(f)

    with open(args.sim_res, 'r') as f:
        sim_res = json.load(f)

    sim_res = filter_res(sim_res, app_arg_filtered_list)
    hw_res = filter_res(hw_res, app_arg_filtered_list)
    sim_res = truncate_kernel(sim_res, args.limit_kernel_num)
    sim_res, hw_res = find_common(sim_res, hw_res)
    hw_res = process_hw(hw_res)
    
    df_sim = json2df(sim_res)
    df_hw = json2df(hw_res)
    df_sim.rename(columns={f"{key}": f"{key}_sim" for key in cmp_list}, inplace=True)
    df_hw.rename(columns={f"{key}": f"{key}_hw" for key in cmp_list}, inplace=True)
    df_kernels = pd.concat([df_sim, df_hw.iloc[:,3:]], axis=1)  # all res kernels level
    
    writer = pd.ExcelWriter(args.output_file, engine='xlsxwriter')
    
    def write_df_and_bench_summary(df, prefix):
        df.to_excel(writer, sheet_name=f'{prefix}', index=False)
        df_summary = get_summary(df)
        df_summary.to_excel(writer, sheet_name=f'{prefix}_summary', index=False)
        
        df_group_bench = df.groupby("bench", sort=False, as_index=False)
        for bench, group in df_group_bench:
            df_summary = get_summary(group)
            df_summary.to_excel(writer, sheet_name=f'{prefix}_{bench}_summary', index=False)
    
    df_apps = df_kernels.groupby(["bench", "app"], sort=False, as_index=False).mean(numeric_only=True)
    write_df_and_bench_summary(df_apps, 'apps')
    
    write_df_and_bench_summary(df_kernels, 'kernels')
    
    writer.close()
    print("Done!")