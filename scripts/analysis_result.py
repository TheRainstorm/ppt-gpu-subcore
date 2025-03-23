import argparse
import json
import sys, os
import pandas as pd
import numpy as np

# trick to import from parent directory
curr_dir = os.path.dirname(__file__)
par_dir = os.path.dirname(curr_dir)
sys.path.insert(0, os.path.abspath(par_dir))

cmp_list = ['warp_inst_executed', 'achieved_occupancy', 'gpu_active_cycles', 'sm_active_cycles_sum', 'ipc',
            'gmem_diverg', "l1_hit_rate", "l1_hit_rate_g", "l1_hit_rate_ldg", "l1_hit_rate_stg",
            "l2_hit_rate", "l2_hit_rate_ld", "l2_hit_rate_st",
            "l2_ld_trans","l2_st_trans","l2_tot_trans",
            "dram_ld_trans","dram_st_trans","dram_tot_trans",
            "gmem_tot_reqs", "gmem_ld_sectors", "gmem_st_sectors", "gmem_tot_sectors", "gmem_ld_diverg"]
            # 'l1_hit_rate', 'l2_hit_rate', 'gmem_tot_reqs', 'gmem_tot_sectors', 'gmem_diverg', 'l2_tot_trans', 'dram_tot_trans']
extra_list = ['sim_time', 'sim_time_memory', 'sim_time_compute', 'warp_inst_executed', 'gpu_active_cycles', 'AMAT']

def divide_or_zero(a, b):
    return a/b if b != 0 else 0

def process_gpgpu_sim(res):
    global extra_list
    for app_arg, app_res in res.items():
        for i, kernel_res in enumerate(app_res):
            kernel_res['warp_inst_executed'] = kernel_res['gpu_inst']//32
            kernel_res['achieved_occupancy'] = kernel_res['gpu_occupancy'] / 100
            kernel_res['gpu_active_cycles'] = kernel_res['gpu_cycle']
            kernel_res['ipc'] = kernel_res['gpu_ipc']
            kernel_res['sim_time'] = kernel_res['gpgpu_simulation_time']
    return res

def process_sim(sim_res):
    global extra_list
    extra_flag = False
    for app_arg, app_res in sim_res.items():
        for i, kernel_res in enumerate(app_res):
            kernel_res['warp_inst_executed'] = kernel_res['warp_inst_executed']
            kernel_res['achieved_occupancy'] = kernel_res['achieved_occupancy'] / 100
            kernel_res['gpu_active_cycles'] = kernel_res['gpu_active_cycles']
            kernel_res['sm_active_cycles_sum'] = kernel_res['sm_active_cycles_sum']
            kernel_res['ipc'] = kernel_res['ipc']

            kernel_res["l1_hit_rate"] = kernel_res["l1_hit_rate"] / 100
            kernel_res["l2_hit_rate"] = kernel_res["l2_hit_rate"] / 100
            kernel_res['gmem_tot_reqs'] = kernel_res['memory_stats']['gmem_tot_reqs']
            kernel_res['gmem_tot_sectors'] = kernel_res['memory_stats']['gmem_tot_trans']
            kernel_res['l2_tot_trans'] = kernel_res['memory_stats']['l2_tot_trans_gmem']
            kernel_res['dram_tot_trans'] = kernel_res['memory_stats']['dram_tot_trans_gmem']
            
            # more memory
            if 'lmem_hit_rate' in kernel_res['memory_stats']:
                try:
                    # ppt-gpu ori
                    memory_stats = kernel_res['memory_stats']
                    kernel_res['l1_hit_rate'] = memory_stats['umem_hit_rate']
                    kernel_res['l1_hit_rate_g'] = memory_stats['gmem_hit_rate']
                    kernel_res['l1_hit_rate_ldg'] = memory_stats['gmem_hit_rate_lds']
                    kernel_res['l1_hit_rate_l'] = memory_stats['lmem_hit_rate']
                    kernel_res['l1_hit_rate_stg'] = divide_or_zero(
                        kernel_res['l1_hit_rate']*memory_stats['gmem_tot_trans'] - memory_stats['gmem_hit_rate_lds']*memory_stats['gmem_ld_trans'],
                        memory_stats['gmem_st_trans'])
                    
                    kernel_res['l2_hit_rate'] = memory_stats['hit_rate_l2']
                    kernel_res['l2_hit_rate_ld'] = kernel_res['l2_hit_rate_st'] = memory_stats['hit_rate_l2']

                    kernel_res['gmem_ld_sectors'] = memory_stats['gmem_ld_trans']  # ppt-gpu has no sector level
                    kernel_res['gmem_st_sectors'] = memory_stats['gmem_st_trans']
                    kernel_res['gmem_tot_sectors'] = memory_stats['gmem_tot_trans']

                    kernel_res['l2_ld_trans'] = memory_stats['l2_ld_trans_gmem']
                    kernel_res['l2_st_trans'] = memory_stats['l2_st_trans_gmem']
                    kernel_res['l2_tot_trans'] = memory_stats['l2_tot_trans_gmem']
                    
                    kernel_res['dram_tot_trans'] = memory_stats['dram_tot_trans_gmem']
                    if kernel_res['dram_tot_trans']==0:
                        kernel_res['dram_ld_trans'] = kernel_res['dram_st_trans'] = 0
                    else:
                        kernel_res['dram_ld_trans'] = memory_stats['dram_ld_trans_gmem']
                        kernel_res['dram_st_trans'] = memory_stats['dram_st_trans_gmem']

                    kernel_res['gmem_tot_sectors_per_sm'] = memory_stats['l1_sm_trans_gmem']
                    kernel_res['atom_tot_reqs'] = memory_stats['atom_tot_reqs']
                    kernel_res['red_tot_reqs'] = memory_stats['red_tot_reqs']
                    kernel_res['atom_red_trans'] = memory_stats['atom_red_tot_trans']
                    
                    kernel_res['gmem_ld_diverg'] = memory_stats['gmem_ld_diverg']
                    kernel_res['gmem_st_diverg'] = memory_stats['gmem_st_diverg']
                    kernel_res['gmem_diverg'] = memory_stats['gmem_tot_diverg']
                except Exception as e:
                    print(f"{e}: {app_arg} {i}")
                    exit(1)
            else:
                kernel_res["l1_hit_rate_g"] = kernel_res['memory_stats']["l1_hit_rate_g"]
                kernel_res["l1_hit_rate_ldg"] = kernel_res['memory_stats']["l1_hit_rate_ldg"]
                kernel_res["l1_hit_rate_stg"] = kernel_res['memory_stats']["l1_hit_rate_stg"]
                kernel_res["l2_hit_rate_ld"] = kernel_res['memory_stats']["l2_hit_rate_ld"]
                kernel_res["l2_hit_rate_st"] = kernel_res['memory_stats']["l2_hit_rate_st"]
                
                kernel_res['l2_ld_trans'] = kernel_res['memory_stats']['l2_ld_trans_gmem']
                kernel_res['l2_st_trans'] = kernel_res['memory_stats']['l2_st_trans_gmem']
                kernel_res['dram_ld_trans'] = kernel_res['memory_stats']['dram_ld_trans']
                kernel_res['dram_st_trans'] = kernel_res['memory_stats']['dram_st_trans']
                
                kernel_res['gmem_ld_sectors'] = kernel_res['memory_stats']['gmem_ld_sectors']
                kernel_res['gmem_st_sectors'] = kernel_res['memory_stats']['gmem_st_sectors']
                
                kernel_res['gmem_diverg'] = kernel_res['gmem_tot_sectors'] / kernel_res['gmem_tot_reqs'] if kernel_res['gmem_tot_reqs'] != 0 else 0
                kernel_res['gmem_ld_diverg'] = kernel_res['memory_stats']['gmem_ld_diverg']
            
            # extra
            kernel_res['AMAT'] = kernel_res['AMAT']
            kernel_res['sim_time_memory'] = kernel_res["simulation_time"]["memory"]
            kernel_res['sim_time_compute'] = kernel_res["simulation_time"]["compute"]
            kernel_res['sim_time'] = kernel_res['sim_time_memory'] + kernel_res['sim_time_compute']
            try:
                kernel_res['ACPAO'] = kernel_res['ACPAO']
                kernel_res['grid_size'] = kernel_res['grid_size']
                kernel_res['block_size'] = kernel_res['block_size']
                kernel_res['max_active_block_per_sm'] = kernel_res['max_active_block_per_sm']
                kernel_res['allocted_block_per_sm'] = kernel_res['allocted_block_per_sm']
                kernel_res['kernel_lat'] = kernel_res['kernel_detail']['kernel_lat']
                kernel_res['gpu_active_cycles'] = kernel_res['gpu_active_cycles']
                kernel_res['tot_ipc'] = kernel_res['tot_ipc']
                kernel_res['tot_cpi'] = 1/kernel_res['tot_ipc']
                kernel_res['active_block_per_cycle_sm'] = kernel_res['active_block_per_cycle_sm']
                kernel_res['active_warp_per_cycle_smsp'] = kernel_res['active_warp_per_cycle_smsp']
                kernel_res['issue_warp_per_cycle_smsp'] = kernel_res['issue_warp_per_cycle_smsp']
                kernel_res['warp_cpi'] = kernel_res['warp_cpi']
                kernel_res['tot_warps_'] = kernel_res['tot_warps_instructions_executed']
                kernel_res['warp_cpi'] = kernel_res['warp_cpi']
                kernel_res['sm_warp_inst_executed'] = kernel_res['sm_warps_instructions_executed']
                extra_flag = True
            except:
                pass
    if extra_flag:
        extra_list += ['ACPAO', 'grid_size', 'block_size', 'max_active_block_per_sm', 'allocted_block_per_sm', 'kernel_lat', 'tot_ipc', 'tot_cpi', 'active_block_per_cycle_sm', 'active_warp_per_cycle_smsp', 'issue_warp_per_cycle_smsp', 'warp_cpi','sm_warp_inst_executed']
    return sim_res

def process_hw(hw_res):
    for app_arg, app_res in hw_res.items():
        for i, kernel_res in enumerate(app_res):
            kernel_res['warp_inst_executed'] = kernel_res['inst_executed']
            kernel_res['achieved_occupancy'] = kernel_res['achieved_occupancy']
            kernel_res['gpu_active_cycles'] = kernel_res['active_cycles_sys']
            kernel_res['sm_active_cycles_sum'] = kernel_res['active_cycles']
            kernel_res['ipc'] = kernel_res['ipc']
            
            kernel_res['l1_hit_rate'] = kernel_res['tex_cache_hit_rate'] / 100
            kernel_res['l1_hit_rate_g'] = kernel_res['global_hit_rate'] / 100
            try:
                kernel_res['l1_hit_rate_ldg'] = kernel_res['global_hit_rate_ld'] / 100
                kernel_res['l1_hit_rate_stg'] = kernel_res['global_hit_rate_st'] / 100
            except:
                pass
            
            kernel_res['l2_hit_rate'] = kernel_res['l2_tex_hit_rate'] / 100
            kernel_res['l2_hit_rate_ld'] = kernel_res['l2_tex_read_hit_rate'] / 100
            kernel_res['l2_hit_rate_st'] = kernel_res['l2_tex_write_hit_rate'] / 100
        
            kernel_res['gmem_tot_reqs'] = kernel_res['global_load_requests'] + kernel_res['global_store_requests']
            kernel_res['gmem_ld_reqs'] = kernel_res['global_load_requests']
            kernel_res['gmem_st_reqs'] = kernel_res['global_store_requests']
            kernel_res['gmem_tot_sectors'] = kernel_res['gld_transactions'] + kernel_res['gst_transactions']
            
            kernel_res['gmem_diverg'] = kernel_res['gmem_tot_sectors'] / kernel_res['gmem_tot_reqs'] if kernel_res['gmem_tot_reqs'] != 0 else 0
            kernel_res['gmem_ld_sectors'] = kernel_res['gld_transactions']
            kernel_res['gmem_st_sectors'] = kernel_res['gst_transactions']
            try:
                kernel_res['gmem_ld_diverg'] = kernel_res['gld_transactions_per_request']
            except:
                kernel_res['gmem_ld_diverg'] = kernel_res['gld_transactions']/kernel_res['global_load_requests'] if kernel_res['global_load_requests'] != 0 else 0
            kernel_res['l2_tot_trans'] = kernel_res['l2_read_transactions'] + kernel_res['l2_write_transactions']
            kernel_res['l2_ld_trans'] = kernel_res['l2_read_transactions']
            kernel_res['l2_st_trans'] = kernel_res['l2_write_transactions']
            kernel_res['dram_tot_trans'] = kernel_res['dram_read_transactions'] + kernel_res['dram_write_transactions']
            kernel_res['dram_ld_trans'] = kernel_res['dram_read_transactions']
            kernel_res['dram_st_trans'] = kernel_res['dram_write_transactions']
    return hw_res

def json2df(json_data):
    '''
    list of dict 形式
    '''
    data = []
    for app_arg, app_res in json_data.items():
        bench = suite_info['map'][app_arg][0]
        for i, kernel_res in enumerate(app_res):
            kernel_res['bench'] = bench
            kernel_res['app'] = app_arg
            kernel_res['kernel_id'] = kernel_res['kernel_name']
            data.append(kernel_res)
    return pd.DataFrame(data, columns=["bench", "app", "kernel_id"] + cmp_list)

def json2df_extra(json_data):
    data = []
    for app_arg, app_res in json_data.items():
        bench = suite_info['map'][app_arg][0]
        for i, kernel_res in enumerate(app_res):
            kernel_res['bench'] = bench
            kernel_res['app'] = app_arg
            kernel_res['kernel_id'] = kernel_res['kernel_name']
            data.append(kernel_res)
    return pd.DataFrame(data, columns=["bench", "app", "kernel_id"] + extra_list)

def interleave(df_sim, df_hw):
    df_sim.rename(columns={f"{key}": f"{key}_sim" for key in cmp_list}, inplace=True)
    df_hw.rename(columns={f"{key}": f"{key}_hw" for key in cmp_list}, inplace=True)
    
    c1 = df_sim.columns[3:]
    c2 = df_hw.columns[3:]
    # interleaved colums
    waved = [c for zipped in zip(c1, c2) 
               for c in zipped]
    waved = list(df_sim.columns[:3]) + waved # insert common
    out = pd.concat([df_sim, df_hw.iloc[:,3:]], axis=1).reindex(columns=waved)
    return out
    
def get_summary(df):
    data = []
    for key in cmp_list:
        y1 = np.array(df[f"{key}_hw"].to_numpy())
        y2 = np.array(df[f"{key}_sim"].to_numpy())
        non_zero_idxs = y1 != 0
        MAE = np.mean(np.abs(y1[non_zero_idxs] - y2[non_zero_idxs])/y1[non_zero_idxs])
        NRMSE = np.sqrt(np.mean((y1 - y2)**2)) / np.mean(y1)
        corr = np.corrcoef(y1, y2)[0, 1]
        data.append({"MAE": MAE, "NRMSE": NRMSE, "corr": corr})
    df_summary = pd.DataFrame(data, index=cmp_list)
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
                        default="analysis_res.xlsx")
    parser.add_argument("-B", "--benchmark_list",
                        help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for the benchmark suite names.",
                        default="")
    parser.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
    parser.add_argument("--gpgpu-sim",action="store_true", help="use gpgpu-sim as sim res")
    args = parser.parse_args()


    from scripts.common import *
    from scripts.draw.draw_1 import filter_res,filter_hw_kernel,find_common

    apps = gen_apps_from_suite_list(args.benchmark_list)
    app_and_arg_list = get_app_arg_list(apps)
    app_arg_filtered_list = filter_app_list(app_and_arg_list, args.app_filter)

    with open(args.hw_res, 'r') as f:
        hw_res = json.load(f)

    with open(args.sim_res, 'r') as f:
        sim_res = json.load(f)

    sim_res = filter_res(sim_res, app_arg_filtered_list)
    hw_res = filter_res(hw_res, app_arg_filtered_list)
    hw_res = filter_hw_kernel(sim_res, hw_res)
    sim_res, hw_res = find_common(sim_res, hw_res)
    if args.gpgpu_sim:
        sim_res = process_gpgpu_sim(sim_res)
    else:
        sim_res = process_sim(sim_res)
    hw_res = process_hw(hw_res)
    
    df_sim = json2df(sim_res)
    df_sim_ex = json2df_extra(sim_res)
    df_hw = json2df(hw_res)
    df_kernels = interleave(df_sim, df_hw)
    
    def write_df_and_bench_summary(writer, df, prefix):
        start = 0
        df.to_excel(writer, sheet_name=f'{prefix}', index=False)
        start += len(df) + 2
        
        df_list = []
        keys = ["all"]
        df_list.append(get_summary(df))
        
        bench_groupby = df.groupby("bench", sort=False, as_index=False)
        for bench, group in bench_groupby:
            df_list.append(get_summary(group))
            keys.append(bench)
        
        df_summary = pd.concat(df_list, keys=keys, axis=1)
        df_summary.to_excel(writer, sheet_name=f'{prefix}_summary', index=True)
        
        # 计算每个指标 的 MAE，记录到 一个 MAE datasheet
        metric_maes = [df.iloc[:,:3]]
        metric_list = df.columns[3:]
        for i in range(len(metric_list)//2):
            metric = metric_list[2*i].replace("_sim", "")
            MAE = (df.iloc[:,3+2*i] - df.iloc[:,3+2*i+1])/df.iloc[:,3+2*i+1]
            MAE.name = metric
            metric_maes.append(MAE)
        
        df_metrics = pd.concat(metric_maes, axis=1)
        df_metrics.to_excel(writer, sheet_name=f'{prefix}_MAE', index=True)
        
    df_apps = df_kernels.groupby(["bench", "app"], sort=False, as_index=False).mean(numeric_only=True)
    df_apps.insert(2, "kernel_id", df_apps['app'])  # 补充空位
    df_apps_ex = df_sim_ex.groupby(["bench", "app"], sort=False, as_index=False).mean(numeric_only=True)
    
    with pd.ExcelWriter(args.output_file, engine='xlsxwriter') as writer:
        write_df_and_bench_summary(writer, df_apps, 'apps')
        write_df_and_bench_summary(writer, df_kernels, 'kernels')
        df_apps_ex.to_excel(writer, sheet_name='apps_extra', index=False)
        df_sim_ex.to_excel(writer, sheet_name='kernels_extra', index=False)
    
    print(f"Results saved to {args.output_file}")
    print("Done!")