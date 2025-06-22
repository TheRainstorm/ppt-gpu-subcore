import argparse

import json
import subprocess
import shlex
import re
import os
import yaml
import hashlib
from datetime import datetime

def get_app_config(app_path):
    app_config_path = os.path.join(app_path, "app_config.py")
    app_config = {}
    with open(app_config_path, "r") as file:
        exec(file.read(), app_config)
    return app_config

parser = argparse.ArgumentParser(
    description='Simulate all app defined'
)
parser.add_argument("-B", "--benchmark_list",
                 help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for the benchmark suite names.",
                 default="")
parser.add_argument("--apps",
                    nargs="*",
                    help="only update specific apps data")
parser.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
parser.add_argument("-T", "--trace_dir",
                    required=True,
                    help="The root of all the trace file")
parser.add_argument("-R", "--report_dir",
                    required=True,
                    help="The root of all the output file")
parser.add_argument("--not-full", dest="full",
                 action="store_false",
                 help="get full sim result")
parser.add_argument("-o", "--output",
                 default="sim_res.json")
parser.add_argument("-C", "--app_config_cache",
                 default="kernels.json",
                 help='cache app_config.py database')
args = parser.parse_args()

from common import *

# defined_apps = {}
# parse_app_definition_yaml(args.benchmarks_yaml, defined_apps)
apps = gen_apps_from_suite_list(args.benchmark_list)
app_and_arg_list = get_app_arg_list(apps)
# args.apps = process_args_apps(args.apps, defined_apps)
args.apps = filter_app_list(app_and_arg_list, args.app_filter)

def parse_kernel_log(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    
    kernel_res = {
        "kernel_name": re.search(r'kernel name: (\w+)', data).group(1),
        # occupancy
        "achieved_occupancy": re.search(r'achieved occupancy: (\d+(\.\d+)?) %', data).group(1),
        
        "gpu_active_cycle_min": re.search(r'\* GPU active cycles \(min\): ([\d,]+)', data).group(1),  # 6,954,480 not 6954480
        "gpu_active_cycle_max": re.search(r'\* GPU active cycles \(max\): ([\d,]+)', data).group(1),
        "sm_active_cycles_sum": re.search(r'\* SM active cycles \(sum\): ([\d,]+)', data).group(1),
        "sm_elapsed_cycles_sum": re.search(r'\* SM elapsed cycles \(sum\): ([\d,]+)', data).group(1),
        "my_gpu_active_cycle_min": re.search(r'My GPU active cycles \(min\): ([\d,]+)', data).group(1),
        "my_gpu_active_cycle_max": re.search(r'My GPU active cycles \(max\): ([\d,]+)', data).group(1),
        "my_sm_active_cycles_sum": re.search(r'My SM active cycles \(sum\): ([\d,]+)', data).group(1),
        "my_sm_elapsed_cycles_sum": re.search(r'My SM elapsed cycles \(sum\): ([\d,]+)', data).group(1),
        
        "warp_inst_executed": re.search(r'Warp instructions executed: ([\d,]+)', data).group(1),
        "ipc": re.search(r'- Instructions executed per clock cycle \(IPC\): (\d+(\.\d*)?)', data).group(1),
        "my_ipc": re.search(r'My Instructions executed per clock cycle \(IPC\): (\d+(\.\d*)?)', data).group(1),
        "kernel_exec_time_us": re.search(r'Kernel execution time: (\d+((\.\d*)?)?) us', data).group(1),
        
        # memory
        "AMAT": re.search(r'AMAT: (\d+)', data).group(1),
        "ACPAO": re.search(r'ACPAO: (\d+)', data).group(1),
        
        "l1_hit_rate": re.search(r'unified L1 cache hit rate: (\d+(\.\d+)?) %', data).group(1),
        "l2_hit_rate": re.search(r'L2 cache hit rate: (\d+(\.\d+)?) %', data).group(1),
        "gmem_read_requests": re.search(r'GMEM read requests: (\d+)', data).group(1),
        "gmem_write_requests": re.search(r'GMEM write requests: (\d+)', data).group(1),
        "gmem_read_trans": re.search(r'GMEM read transactions: (\d+)', data).group(1),
        "gmem_write_trans": re.search(r'GMEM write transactions: (\d+)', data).group(1),
        "l2_read_trans": re.search(r'L2 read transactions: (\d+)', data).group(1),
        "l2_write_trans": re.search(r'L2 write transactions: (\d+)', data).group(1),
        "dram_total_trans": re.search(r'DRAM total transactions: (-?\d+)', data).group(1),
        
        # other
        "sim_time_mem": re.search(r'Memory model: (\d+(\.\d*)?) sec', data).group(1),
        "sim_time_comp": re.search(r'Compute model: (\d+(\.\d*)?) sec', data).group(1),
    }
    
    debug_data = {}
    try:
        debug_data = {
            "diverge_flag": re.search(r'diverge_flag: (\d+)', data).group(1),
            "l2_parallelism": re.search(r'l2_parallelism: (\d+)', data).group(1),
            "dram_parallelism": re.search(r'dram_parallelism: (\d+)', data).group(1),
            "l1_cycles_no_contention": re.search(r'l1_cycles_no_contention: (\d+(\.\d*)?)', data).group(1),
            "l2_cycles_no_contention": re.search(r'l2_cycles_no_contention: (\d+(\.\d*)?)', data).group(1),
            "dram_cycles_no_contention": re.search(r'dram_cycles_no_contention: (\d+(\.\d*)?)', data).group(1),
            "dram_queuing_delay_cycles": re.search(r'dram_queuing_delay_cycles: (\d+(\.\d*)?)', data).group(1),
            "noc_queueing_delay_cycles": re.search(r'noc_queueing_delay_cycles: (\d+(\.\d*)?)', data).group(1),
            "mem_cycles_no_contention": re.search(r'mem_cycles_no_contention: (\d+)', data).group(1),
            "mem_cycles_ovhds": re.search(r'mem_cycles_ovhds: (\d+)', data).group(1),
        }
    except:
        pass
    
    kernel_res.update(debug_data)
    
    json_str = re.search(r'CPI\ stack:\ (\{.*\})', data, re.DOTALL).group(1)
    kernel_res['cpi_stack'] = json.loads(json_str)
    
    # convert to float
    for k,v in kernel_res.items():
        if k=="kernel_name":
            continue
        if type(v) == dict:
            continue
        if ',' in v:
            v = v.replace(',', '')
        kernel_res[k] = float(v)
    return kernel_res

def parse_kernel_json(file_path, full=True):
    '''
    定义了模型至少需要输出哪些数据
    '''
    with open(file_path, 'r') as f:
        data_json = json.load(f)
    
    if full:
        data_json_new = data_json
    else:
        data_json_new = {}
    
    # 基础
    data_json_new['warp_inst_executed'] = data_json['tot_warps_instructions_executed']
    data_json_new["achieved_occupancy"] = data_json["achieved_occupancy"]
    
    # 预测的 cycle，有两个，一个 min 一个 max，默认使用 max 作为最后结果
    data_json_new["gpu_active_cycle_min"] = data_json["gpu_act_cycles_min"]  # 单个 SM cycle 最小值
    data_json_new["gpu_active_cycle_max"] = data_json["gpu_act_cycles_max"]  # 单个 SM cycle 最大值
    data_json_new["gpu_active_cycles"] = data_json["gpu_act_cycles_max"]  # 单个 SM cycle
    # SM 累加的 cycle，有 active 和 elapse 两个
    data_json_new["sm_active_cycles_sum"] = data_json["sm_act_cycles.sum"]  # 所有 SM cycle 总和
    data_json_new["sm_elapsed_cycles_sum"] = data_json["sm_elp_cycles.sum"] # 所有 SM elapsed cycle 总和（和 active 区别：active 排除了非活跃的 SM）
    # ipc
    data_json_new["ipc"] = data_json["tot_ipc"]  # SM 平均 IPC
    
    # 针对我的模型，尝试不同选择不同 cycle
    ## select other result
    # result = data_json["result"]
    # kernel_detail = data_json["kernel_detail"]
    # kernel_lat = kernel_detail['kernel_lat']
    # data_json_new["my_gpu_active_cycle_max"] = result['ours_smsp_max'] + kernel_lat
    # data_json_new["my_gpu_active_cycle_max"] = result['ours_smsp_avg'] + kernel_lat
    # data_json_new["my_gpu_active_cycle_max"] = result['ours_smsp_avg_tail'] + kernel_lat
    # data_json_new["my_gpu_active_cycle_max"] = result['ours_smsp_avg_LI'] + kernel_lat
    # data_json_new["my_gpu_active_cycle_max"] = result['ours_smsp_avg_tail_LI'] + kernel_lat
    # data_json_new["my_gpu_active_cycle_max"] = result['ours_smsp_avg_scale2'] + kernel_lat
    # data_json_new["my_gpu_active_cycle_max"] = result['ours_smsp_avg_tail_scale2'] + kernel_lat
    # data_json_new["my_gpu_active_cycle_max"] = result['ours_smsp_avg_tail_scale2_LI'] + kernel_lat

    # memory
    if 'umem_hit_rate' not in data_json["memory_stats"]:
        # compati to ppt-gpu (get_stat_sim)
        data_json["memory_stats"]['umem_hit_rate'] = data_json["memory_stats"]['l1_hit_rate']
        data_json["memory_stats"]['hit_rate_l2'] = data_json["memory_stats"]['l2_hit_rate']
        data_json["memory_stats"]['gmem_ld_trans'] = data_json["memory_stats"]['gmem_ld_sectors']
        data_json["memory_stats"]['gmem_st_trans'] = data_json["memory_stats"]['gmem_st_sectors']
        data_json["memory_stats"]['gmem_tot_trans'] = data_json["memory_stats"]['gmem_tot_sectors']
        data_json["memory_stats"]['l2_ld_trans_gmem'] = data_json["memory_stats"]['l2_ld_trans']
        data_json["memory_stats"]['l2_st_trans_gmem'] = data_json["memory_stats"]['l2_st_trans']
        data_json["memory_stats"]['l2_tot_trans_gmem'] = data_json["memory_stats"]['l2_tot_trans']
        data_json["memory_stats"]['dram_tot_trans_gmem'] = data_json["memory_stats"]['dram_tot_trans']
    
    data_json_new["l1_hit_rate"] = data_json["memory_stats"]["umem_hit_rate"]*100
    data_json_new["l2_hit_rate"] = data_json["memory_stats"]["hit_rate_l2"]*100
    data_json_new["gmem_read_requests"] = data_json["memory_stats"]["gmem_ld_reqs"]
    data_json_new["gmem_write_requests"] = data_json["memory_stats"]["gmem_st_reqs"]
    data_json_new["gmem_read_trans"] = data_json["memory_stats"]["gmem_ld_trans"]
    data_json_new["gmem_write_trans"] = data_json["memory_stats"]["gmem_st_trans"]
    data_json_new["l2_read_trans"] = data_json["memory_stats"]["l2_ld_trans_gmem"]
    data_json_new["l2_write_trans"] = data_json["memory_stats"]["l2_st_trans_gmem"]
    data_json_new["dram_total_trans"] = data_json["memory_stats"]["dram_tot_trans_gmem"]
    
    data_json_new["gmem_tot_reqs"] = data_json["memory_stats"]["gmem_tot_reqs"]
    data_json_new["gmem_tot_sectors"] = data_json["memory_stats"]["gmem_tot_trans"]
    data_json_new['l2_tot_trans'] = data_json['memory_stats']['l2_tot_trans_gmem']
        
    data_json_new["AMAT"] = data_json["AMAT"]
    
    # CPI info
    # "warp_stats/warp_stats"
    # scheduler_stats/stall_types
    
    return data_json_new

print("Start get sim result")
collect_data = {}
if os.path.exists(args.output):
    if args.app_filter != args.benchmark_list: # 只有少数 app 时，读取旧数据
        with open(args.output, 'r') as f: # merge old data
            print(f"load old data: {args.output}")
            collect_data = json.load(f)
    # os.rename(args.output, args.output + '.bak')

app_config_cache = {}
if os.path.exists(args.app_config_cache):
    with open(args.app_config_cache, 'r') as f:
        print(f"load app_config_cache: {args.app_config_cache}")
        app_config_cache = json.load(f)

for app_and_arg in app_and_arg_list:
    app = app_and_arg.split('/')[0]
    app_report_dir = os.path.join(args.report_dir, app_and_arg)
    app_trace_dir = os.path.join(args.trace_dir, app_and_arg)
    
    if args.apps and app_and_arg not in args.apps:
        continue
    
    if not os.path.exists(app_report_dir):
        print(f"{app_and_arg} not found")
        continue
    # get all sim log file
    # file_list = []
    # for file in os.listdir(app_report_dir):
    #     m = re.search(r'kernel\_(?P<kernel_id>\d+)\_pred_out.json', file)
    #     if m:
    #         file_list.append( (int(m.group('kernel_id')), os.path.join(app_report_dir, file)) ) 
    ## sort by kernel_id
    # file_list.sort(key=lambda x: x[0])

    # 获得 app config 中的 kernel 数
    if app_and_arg in app_config_cache:
        app_kernels_id = app_config_cache[app_and_arg]
    else:
        app_config = get_app_config(app_trace_dir)
        app_kernels_id = app_config['app_kernels_id']
        app_config_cache[app_and_arg] = app_kernels_id
    
    app_res = []
    success = True
    try:
        for kernel_id in app_kernels_id:
            file_path = os.path.join(app_report_dir, f'kernel_{kernel_id}_pred_out.json')
            if not os.path.exists(file_path):
                print(f"{app_and_arg} kernel {kernel_id} not found")
                if app_and_arg in collect_data:
                    print(f"remove {app_and_arg} from collect_data")
                    del collect_data[app_and_arg]
                    # with open(args.output, 'w') as f:
                    #     json.dump(collect_data, f, indent=4)
                success = False
                break
            # k_res = parse_kernel_log(file_path)
            k_res = parse_kernel_json(file_path, args.full)
            k_res['kernel_id'] = kernel_id  # must keep kernel_id(1-based)
            app_res.append(k_res)
    except Exception as e:
        print(f"==========\nError in {app_and_arg}")
        print(f"{kernel_id} {file_path}")
        print(e)
        print("==========")
        continue
    
    print(f"{app_and_arg}: {len(app_res)}")
    if success:
        collect_data[app_and_arg] = app_res

# make dir
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, 'w') as f:
    json.dump(collect_data, f, indent=4)

with open(args.app_config_cache, 'w') as f:
    json.dump(app_config_cache, f, indent=4)
