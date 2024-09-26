import argparse

import json
import subprocess
import shlex
import re
import os
import yaml
import hashlib
from datetime import datetime

parser = argparse.ArgumentParser(
    description='Simulate all app defined'
)
parser.add_argument("-B", "--benchmark_list",
                 help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for the benchmark suite names.",
                 default="rodinia_2.0-ft")
parser.add_argument("--apps",
                    nargs="*",
                    help="only update specific apps data")
parser.add_argument("-Y", "--benchmarks_yaml",
                    required=True,
                    help='benchmarks_yaml path')
parser.add_argument("-T", "--trace_dir",
                    required=True,
                    help="The root of all the trace file")
parser.add_argument("-o", "--output",
                 default="sim_res.json")
args = parser.parse_args()

from common import *

defined_apps = {}
parse_app_definition_yaml(args.benchmarks_yaml, defined_apps)
apps = gen_apps_from_suite_list(args.benchmark_list.split(","), defined_apps)
args.apps = process_args_apps(args.apps, defined_apps)
app_and_arg_list = get_app_arg_list(apps)

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

def parse_kernel_json(file_path):
    with open(file_path, 'r') as f:
        data_json = json.load(f)
    
    # rename
    # data_json["achieved_occupancy"] = data_json["placeholder"]
    data_json['warp_inst_executed'] = data_json['tot_warps_instructions_executed']
    # data_json["gpu_active_cycle_min"] = data_json["placeholder"]
    # data_json["gpu_active_cycle_max"] = data_json["placeholder"]
    data_json["sm_active_cycles_sum"] = data_json["sm_act_cycles.sum"]
    data_json["sm_elapsed_cycles_sum"] = data_json["sm_elp_cycles.sum"]
    data_json["my_gpu_active_cycle_max"] = data_json["my_gpu_act_cycles_max"]
    data_json["my_sm_active_cycles_sum"] = data_json["my_sm_act_cycles.sum"]
    data_json["my_sm_elapsed_cycles_sum"] = data_json["my_sm_elp_cycles.sum"]
    data_json["ipc"] = data_json["my_tot_ipc"]
    data_json["l1_hit_rate"] = data_json["memory_stats"]["umem_hit_rate"]*100
    data_json["l2_hit_rate"] = data_json["memory_stats"]["hit_rate_l2"]*100
    data_json["gmem_read_requests"] = data_json["memory_stats"]["gmem_ld_reqs"]
    data_json["gmem_write_requests"] = data_json["memory_stats"]["gmem_st_reqs"]
    data_json["gmem_read_trans"] = data_json["memory_stats"]["gmem_ld_trans"]
    data_json["gmem_write_trans"] = data_json["memory_stats"]["gmem_st_trans"]
    data_json["l2_read_trans"] = data_json["memory_stats"]["l2_ld_trans_gmem"]
    data_json["l2_write_trans"] = data_json["memory_stats"]["l2_st_trans_gmem"]
    data_json["dram_total_trans"] = data_json["memory_stats"]["dram_tot_trans_gmem"]
    
    return data_json

collect_data = {}
# when get single app, load old data
if args.apps:
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            collect_data = json.load(f)

for app_and_arg in app_and_arg_list:
    app = app_and_arg.split('/')[0]
    app_trace_dir = os.path.join(args.trace_dir, app_and_arg)
    
    if args.apps and app_and_arg not in args.apps:
        continue
    
    if not os.path.exists(app_trace_dir):
        print(f"{app_and_arg} not found")
        continue
    # get all sim log file
    # file_list = []
    # for file in os.listdir(app_trace_dir):
    #     m = re.search(r'kernel\_(?P<kernel_id>\d+)\_(?P<type>SASS|PTX)\_.*\.out', file)
    #     if m:
    #         file_list.append( (int(m.group('kernel_id')), os.path.join(app_trace_dir, file)) ) 
    file_list = []
    for file in os.listdir(app_trace_dir):
        m = re.search(r'kernel\_(?P<kernel_id>\d+)\_pred_out.json', file)
        if m:
            file_list.append( (int(m.group('kernel_id')), os.path.join(app_trace_dir, file)) ) 

    app_res = [ {} for i in range(len(file_list)) ]
    try:
        for kernel_id, file_path in file_list:
            app_res[kernel_id-1] = parse_kernel_json(file_path)
    except Exception as e:
        print(f"Error in {app_and_arg}")
        print(f"{kernel_id}/{len(file_list)} {file_path}")
        print(e)
        exit(1)
    
    print(f"{app_and_arg}: {len(app_res)}")
    collect_data[app_and_arg] = app_res

    with open(args.output, 'w') as f:
        json.dump(collect_data, f, indent=4)
