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

app_and_arg_list = []
for app in apps:
    exec_dir, data_dir, exe_name, args_list = app
    for argpair in args_list:
        mem_usage = argpair["accel-sim-mem"]
        app_and_arg_list.append(os.path.join( exe_name, get_argfoldername( argpair["args"] ) ))  # backprop-rodinia-2.0-ft/4096___data_result_4096_txt

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
    
    # convert to float
    for k,v in kernel_res.items():
        if k!="kernel_name":
            if ',' in v:
                v = v.replace(',', '')
            kernel_res[k] = float(v)
    return kernel_res

collect_data = {}

for app_and_arg in app_and_arg_list:
    app = app_and_arg.split('/')[0]
    app_trace_dir = os.path.join(args.trace_dir, app_and_arg)
    
    if not os.path.exists(app_trace_dir):
        print(f"{app_and_arg} not found")
        continue
    # get all sim log file
    file_list = []
    for file in os.listdir(app_trace_dir):
        m = re.search(r'kernel\_(?P<kernel_id>\d+)\_(?P<type>SASS|PTX)\_.*\.out', file)
        if m:
            file_list.append( (int(m.group('kernel_id')), os.path.join(app_trace_dir, file)) ) 

    app_res = [ {} for i in range(len(file_list)) ]
    try:
        for kernel_id, file_path in file_list:
            app_res[kernel_id-1] = parse_kernel_log(file_path)
    except:
        print(f"Error in {app_and_arg}")
        print(f"{kernel_id} {file_path}")
        exit(1)
    
    print(f"{app_and_arg}: {len(app_res)}")
    collect_data[app_and_arg] = app_res

with open(args.output, 'w') as f:
    json.dump(collect_data, f, indent=4)