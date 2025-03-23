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
                default="")
parser.add_argument("--apps",
                    nargs="*",
                    help="only update specific apps data")
parser.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
parser.add_argument("-T", "--trace_dir",
                    required=True,
                    help="The root of all the trace file")
# parser.add_argument("-f", "--profiling_filename",
#                  default="profiling.csv")
parser.add_argument("-o", "--output",
                 default="hw_res.json")
parser.add_argument("-c", "--limit_kernel_num",
                    type=int,
                    default=300,
                    help="trace tool only trace max 300 kernel, nvprof can't set trace limit(nsight-sys can)." \
                        "so we limit the kernel number when get stat")
parser.add_argument('-l', "--loop-cnt",
                 default=-1,
                 type=int,
                 help="limit only use specific loop result")
args = parser.parse_args()

from common import *

# defined_apps = {}
# parse_app_definition_yaml(args.benchmarks_yaml, defined_apps)
apps = gen_apps_from_suite_list(args.benchmark_list)
app_and_arg_list = get_app_arg_list(apps)
# args.apps = process_args_apps(args.apps, defined_apps)
args.apps = filter_app_list(app_and_arg_list, args.app_filter)

def parse_gpgpu_sim_log_file(profiling_file, skip_unit_row=False):
    '''
    返回 dict 列表
    '''
    result = []
    with open(profiling_file, 'r') as f:
        log = f.read()
    
    kernel_name_list = re.findall(r'kernel_name = (?P<value>.*) $', log, re.MULTILINE)
    gpu_cycle = list(map(int, re.findall(r'^gpu_sim_cycle = (?P<value>\d+)', log, re.MULTILINE)))
    gpu_inst = list(map(int, re.findall(r'^gpu_sim_insn = (?P<value>\d+)', log, re.MULTILINE)))
    gpu_ipc = list(map(float, re.findall(r'^gpu_ipc = \s*(?P<value>[\d\.]+)', log, re.MULTILINE)))
    gpu_occupancy = list(map(float, re.findall(r'^gpu_occupancy = (?P<value>[\d\.]+)\%', log, re.MULTILINE)))
    gpgpu_simulation_time_acc = list(map(int, re.findall(r'^gpgpu_simulation_time = .*\((?P<value>\d+) sec\)', log, re.MULTILINE)))
    gpgpu_simulation_time = [gpgpu_simulation_time_acc[0]] + [gpgpu_simulation_time_acc[i] - gpgpu_simulation_time_acc[i - 1] for i in range(1, len(gpgpu_simulation_time_acc))]
    L1D_total_cache_miss_rate = 0.2930
    l1_miss_rate = list(map(float, re.findall(r'L1D_total_cache_miss_rate = \s*(?P<value>[\d\.]+)', log, re.MULTILINE)))
    l2_miss_rate = list(map(float, re.findall(r'L2_total_cache_miss_rate = \s*(?P<value>[\d\.]+)', log, re.MULTILINE)))
    l1_hit_rate =[1 - x for x in l1_miss_rate]
    l2_hit_rate =[1 - x for x in l2_miss_rate]
    
    dram_read = list(map(int, re.findall(r'^total dram reads = (?P<value>\d+)', log, re.MULTILINE)))
    dram_write = list(map(int, re.findall(r'^total dram writes = (?P<value>\d+)', log, re.MULTILINE)))
    dram_tot = [x + y for x, y in zip(dram_read, dram_write)]
    
    kernel_num = len(gpu_cycle)
    for i in range(kernel_num):
        kernel_data = {
            'kernel_name': kernel_name_list[i],
            'kernel_id': i + 1,
            'gpu_cycle': gpu_cycle[i],
            'gpu_inst': gpu_inst[i],
            'gpu_ipc': gpu_ipc[i],
            'gpu_occupancy': gpu_occupancy[i],
            # memory
            'l1_hit_rate': l1_hit_rate[i],
            'l2_hit_rate': l2_hit_rate[i],
            'dram_tot_trans': dram_tot[i],
            'dram_ld_trans': dram_read[i],
            'dram_st_trans': dram_write[i],
            # sim time
            'gpgpu_simulation_time': gpgpu_simulation_time[i],
        }
        result.append(kernel_data)
    return result

collect_data = {}

print("Start get gpgpu sim res")
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
    
    gpgpu_sim_log_path = os.path.join(app_trace_dir, f'gpgpu-sim.{app}.0.log')
    app_res = parse_gpgpu_sim_log_file(gpgpu_sim_log_path)
    
    print(f"{app_and_arg}: {len(app_res)}")
    collect_data[app_and_arg] = app_res
    
    with open(args.output, 'w') as f:
        json.dump(collect_data, f, indent=4)
