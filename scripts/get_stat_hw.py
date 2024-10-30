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
parser.add_argument("--select", default="nvprof",
                 choices=["nvprof", "ncu", "ncu-cpi", "nvprof-cpi"],
                 help="get which tool's stat")
parser.add_argument("--loop",
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

def parse_csv_file(profiling_file):
    '''
    将 csv 每一行转换成一个 dict，其中数字被尝试转换成 float，字符串不变
    返回 dict 列表
    '''
    result = []
    # with open(profiling_file, 'r') as f:
    #     lines = iter(f.readlines())
    #     for line in lines:
    #         # skip header
    #         if line.startswith('='):
    #             continue
    #         keys_list = line.split(',')
    #         keys_list = [key.strip().replace('"', '') for key in keys_list]
    #         next(lines) # skip ,,,, line
    #         break
    #     # to the end
    #     for line in lines:
    #         values_list = line.split(',')
    #         kernel_data = {}
    #         for key, value in zip(keys_list, values_list):
    #             try:
    #                 # digit
    #                 v = float(value)
    #             except:
    #                 # string
    #                 v = value.replace('"', '')
    #             kernel_data[key] = v
    #         result.append(kernel_data)
    # 需要手动处理 "" 括起来的字符串，不能被split。于是转用 csv 库
    import csv
    with open(profiling_file, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            if row[0].startswith('='):
                continue
            keys_list = row
            # next(csvreader)
            break
        for row in csvreader:
            kernel_data = {}
            for key, value in zip(keys_list, row):
                # m = re.match(r'[\d,.]+', value)
                # if m:
                #     print(value)
                if ',' in value:
                    value = value.replace(',', '')
                try:
                    # digit
                    value = float(value)
                except:
                    pass
                kernel_data[key] = value
            result.append(kernel_data)
    return result

def get_average_csv(profiling_file_list):
    '''
    输入若干 csv 文件，返回平均后的字典列表
    '''
    res_list = []
    for profiling_file in profiling_file_list:
        res_list.append(parse_csv_file(profiling_file))
        
    loop_num = len(res_list)
    
    # sum all res
    acc_res = res_list[0]
    for res in res_list[1:]:
        for i,kernel_data in enumerate(res):  # 第 i 个 kernel
            for key, value in kernel_data.items():
                if type(value) == str:
                    continue
                acc_res[i][key] += value
    # average
    for i,kernel_data in enumerate(acc_res):
        for key, value in kernel_data.items():
            if type(value) == str:
                continue
            acc_res[i][key] /= loop_num

    return acc_res

collect_data = {}

print("Start get hw result")
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
    
    # get all profling file
    profiling_res = {}
    for file in os.listdir(app_trace_dir):
        if file.startswith('profiling.') and file.endswith('.csv'):
            _, select , cnt, _ = file.split('.')
            if select not in profiling_res:
                profiling_res[select] = []
            profiling_res[select].append(os.path.join(app_trace_dir, file))
    
    # sort and only use pre loop cnt result
    for select, profiling_file_list in profiling_res.items():
        profiling_file_list.sort()
        if args.loop != -1:
            profiling_file_list = profiling_file_list[:args.loop]
    
    if 'nvprof' in args.select:
        acc_res = get_average_csv(profiling_res[args.select])
        
        # skip first empty kernel line
        acc_res = acc_res[1:]
        # nvprof: fix kernel_name
        for i,kernel_data in enumerate(acc_res):
            acc_res[i]['kernel_name'] = re.search(r'\w+', acc_res[i]['Kernel']).group(0)  # delete function params define
            del acc_res[i]['Kernel']
    else:
        acc_res = get_average_csv(profiling_res[args.select])

        # ncu 获得的结果一行不是 kernel，而是一个 metric
        # 需要"转置"一下，获得 kernel 关于 metric 的结果
        kernel_data_dict= {}
        for i,row_data in enumerate(acc_res):
            id = row_data['ID']
            metric_name = row_data['Metric Name']
            metric_value = row_data['Metric Value']
            kernel_name = re.search(r'\w+', row_data['Kernel Name']).group(0)
            
            if id not in kernel_data_dict:
                kernel_data_dict[id] = {}
                kernel_data_dict[id]['kernel_name'] = kernel_name
            # add new metric
            kernel_data_dict[id]['ID'] = id
            kernel_data_dict[id][metric_name] = metric_value
        acc_res = list(kernel_data_dict.values())
    
    # limit kernel number
    acc_res = acc_res[:args.limit_kernel_num]
    
    print(f"{app_and_arg}: {len(acc_res)}")
    collect_data[app_and_arg] = acc_res
    
    with open(args.output, 'w') as f:
        json.dump(collect_data, f, indent=4)
