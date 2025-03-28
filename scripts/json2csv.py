
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import csv


import sys, os
# curr_dir = os.path.dirname(__file__)
# par_dir = os.path.dirname(curr_dir)
# sys.path.insert(0, os.path.abspath(par_dir))
# from common import *

from enum import IntEnum

class JSON2CSV(IntEnum):
    NCU = 1
    PPT_GPU = 2
    BOTH = 3
    GENERAL = 4

def dump_to_csv_general(json_data, output_file='output.csv'):
    csvfile = open(output_file, 'w', newline='')
    csv_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    columns = ['app', 'kernel_id']
    for app_arg, app_res in json_data.items():
        for kernel_res in app_res:
            for key in kernel_res.keys():
                if key not in columns:
                    columns.append(key)
    csv_writer.writerow(columns)
    for app_arg, app_res in json_data.items():
        for i, kernel_res in enumerate(app_res):
            row = [app_arg, i+1]
            for col in columns[2:]:
                if type(kernel_res[col]) == list:
                    row.append("ListSkip")
                else:
                    row.append(kernel_res[col])
            csv_writer.writerow(row)
    
def dump_to_csv(json_data, output_file='output.csv', select=JSON2CSV.NCU):
    csvfile = open(output_file, 'w', newline='')
    csv_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    columns = ['app', 'kernel_name']
    # title
    if select == JSON2CSV.NCU:
        columns += ['gpc__cycles_elapsed.avg', 'gpc__cycles_elapsed.max', 'sys__cycles_active.sum']
    elif select == JSON2CSV.PPT_GPU:
        columns += ['my_gpu_act_cycles_max']
    csv_writer.writerow(columns)
        
    for app_arg, app_res in json_data.items():
        for kernel_res in app_res:
            row = [app_arg]
            for col in columns[1:]:
                row.append(kernel_res[col])
            csv_writer.writerow(row)
    
    csvfile.close()
    print(f"output to {output_file}")

def get_value(d, key_path):
    key_list = key_path.split('/')
    value = d
    for k in key_list:
        value = value[k]
    return value

def dump_to_csv_merge(json_data, json_data2, output_file='output.csv'):
    csvfile = open(output_file, 'w', newline='')
    csv_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    # title
    columns1 = ['app', 'kernel_name', 'gpc__cycles_elapsed.max']
    # columns1 = ['app', 'kernel_name', 'gpc__cycles_elapsed.avg', 'gpc__cycles_elapsed.max', 'sys__cycles_active.sum']
    columns2 = ['my_gpu_act_cycles_max', 'warp_inst_executed', 'grid_size', 'block_size', 'AMAT', 'result/ours_smsp_min', 'result/ours_smsp_avg', 'result/ours_smsp_max', 'result/ours_smsp_avg_tail_LI', 'kernel_detail/kernel_lat', 'kernel_detail/last_inst', 'kernel_detail/tail', 'PPT-GPU_min', 'PPT-GPU_max', 'kernel_detail/comp_cycles_scale']
    csv_writer.writerow(columns1 + columns2)
        
    for app_arg, app_res in json_data.items():
        for kid, kernel_res in enumerate(app_res):
            row = [app_arg]
            for col in columns1[1:]:
                row.append(kernel_res[col])
            # get second json data kernel result
            kernel_res2 = json_data2[app_arg][kid]
            for col in columns2:
                row.append(get_value(kernel_res2,col))
            csv_writer.writerow(row)
    
    csvfile.close()
    print(f"output to {output_file}")

def filter_res(res, app_arg_filtered_list):
    for app in res.copy():
        if app not in app_arg_filtered_list:
            del res[app]
    return res
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script draw CPI stack image, it support: 1) draw single or, side by side campare with second result'
    )
    parser.add_argument("-H", "--json-file",
                        help="ncu hardware result json file path")
    parser.add_argument("-S", "--json-file2",
                        help="ppt-gpu sim result json file path")
    parser.add_argument("-t", "--type",
                        default=JSON2CSV.BOTH,
                        # choices=[JSON2CSV.NCU, JSON2CSV.PPT_GPU, JSON2CSV.BOTH],
                        type=int,
                        help="json file content type")
    parser.add_argument("-o", "--output",
                        default="output.csv",
                        help="output csv file")
    parser.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
    
    args = parser.parse_args()
    
    from common import *
    apps = gen_apps_from_suite_list()
    app_and_arg_list = get_app_arg_list(apps)
    app_arg_filtered_list = filter_app_list(app_and_arg_list, args.app_filter)
    print(f"app_arg_filtered_list: {app_arg_filtered_list}")
    
    if args.type == JSON2CSV.BOTH:
        with open(args.json_file, 'r') as f:
            json_data = json.load(f)
        
        with open(args.json_file2, 'r') as f:
            json_data2 = json.load(f)

        json_data = filter_res(json_data, app_arg_filtered_list)
        json_data2 = filter_res(json_data2, app_arg_filtered_list)
        
        dump_to_csv_merge(json_data, json_data2, output_file=args.output)
    elif args.type == JSON2CSV.NCU:
        with open(args.json_file, 'r') as f:
            json_data = json.load(f)
        json_data = filter_res(json_data, app_arg_filtered_list)
        dump_to_csv(json_data, output_file=args.output, select=JSON2CSV.NCU)
    elif args.type == JSON2CSV.GENERAL:
        with open(args.json_file, 'r') as f:
            json_data = json.load(f)
        json_data = filter_res(json_data, app_arg_filtered_list)
        dump_to_csv_general(json_data, output_file=args.output)