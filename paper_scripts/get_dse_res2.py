
import json
import os
import pandas as pd
from openpyxl import load_workbook
import argparse
from scripts.common import *

parser = argparse.ArgumentParser()
parser.add_argument("-N", "--DSE_param_name", default='SM')
parser.add_argument("-P", "--DSE_prefix", default='/staff/fyyuan/repo/PPT-GPU/tmp_output/DSE_SM/sm_')
parser.add_argument("--param_list", type=int, nargs="+")
# parser.add_argument("--app_and_arg", default='b+tree-rodinia-3.1/file___data_mil_txt_command___data_command_txt')
parser.add_argument("--kernel_id", type=int, default=1)
parser.add_argument("--output_prefix", default='')
parser.add_argument("-B", "--benchmark_list",
                 help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for the benchmark suite names.",
                default="")
parser.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
args = parser.parse_args()

apps = gen_apps_from_suite_list(args.benchmark_list)
app_and_arg_list = get_app_arg_list(apps)
args.apps = filter_app_list(app_and_arg_list, args.app_filter)

# param_list=[int(x) for x in args.param_list_str[1:-1].split()]
param_list=args.param_list

DSE_prefix=args.DSE_prefix
# app_and_arg=args.app_and_arg
kernel_id=args.kernel_id
DSE_param_name=args.DSE_param_name

collect_res = []
for app_and_arg in app_and_arg_list:
    app=app_and_arg.split('/')[0]

    for x in param_list:
        app_report_dir=f'{DSE_prefix}{x}/{app_and_arg}'
        file_path = os.path.join(app_report_dir, f'kernel_{kernel_id}_pred_out.json')
        if not os.path.exists(file_path):
            print(f"File {file_path} not found")
            continue
        with open(file_path, 'r') as f:
            data_json = json.load(f)
        cycle = data_json['gpu_act_cycles_max']
        collect_res.append({
            'app_full': app_and_arg,
            'app': app,
            'cycle': cycle,
            'DSE_param_name': DSE_param_name,
            'DSE_param': x,
            'kernel_id': kernel_id,
        })

    print(collect_res)
    df = pd.DataFrame(collect_res, columns=["app", "app_full", "kernel_id", "DSE_param_name", 'DSE_param', 'cycle'])

if args.output_prefix:
    output_xlsx=f'{args.output_prefix}{DSE_param_name}.xlsx'
else:
    output_xlsx=f'DSE_res_{DSE_param_name}.xlsx'
with pd.ExcelWriter(output_xlsx, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name=f'raw', index=True)
print(f"Write to {output_xlsx}")