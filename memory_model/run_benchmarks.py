import argparse

import json
import os
import sys

from memory_model_warper import memory_model_warpper
from scripts.common import *

parser = argparse.ArgumentParser(
    description=''
)
parser.add_argument('-M', "--model",
                    # choices=['ppt-gpu', 'sdcm'],
                    default='ppt-gpu',
                    help='change memory model, check memory_model_warper.py for available models')
parser.add_argument('-c', "--config",
                    required=True,
                    help='target GPU hardware configuration')
parser.add_argument("-B", "--benchmark_list",
                    help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for the benchmark suite names.",
                    default="")
parser.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
parser.add_argument("-T", "--trace_dir",
                    required=True,
                    help="The root of all the trace file")
parser.add_argument("-o", "--output",
                    default="memory_res.json")
parser.add_argument("-l", "--log_file",
                    default="run_memory_model.log")
parser.add_argument("--granularity",
                    type=int,
                    default=2,
                    help='1=One Thread Block per SM or 2=Active Thread Blocks per SM or 3=All Thread Blocks per SM')
parser.add_argument('--use-approx', 
                    action='store_true',
                    help='sdcm use approx')
parser.add_argument('--filter-l2', 
                    action='store_true',
                    help='L1 hit bypass L2')
args = parser.parse_args()

apps = gen_apps_from_suite_list(args.benchmark_list)
app_and_arg_list = get_app_arg_list(apps)
args.apps = filter_app_list(app_and_arg_list, args.app_filter)

log_file = open(args.log_file, "a")
def logging(*args, **kwargs):
    args = (f"{now_timestamp()}: ", ) + args
    print(*args, **kwargs, file=log_file, flush=True)
    print(*args, **kwargs, file=sys.stderr)

failed_list = []
logging(f"CMD: {' '.join(sys.argv)}")
print(f"filter apps: {args.apps}")
logging(f"START")

sim_res = {}
if os.path.exists(args.output):
    with open(args.output, 'r') as f:
        try:
            sim_res = json.load(f)
        except:
            pass
            
for app_and_arg in app_and_arg_list:
    app = app_and_arg.split('/')[0]
    app_trace_dir = os.path.join(args.trace_dir, app_and_arg)
    
    if args.apps and app_and_arg not in args.apps:
        continue
    
    logging(f"{app_and_arg} start")
    app_res = memory_model_warpper(args.config, app_trace_dir, args.model, use_approx=args.use_approx, granularity=args.granularity, filter_L2=args.filter_l2)
    avg_l1_hit_rate = sum([res['l1_hit_rate'] for res in app_res]) / len(app_res)
    avg_l2_hit_rate = sum([res['l2_hit_rate'] for res in app_res]) / len(app_res)
    print(f"avg_l1_hit_rate: {avg_l1_hit_rate}, avg_l2_hit_rate: {avg_l2_hit_rate}")
    sim_res[app_and_arg] = app_res
    logging(f"{app_and_arg} finished")

    with open(args.output, 'w') as f:
        json.dump(sim_res, f, indent=4)

logging(f"failed list: {failed_list}")
logging(f"End")
log_file.close()