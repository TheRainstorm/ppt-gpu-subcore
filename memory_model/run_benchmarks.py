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
                    choices=['ppt-gpu', 'sdcm'],
                    default='ppt-gpu',
                    help='change memory model')
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
for app_and_arg in app_and_arg_list:
    app = app_and_arg.split('/')[0]
    app_trace_dir = os.path.join(args.trace_dir, app_and_arg)
    
    if args.apps and app_and_arg not in args.apps:
        continue
    
    logging(f"{app_and_arg} start")
    app_res = memory_model_warpper(args.config, app_trace_dir, args.model)
    sim_res[app_and_arg] = app_res
    logging(f"{app_and_arg} finished")

    with open(args.output, 'w') as f:
        json.dump(sim_res, f, indent=4)

logging(f"failed list: {failed_list}")
logging(f"End")
log_file.close()