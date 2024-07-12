import argparse

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
parser.add_argument("-l", "--log_file",
                    default="run_sim.log",
                    help="Save sim")
parser.add_argument("-H", "--hw_config",
                    default="TITANV",
                    help="The name of hardware config, check ./hardware/*.py")
parser.add_argument("-g", "--granularity",
                    default="2",
                    help="1=One Thread Block per SM or 2=Active Thread Blocks per SM or 3=All Thread Blocks per SM")
parser.add_argument("--no-overwrite", dest="overwrite",
                 action="store_false",
                 help="if overwrite=False, then don't simulate already have .out file app")
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

log_file = open(args.log_file, "a")
def logging(*args, **kwargs):
    print(*args, **kwargs, file=log_file, flush=True)

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True)
    return res.returncode

for app_and_arg in app_and_arg_list:
    app = app_and_arg.split('/')[0]
    app_trace_dir = os.path.join(args.trace_dir, app_and_arg)
    
    already_simulated = False
    for file in os.listdir(app_trace_dir):
        if file.endswith(".out") and not args.overwrite:
            already_simulated = True
            break
    if already_simulated:
        logging(f"{app_and_arg} already simulated")
        continue

    logging(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {app_and_arg}")
    cmd = f"python ppt.py --app {app_trace_dir} --sass --config {args.hw_config} --granularity {args.granularity}"
    # logging(cmd)
    try:
        exit_status = run_cmd(cmd)
        if exit_status!=0:
            logging(f"{app} failed")
        else:
            logging(f"{app} success")
    except KeyboardInterrupt:
        log_file.close()
        exit(0)
logging(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: End")
log_file.close()