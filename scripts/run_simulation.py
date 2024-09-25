import argparse

import subprocess
import shlex
import re
import os
import sys
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
parser.add_argument("--apps",
                    nargs="*",
                    help="only run specific apps")
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
parser.add_argument("--libmpich-path",
                    default="/usr/lib/x86_64-linux-gnu/libmpich.so",
                    help="path to libmpich.so")
parser.add_argument("-M", "--mpi-run",
                    default="",
                    help="mpirun string, e.g mpirun -n 2")
parser.add_argument("-R", "--report-output-dir",
                    default="output",
                    help="output to a seprate dir, not in the trace dir")
parser.add_argument("--hw-res",
                    help="hw res json file. use to fix l2 cache miss rate")
parser.add_argument("--no-overwrite", dest="overwrite",
                 action="store_false",
                 help="if overwrite=False, then don't simulate already have .out file app")
args = parser.parse_args()

from common import *

defined_apps = {}
parse_app_definition_yaml(args.benchmarks_yaml, defined_apps)
apps = gen_apps_from_suite_list(args.benchmark_list.split(","), defined_apps)
app_and_arg_list = get_app_arg_list(apps)

log_file = open(args.log_file, "a")
def logging(*args, **kwargs):
    print(*args, **kwargs, file=log_file, flush=True)

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True)
    return res.returncode

logging(f"Start")
failed_list = []
logging(f"CMD: {' '.join(sys.argv)}")
print(args.apps)
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
    
    if args.apps and app_and_arg not in args.apps:
        continue

    logging(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {app_and_arg}")
    hw_res_option_str = f"--hw-res {args.hw_res}" if args.hw_res else ""
    if args.mpi_run != "":
        cmd = f"{args.mpi_run} python ppt.py --mpi --libmpich-path {args.libmpich_path} --app {app_trace_dir} --sass --config {args.hw_config} --granularity {args.granularity} {hw_res_option_str} --report-output-dir {args.report_output_dir}"
    else:
        cmd = f"python ppt.py --app {app_trace_dir} --libmpich-path {args.libmpich_path} --sass --config {args.hw_config} --granularity {args.granularity} {hw_res_option_str} --report-output-dir {args.report_output_dir}"
    # logging(cmd)
    try:
        exit_status = run_cmd(cmd)
        if exit_status!=0:
            logging(f"{app} failed")
            failed_list.append(app_and_arg)
            print(f"{app} failed")
        else:
            logging(f"{app} success")
    except KeyboardInterrupt:
        log_file.close()
        exit(0)
print(f"failed list: {failed_list}")
logging(f"failed list: {failed_list}")
logging(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: End")
log_file.close()