import argparse

import signal
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
                default="")
parser.add_argument("--apps",
                    nargs="*",
                    help="only run specific apps")
parser.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
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
# parser.add_argument("--libmpich-path",
#                     default="/usr/lib/x86_64-linux-gnu/libmpich.so",
#                     help="path to libmpich.so")
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
parser.add_argument("-t", "--time-out",
                    type=int,
                    default=3*60*60, # 3h
                    help="Set time out seconds, if app run longer than this, kill it")
parser.add_argument("--ppt-src", default="ppt.py", help="ppt.py src path")
parser.add_argument("--extra-params", default="", help="ppt.py extra param")
args = parser.parse_args()

if args.extra_params.startswith("<"):
    args.extra_params = args.extra_params[1:-1]
    print(f"extra_params: {args.extra_params}")
    
from common import *

# defined_apps = {}
# parse_app_definition_yaml(args.benchmarks_yaml, defined_apps)
apps = gen_apps_from_suite_list(args.benchmark_list)
app_and_arg_list = get_app_arg_list(apps)
# args.apps = process_args_apps(args.apps, defined_apps)
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
for app_and_arg in app_and_arg_list:
    app = app_and_arg.split('/')[0]
    app_trace_dir = os.path.join(args.trace_dir, app_and_arg)
    
    if args.apps and app_and_arg not in args.apps:
        continue
    
    kernels = suite_info['kernels'].get(app_and_arg, [])
    kernel_ids = ' '.join([str(k) for k in kernels])

    logging(f"{app_and_arg} start")
    hw_res_option_str = f"--hw-res {args.hw_res}" if args.hw_res else ""
    no_overwrite_str = "--no-overwrite" if not args.overwrite else ""
    if args.mpi_run != "":
        cmd = f"{args.mpi_run} python {args.ppt_src} {args.extra_params} --mpi --app {app_trace_dir} --sass --config {args.hw_config} --granularity {args.granularity} {hw_res_option_str} --report-output-dir {args.report_output_dir} --kernel {kernel_ids} {no_overwrite_str}"
    else:
        cmd = f"python {args.ppt_src} {args.extra_params} --app {app_trace_dir} --sass --config {args.hw_config} --granularity {args.granularity} {hw_res_option_str} --report-output-dir {args.report_output_dir} --kernel {kernel_ids} {no_overwrite_str}"

    # print(cmd)
    try:
        p = subprocess.Popen(shlex.split(cmd), start_new_session=True)
        p.wait(timeout=args.time_out)
        if p.returncode != 0:
            logging(f"{app_and_arg} failed")
            failed_list.append(app_and_arg)
        else:
            logging(f"{app_and_arg} finished")
    except subprocess.TimeoutExpired:
        logging(f"Timeout in {app_and_arg}")
        failed_list.append(app_and_arg)
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        logging(f"Killed {app_and_arg}")
    except KeyboardInterrupt:
        logging(f"Ctrl-C {app_and_arg}")
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        log_file.close()
        exit(-1)
    except Exception as e:
        logging(f"Exception in {app_and_arg}: {e}")
        failed_list.append(app_and_arg)
    
logging(f"failed list: {failed_list}")
logging(f"End")
log_file.close()