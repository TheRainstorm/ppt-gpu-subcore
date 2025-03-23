import argparse

import signal
import subprocess
import os
from datetime import datetime
import sys

parser = argparse.ArgumentParser(
    description='run gpgpu-sim'
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
parser.add_argument("-c", "--kernel_number",
                    type=int,
                    default=300,
                    help="Sets a hard limit to the number of traced limits")
# parser.add_argument("--loop-cnt",
#                     type=int,
#                     default=3,
#                     help="run multiple times")
parser.add_argument("-l", "--log_file",
                    default="run_hw_profiling.log")
parser.add_argument("-n", "--norun",
                    action="store_true")
parser.add_argument("-r", "--run_script",
                    default="run_profiling.sh")
# parser.add_argument("-o", "--profiling_filename",
#                     default="profiling.csv")
parser.add_argument("--no-overwrite", dest="overwrite",
                    action="store_false",
                    help="if overwrite=False, then don't profile when cvs exist")
parser.add_argument("-t", "--time-out",
                    type=int,
                    default=3*60*60, # 3h
                    help="Set time out seconds, if app run longer than this, kill it")
parser.add_argument("-p", "--nproc", type=int,
                    default=64, help='number of processes to run in parallel')
parser.add_argument('--gpgpu-sim-config-path',
                    default="e.g /gpgpu-sim/configs/tested-cfgs/SM7_TITANV/gpgpusim.config",
                    help="path to gpgpu-sim config")
parser.add_argument('--gpgpu-sim-lib-path',
                    default="e.g /gpgpu-sim/lib/gcc-7.5.0/cuda-11000/release/libcudart.so",
                    help="path to gpgpu-sim libcudart.so")
args = parser.parse_args()

from common import *

NCU = os.environ.get("NCU", "ncu")

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

def run_one(app_and_arg, run_dir, run_script_path, time_out):
    success = False
    os.chdir(run_dir)
    try:
        p = subprocess.Popen(["bash", run_script_path], start_new_session=True)
        p.wait(timeout=None if time_out==0 else time_out)
        
        if p.returncode != 0:
            # logging(f"{exe_name} failed")
            pass
        else:
            success = True
            # logging(f"{app_and_arg} finished")
    except subprocess.TimeoutExpired:
        # logging(f"Timeout in {app_and_arg}")
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        # logging(f"Killed {app_and_arg}")
    except KeyboardInterrupt:
        # logging(f"Ctrl-C {app_and_arg}")
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        # log_file.close()
    return success, app_and_arg

failed_list = []
logging(f"run GPGPU-sim")
logging(f"START")

# process parallel
from multiprocessing import Pool
pool = Pool(args.nproc)

results = []
for app in apps:
    exec_dir, data_dir, exe_name, args_list = app
    exec_dir = os.path.expandvars(exec_dir)
    data_dir = os.path.expandvars(data_dir)
    for argpair in args_list:
        argstr = "" if argpair["args"]==None else argpair["args"]
        mem_usage = argpair["accel-sim-mem"]
        app_and_arg = os.path.join( exe_name, get_argfoldername( argstr ) )  # backprop-rodinia-2.0-ft/4096___data_result_4096_txt
        exec_path = os.path.join(exec_dir, exe_name)
        run_dir = os.path.join(args.trace_dir, app_and_arg)
        
        if args.apps and app_and_arg not in args.apps:
            continue
        
        # mkdir run_dir
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        # link data dir
        curr_data_dir = os.path.join(run_dir, "data")
        if not os.path.lexists(curr_data_dir):
            os.symlink(data_dir, curr_data_dir)
            
        # link gpgpu-sim config
        gpgpu_sim_config_path = os.path.join(run_dir, "gpgpusim.config")
        if not os.path.lexists(gpgpu_sim_config_path):
            os.symlink(args.gpgpu_sim_config_path, gpgpu_sim_config_path)
        
        loop = 0
        profiling_output = os.path.join(run_dir, f"gpgpu-sim.{exe_name}.{loop}.log")
        if os.path.exists(profiling_output) and not args.overwrite:
            logging(f"{app_and_arg} exists, skip")
            continue
        
        logging(f"{app_and_arg} start")
        # fix gpgpu-sim not work
        sh_contents = f'export CUOBJDUMP_SIM_FILE=jj\n'\
            f'LD_PRELOAD={args.gpgpu_sim_lib_path} {exec_path} {argstr} > {profiling_output}\n'
        
        run_script_path = os.path.join(run_dir, args.run_script)
        with open(run_script_path, "w") as f:
            f.write(sh_contents)
        subprocess.call(['chmod', 'u+x', run_script_path])
    
        results.append(pool.apply_async(run_one, (app_and_arg, run_dir, run_script_path, args.time_out)))
pool.close()
pool.join()

failed_list = []
for res in results:
    success, app_and_arg = res.get()
    if not success:
        failed_list.append(app_and_arg)

logging(f"END")
logging(f"failed list: {failed_list}")
log_file.close()