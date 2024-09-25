import argparse

import subprocess
import os
from datetime import datetime

parser = argparse.ArgumentParser(
    description='run hw profiling'
)
parser.add_argument("-B", "--benchmark_list",
                    help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for the benchmark suite names.",
                    default="rodinia_2.0-ft")
parser.add_argument("--apps",
                    nargs="*",
                    help="only run specific apps")
parser.add_argument("-Y", "--benchmarks_yaml",
                    required=True,
                    help='benchmarks_yaml path')
parser.add_argument("-T", "--trace_dir",
                    required=True,
                    help="The root of all the trace file")
parser.add_argument("-D", "--device_num",
                    help="CUDA device number",
                    default="0")
parser.add_argument("-t", "--loop_cnt",
                    type=int,
                    default=3,
                    help="run multiple times")
parser.add_argument("-l", "--log_file",
                    default="run_hw_profiling.log")
parser.add_argument("-n", "--norun",
                    action="store_true")
parser.add_argument("-r", "--run_script",
                    default="run_profiling.sh")
parser.add_argument("-o", "--profiling_filename",
                    default="profiling.csv")
parser.add_argument("--no-overwrite", dest="overwrite",
                    action="store_false",
                    help="if overwrite=False, then don't profile when cvs exist")
parser.add_argument("--ncu",
                    action="store_true",
                    help="use ncu to profile")
args = parser.parse_args()

from common import *

defined_apps = {}
parse_app_definition_yaml(args.benchmarks_yaml, defined_apps)
apps = gen_apps_from_suite_list(args.benchmark_list.split(","), defined_apps)

log_file = open(args.log_file, "a")
def logging(*args, **kwargs):
    print(*args, **kwargs, file=log_file, flush=True)

for loop in range(args.loop_cnt):
    logging(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: loop {loop}")
    for app in apps:
        exec_dir, data_dir, exe_name, args_list = app
        exec_dir = os.path.expandvars(exec_dir)
        data_dir = os.path.expandvars(data_dir)
        for argpair in args_list:
            argstr = argpair["args"]
            mem_usage = argpair["accel-sim-mem"]
            app_and_arg = os.path.join( exe_name, get_argfoldername( argstr ) )  # backprop-rodinia-2.0-ft/4096___data_result_4096_txt
            exec_path = os.path.join(exec_dir, exe_name)
            run_dir = os.path.join(args.trace_dir, app_and_arg)
            # link data dir (skip)
            
            if args.apps and app_and_arg not in args.apps:
                continue
            
            profile_type = "nvprof" if not args.ncu else "ncu"
            profiling_output = os.path.join(run_dir, f"{args.profiling_filename}.{profile_type}.{loop}")
            if os.path.exists(profiling_output) and not args.overwrite:
                logging(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {app_and_arg} exists, skip")
                continue
            
            logging(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {app_and_arg}")
            sh_contents = ""
            # nvprof
            # --concurrent-kernels off 
            if args.ncu:
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'ncu --csv --log-file {profiling_output} --metric=regex:"smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio",smsp__average_warp_latency_per_inst_issued.ratio ' \
                        f'{exec_path} {argstr}'
            else:
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'nvprof --print-gpu-trace --csv --log-file {profiling_output} ' \
                        f'-u us -e active_cycles_pm,active_warps_pm,elapsed_cycles_sm,elapsed_cycles_pm,active_cycles,active_warps,elapsed_cycles_sys,active_cycles_sys ' \
                        f'-m achieved_occupancy,inst_executed,inst_issued,ipc,issued_ipc,global_hit_rate,tex_cache_hit_rate,l2_tex_hit_rate,global_load_requests,global_store_requests,gld_transactions,gst_transactions,l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions ' \
                        f'{exec_path} {argstr}'
            
            run_script_path = os.path.join(run_dir, args.run_script)
            with open(run_script_path, "w") as f:
                f.write(sh_contents)
            
            failed_list = []
            if not args.norun:
                saved_dir = os.getcwd()
                os.chdir(run_dir)
                if subprocess.run(f"bash {run_script_path}", shell=True).returncode != 0:
                    failed_list.append(exe_name)
                    logging(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {exe_name} failed")
                os.chdir(saved_dir)
logging(f"failed list: {failed_list}")
print(f"failed list: {failed_list}")
log_file.close()