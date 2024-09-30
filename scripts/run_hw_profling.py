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
parser.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
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
# parser.add_argument("-o", "--profiling_filename",
#                     default="profiling.csv")
parser.add_argument("--no-overwrite", dest="overwrite",
                    action="store_false",
                    help="if overwrite=False, then don't profile when cvs exist")
parser.add_argument("--select", default="nvprof",
                    choices=["nvprof", "ncu", "ncu-cpi"],
                    help="select which metrics to profile")
args = parser.parse_args()

from common import *

# defined_apps = {}
# parse_app_definition_yaml(args.benchmarks_yaml, defined_apps)
apps = gen_apps_from_suite_list(args.benchmark_list.split(","), defined_apps)
app_and_arg_list = get_app_arg_list(apps)
# args.apps = process_args_apps(args.apps, defined_apps)
args.apps = filter_app_list(app_and_arg_list, args.app_filter)

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
            
            profiling_output = os.path.join(run_dir, f"profiling.{args.select}.{loop}.csv")
            if os.path.exists(profiling_output) and not args.overwrite:
                logging(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {app_and_arg} exists, skip")
                continue
            
            logging(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {app_and_arg}")
            sh_contents = ""
            # nvprof
            # --concurrent-kernels off 
            if args.select == "ncu":
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'ncu --csv --log-file {profiling_output} --metric='\
                        f'gpc__cycles_elapsed.avg,gpc__cycles_elapsed.max,'\
                        f'smsp__inst_executed.sum,smsp__inst_executed.sum,smsp__inst_executed.avg.per_cycle_active,smsp__inst_issued.avg.per_cycle_active,sm__cycles_active.sum,sys__cycles_active.sum,sm__cycles_elapsed.sum,sm__cycles_elapsed.sum,sys__cycles_elapsed.sum,sm__warps_active.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,lts__t_sector_hit_rate.pct,lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,dram__sectors_read.sum,dram__sectors_write.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,lts__t_sectors_srcunit_tex_op_read.sum,l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active,l1tex__tex_writeback_active.avg.pct_of_peak_sustained_active' \
                        f' {exec_path} {argstr}'
            elif args.select == 'ncu-cpi': # cpi stack
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'ncu --csv --log-file {profiling_output} --metrics="smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio",smsp__average_warp_latency_per_inst_issued.ratio ' \
                        f' {exec_path} {argstr}'
            else:
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'nvprof --print-gpu-trace --csv --log-file {profiling_output} ' \
                        f'-u us -e active_cycles_pm,active_warps_pm,elapsed_cycles_sm,elapsed_cycles_pm,active_cycles,active_warps,elapsed_cycles_sys,active_cycles_sys ' \
                        f'-m achieved_occupancy,inst_executed,inst_issued,ipc,issued_ipc,global_hit_rate,tex_cache_hit_rate,l2_tex_hit_rate,global_load_requests,global_store_requests,gld_transactions,gst_transactions,l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions ' \
                        f' {exec_path} {argstr}'
            
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