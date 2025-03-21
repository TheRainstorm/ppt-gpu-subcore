import argparse

import signal
import subprocess
import os
from datetime import datetime
import sys

parser = argparse.ArgumentParser(
    description='run hw profiling'
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
parser.add_argument("-D", "--device_num",
                    help="CUDA device number",
                    default="0")
parser.add_argument("-c", "--kernel_number",
                    type=int,
                    default=300,
                    help="Sets a hard limit to the number of traced limits")
parser.add_argument("--loop-cnt",
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
parser.add_argument("-t", "--time-out",
                    type=int,
                    default=3*60*60, # 3h
                    help="Set time out seconds, if app run longer than this, kill it")
parser.add_argument("--select", default="nvprof",
                    # choices=["nvprof", "ncu", "ncu-cpi", "nvprof-cpi", "ncu-full", "ncu-rep"],
                    help="select which metrics to profile")
parser.add_argument('--replay-control',
                    # default=" --replay-mode application --cache-control all ",
                    default=" ",
                    help="ncu replay control")
parser.add_argument('--ncu-rep-dir',
                    default="",
                    help="copy ncu rep to dst dir")
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

failed_list = []
logging(f"run hw profiling {args.select}")
logging(f"START")
for loop in range(args.loop_cnt):
    logging(f"loop {loop}")
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
            
            profiling_output = os.path.join(run_dir, f"profiling.{args.select}.{loop}.csv")
            if os.path.exists(profiling_output) and not args.overwrite:
                logging(f"{app_and_arg} exists, skip")
                continue
            
            logging(f"{app_and_arg} start")
            run_sh_contents = f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'nvprof {exec_path} {argstr}\n'
            run_sh_path = os.path.join(run_dir, "run.sh")
            with open(run_sh_path, "w") as f:
                f.write(run_sh_contents)
            subprocess.call(['chmod', 'u+x', run_sh_path])
            sh_contents = ""
            # nvprof
           
            # --concurrent-kernels off 
            if args.select == "ncu-full":
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'{NCU} {args.replay_control} --set full --csv --page raw --log-file {profiling_output} --launch-count {args.kernel_number}'\
                        f' {exec_path} {argstr}\n'
            elif args.select == "ncu":
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'{NCU} {args.replay_control} --csv --log-file {profiling_output} --launch-count {args.kernel_number} --metric='\
                        f'gpc__cycles_elapsed.avg,gpc__cycles_elapsed.max,'\
                        f'smsp__inst_executed.sum,smsp__inst_executed.sum,smsp__inst_executed.avg.per_cycle_active,smsp__inst_issued.avg.per_cycle_active,sm__cycles_active.sum,sys__cycles_active.sum,sm__cycles_elapsed.sum,sm__cycles_elapsed.sum,sys__cycles_elapsed.sum,sm__warps_active.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,lts__t_sector_hit_rate.pct,lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,dram__sectors_read.sum,dram__sectors_write.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,lts__t_sectors_srcunit_tex_op_read.sum,l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active,l1tex__tex_writeback_active.avg.pct_of_peak_sustained_active' \
                        f' {exec_path} {argstr}\n'
            elif args.select == 'ncu-cpi': # cpi stack
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'{NCU} {args.replay_control} --csv --log-file {profiling_output} --launch-count {args.kernel_number} --metrics=regex:"smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio",smsp__average_warp_latency_per_inst_issued.ratio ' \
                        f' {exec_path} {argstr}\n'
            elif args.select == 'nvprof-cpi':
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'nvprof --print-gpu-trace --concurrent-kernels off --csv --log-file {profiling_output} ' \
                        f'-m stall_constant_memory_dependency,stall_exec_dependency,stall_inst_fetch,stall_memory_dependency,stall_memory_throttle,stall_not_selected,stall_other,stall_pipe_busy,stall_sleeping,stall_sync,stall_texture ' \
                        f' {exec_path} {argstr}\n'
            elif args.select == "ncu-cycle":
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'{NCU} {args.replay_control} --csv --log-file {profiling_output} --launch-count {args.kernel_number} --metric='\
                        f'gpc__cycles_elapsed.max'\
                        f' {exec_path} {argstr}\n'
            elif args.select == "ncu-rep":
                if args.ncu_rep_dir == '':
                    print(f"ncu rep dir not specific or not exist")
                    # exit(1)
                    continue
                if not os.path.exists(args.ncu_rep_dir):
                    os.makedirs(args.ncu_rep_dir, exist_ok=True)
                copy_dir = os.path.join(args.ncu_rep_dir, app_and_arg)
                os.makedirs(copy_dir, exist_ok=True)
                
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'{NCU} {args.replay_control} --set full -o profiling.{loop} -f --launch-count {args.kernel_number}'\
                        f' {exec_path} {argstr}\n' \
                        f'cp profiling.{loop}.ncu-rep {copy_dir}/{exe_name}.{loop}.ncu-rep'
            else:
                sh_contents += f'export CUDA_VISIBLE_DEVICES="{args.device_num}";\n' \
                        f'nvprof --print-gpu-trace --concurrent-kernels off --csv --log-file {profiling_output} ' \
                        f'-u us -e active_cycles_pm,active_warps_pm,elapsed_cycles_sm,elapsed_cycles_pm,active_cycles,active_warps,elapsed_cycles_sys,active_cycles_sys ' \
                        f'-m achieved_occupancy,inst_executed,inst_issued,ipc,issued_ipc,global_hit_rate,tex_cache_hit_rate,l2_tex_hit_rate,l2_tex_read_hit_rate,l2_tex_write_hit_rate,global_load_requests,global_store_requests,gld_transactions,,gld_transactions_per_request,gst_transactions,l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions ' \
                        f' {exec_path} {argstr}\n'
            
            run_script_path = os.path.join(run_dir, args.run_script)
            with open(run_script_path, "w") as f:
                f.write(sh_contents)
            subprocess.call(['chmod', 'u+x', run_script_path])
            
            success = False
            if not args.norun:
                saved_dir = os.getcwd()
                os.chdir(run_dir)
                try:
                    p = subprocess.Popen(["bash", run_script_path], start_new_session=True)
                    p.wait(timeout=args.time_out)
                    
                    if p.returncode != 0:
                        logging(f"{exe_name} failed")
                        failed_list.append(app_and_arg)
                    else:
                        success = True
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
                if not success:
                    logging(f"{app_and_arg} failed")
                    os.remove(profiling_output)
                os.chdir(saved_dir)
logging(f"END")
logging(f"failed list: {failed_list}")
log_file.close()