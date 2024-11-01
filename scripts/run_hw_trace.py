import argparse

import signal
import subprocess
import os
import datetime
import sys

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
parser.add_argument("-D", "--device_num",
                    help="CUDA device number",
                    default="0")
parser.add_argument("-c", "--kernel_number",
                    type=int,
                    default=300,
                    help="Sets a hard limit to the number of traced limits")
parser.add_argument("--trace_tool",
                    help="nvbit trace tool .so file path")
parser.add_argument("-l", "--log_file",
                    default="run_hw_trace.log")
parser.add_argument("-n", "--norun",
                 action="store_true")
parser.add_argument("--no-overwrite", dest="overwrite",
                    action="store_false",
                    help="if memory_traces exists, then skip")
parser.add_argument("-t", "--time-out",
                    type=int,
                    default=3*60*60, # 3h
                    help="Set time out seconds, if app run longer than this, kill it")
parser.add_argument("-r", "--run_script",
                 default="run_tracing.sh")
args = parser.parse_args()

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

logging(f"run hw trace {args.trace_tool}")
print(f"filter: {args.apps}")
logging(f"START")
failed_list = []
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
        
        logging(f"{app_and_arg} start")
        # mkdir run_dir
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        # link data dir (skip)
        curr_data_dir = os.path.join(run_dir, "data")
        if os.path.lexists(curr_data_dir):
            os.remove(curr_data_dir)
        os.symlink(data_dir, curr_data_dir)
        
        sh_contents = ""
        if args.kernel_number > 0:
            sh_contents += "export TERMINATE_UPON_LIMIT=1; "
            sh_contents += f'export DYNAMIC_KERNEL_LIMIT_END={args.kernel_number}; '
        
        sh_contents += f'\nexport CUDA_VISIBLE_DEVICES="{args.device_num}"'\
                f'\nexport LD_PRELOAD={args.trace_tool}'\
                f'\n{exec_path} {argstr}\n'
        
        run_script_path = os.path.join(run_dir, args.run_script)
        open(run_script_path, "w").write(sh_contents)
        if subprocess.call(['chmod', 'u+x', run_script_path]) != 0:
            exit("Error chmod runfile")

        memory_traces_dir = os.path.join(run_dir, "memory_traces")
        if os.path.exists(memory_traces_dir) and not args.overwrite:
                logging(f"{app_and_arg} exists, skip")
                continue
        
        if not args.norun:
            saved_dir = os.getcwd()
            os.chdir(run_dir)

            try:
                p = subprocess.Popen(["bash", args.run_script], start_new_session=True)
                p.wait(timeout=args.time_out)
                if p.returncode != 0:
                    logging(f"Error invoking nvbit in {app_and_arg}")
                    failed_list.append(app_and_arg)
                else:
                    logging(f"{app_and_arg} finished")
            except subprocess.TimeoutExpired as e:
                logging(f"Timeout in {app_and_arg}")
                failed_list.append(app_and_arg)
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                logging(f"Killed {app_and_arg}")
            except KeyboardInterrupt:
                logging(f"Ctrl-C {app_and_arg}")
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                log_file.close()
                exit(-1)
            os.chdir(saved_dir)
logging(f"END")
logging(f"Failed list: {failed_list}")
log_file.close()
