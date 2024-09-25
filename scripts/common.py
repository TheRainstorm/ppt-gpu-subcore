import hashlib
import os
import re
import yaml


def get_argfoldername( args ):
    if args == "" or args == None:
        return "NO_ARGS"
    else:
        foldername = re.sub(r"[^a-z^A-Z^0-9]", "_", str(args).strip())
        # For every long arg lists - create a hash of the input args
        if len(str(args)) > 256:
            foldername = "hashed_args_" + hashlib.md5(args).hexdigest()
        return foldername

def parse_app_definition_yaml( def_yml, apps ):
    benchmark_yaml = yaml.load(open(def_yml), Loader=yaml.FullLoader)
    for suite in benchmark_yaml:
        apps[suite] = []
        for exe in benchmark_yaml[suite]['execs']:
            exe_name = list(exe.keys())[0]
            args_list = list(exe.values())[0]
            count = 0
            for runparms in args_list:
                args = runparms["args"]
                if "accel-sim-mem" not in runparms:
                    runparms["accel-sim-mem"] = "4G"
                apps[suite + ":" + exe_name + ":" + str(count) ] = []
                apps[suite + ":" + exe_name + ":" + str(count) ].append( ( benchmark_yaml[suite]['exec_dir'],
                                    benchmark_yaml[suite]['data_dirs'],
                                    exe_name, [args]) )
                count += 1
            apps[suite].append(( benchmark_yaml[suite]['exec_dir'],
                                 benchmark_yaml[suite]['data_dirs'],
                                 exe_name, args_list ))
            apps[suite + ":" + exe_name] = []
            apps[suite + ":" + exe_name].append( ( benchmark_yaml[suite]['exec_dir'],
                                 benchmark_yaml[suite]['data_dirs'],
                                 exe_name, args_list ) )
    return

def gen_apps_from_suite_list( suite_list, defined_apps):
    apps = []
    for suite in suite_list:
        apps += defined_apps[suite]
    return apps

def get_app_arg_list(apps):
    app_and_arg_list = []
    for app in apps:
        exec_dir, data_dir, exe_name, args_list = app
        for argpair in args_list:
            mem_usage = argpair["accel-sim-mem"]
            app_and_arg_list.append(os.path.join( exe_name, get_argfoldername( argpair["args"] ) ))  # backprop-rodinia-2.0-ft/4096___data_result_4096_txt
    return app_and_arg_list