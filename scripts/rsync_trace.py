import argparse

import json
import subprocess
import os
from datetime import datetime
from humanize import naturalsize

def get_dir_size(path):
    cmd = f"du {path}"
    res = subprocess.run(cmd, shell=True, capture_output=True)
    size = res.stdout.decode().split('\t')[0]
    # print(cmd)
    # print(res.stdout.decode())
    # exit(1)
    return int(size)*1024

def scan_trace_dir(trace_dir):
    # apps_result['<backprop>']['<arg>'] = {'kernels':2, 'memory_trace_size': "10G", 'sass_trace_size': "10G"}
    apps_result = {}
    for app in os.listdir(trace_dir):
        app_res = {}
        for arg in os.listdir(os.path.join(trace_dir, app)):
            app_arg_path = os.path.join(args.trace_dir, app, arg)
            app_arg_res = {}
            
            # get kernel number
            try:
                app_config_file = os.path.join(app_arg_path, 'app_config.py')
                with open(app_config_file) as f:
                    py_content = f.read()
                    local_ns = {}
                    exec(py_content, local_ns)
                app_arg_res['kernels'] = len(local_ns['app_kernels_id'])
            except:
                app_arg_res['kernels'] = 0
                
            # get trace size
            try:
                app_arg_res['memory_trace_size'] = get_dir_size(os.path.join(app_arg_path, 'memory_traces'))
                app_arg_res['sass_trace_size'] = get_dir_size(os.path.join(app_arg_path, 'sass_traces'))
            except:
                # print(f"warning: {app}/{arg} memory_trace sass_trace no exist")
                continue
            app_res[arg] = app_arg_res
        apps_result[app] = app_res
    return apps_result

def summary_trace_dir_result(apps_result):
    # print(apps_result)
    top_result = {} # suites_result['suites']['<rodinia>']['apps']['<backprop>']['runs']['<args>'] = {'kernels': 10, 'memory_trace_size': "100G", 'sass_trace_size': "100G"}
    top_result['suites'] = {}
    
    suite_result = {}
    suite_result['Apps'] = {}
    
    # create level key
    suite = ""
    for app, app_res in apps_result.items():
        app_result = {}
        app_result['runs'] = {}
        for arg, arg_res in app_res.items():
            app_result['runs'][arg] = arg_res
            
            app_arg = f"{app}/{arg}"
            try:
                suite = suite_info['map'][app_arg][0]
            except:
                print(f"warning: {app_arg} not in yaml")
                continue
        
        if suite and suite not in top_result['suites']:
            top_result['suites'][suite] = {}
            top_result['suites'][suite]['Apps'] = {}
        top_result['suites'][suite]['Apps'][app] = app_result
    
    # sum at each level
    top_result['apps'] = 0
    top_result['app_runs'] = 0  # each app can have multiple runs
    top_result['kernels'] = 0
    top_result['memory_trace_size'] = 0
    top_result['sass_trace_size'] = 0
    for suite, suite_res in top_result['suites'].items():
        suite_res['kernels'] = 0
        suite_res['apps'] = 0
        suite_res['app_runs'] = 0
        suite_res['memory_trace_size'] = 0
        suite_res['sass_trace_size'] = 0
        for app, app_res in suite_res['Apps'].items():
            app_res['kernels'] = 0
            app_res['memory_trace_size'] = 0
            app_res['sass_trace_size'] = 0
            for arg, arg_res in app_res['runs'].items():
                app_res['kernels'] += arg_res['kernels']
                app_res['memory_trace_size'] += arg_res['memory_trace_size']
                app_res['sass_trace_size'] += arg_res['sass_trace_size']
                suite_res['app_runs'] += 1
            suite_res['kernels'] += app_res['kernels']
            suite_res['apps'] += 1
            suite_res['memory_trace_size'] += app_res['memory_trace_size']
            suite_res['sass_trace_size'] += app_res['sass_trace_size']
            suite_res['Apps'][app] = app_res
            
        top_result['apps'] += suite_res['apps']
        top_result['app_runs'] += suite_res['app_runs']
        top_result['kernels'] += suite_res['kernels']
        top_result['memory_trace_size'] += suite_res['memory_trace_size']
        top_result['sass_trace_size'] += suite_res['sass_trace_size']
    
    with open('trace_summary.json', 'w') as f:
        json.dump(top_result, f, indent=4)
    
    import prettytable as pt
    tb = pt.PrettyTable()
    tb_app = pt.PrettyTable()
    tb.field_names = ["Suite", "apps", "app_runs", "Kernels", "Memory trace", "SASS trace"]
    tb_app.field_names = ["Suite", "App", "Args", "Kernels", "Memory trace", "SASS trace"]
    tb.add_row(["Total", top_result['apps'], top_result['app_runs'], top_result['kernels'], naturalsize(top_result['memory_trace_size']), naturalsize(top_result['sass_trace_size'])])
    tb_app.add_row(["Total", "-", "-", top_result['kernels'], naturalsize(top_result['memory_trace_size']), naturalsize(top_result['sass_trace_size'])])

    # print top
    # print(f"Total Apps: {top_result['apps']} ({top_result['app_runs']})")
    # print(f"Total kernels: {top_result['kernels']}")
    # print(f"Total memory trace size: {naturalsize(top_result['memory_trace_size'])}")
    # print(f"Total sass trace size: {naturalsize(top_result['sass_trace_size'])}")
    
    for suite, suite_res in top_result['suites'].items():
        # print(f"Suite: {suite}")
        # print(f"Apps: {suite_res['apps']} ({suite_res['app_runs']})")
        # print(f"kernels: {suite_res['kernels']}")
        # print(f"memory trace size: {naturalsize(suite_res['memory_trace_size'])}")
        # print(f"sass trace size: {naturalsize(suite_res['sass_trace_size'])}")
        tb.add_row([suite, suite_res['apps'], suite_res['app_runs'], suite_res['kernels'], naturalsize(suite_res['memory_trace_size']), naturalsize(suite_res['sass_trace_size'])])
        
        for app, app_res in suite_res['Apps'].items():
        #     print(f"App: {app}")
        #     print(f"kernels: {app_res['kernels']}")
        #     print(f"memory trace size: {naturalsize(app_res['memory_trace_size'])}")
        #     print(f"sass trace size: {naturalsize(app_res['sass_trace_size'])}")
            tb_app.add_row([suite, app, "-", app_res['kernels'], naturalsize(app_res['memory_trace_size']), naturalsize(app_res['sass_trace_size'])])
            
            for arg, arg_res in app_res['runs'].items():
        #         print(f"Args: {arg}")
        #         print(f"kernels: {arg_res['kernels']}")
        #         print(f"memory trace size: {naturalsize(arg_res['memory_trace_size'])}")
        #         print(f"sass trace size: {naturalsize(arg_res['sass_trace_size'])}")
                tb_app.add_row([suite, app, arg, arg_res['kernels'], naturalsize(arg_res['memory_trace_size']), naturalsize(arg_res['sass_trace_size'])])
        tb_app.add_row([suite, '-', "-", suite_res['kernels'], naturalsize(suite_res['memory_trace_size']), naturalsize(suite_res['sass_trace_size'])], divider=True)
    
    print(tb)
    print(tb_app)
    return top_result

parser = argparse.ArgumentParser(
    description='trace tools'
)
parser.add_argument("-T", "--trace_dir",
                    required=True,
                    help="The root of all the trace file")
# sub command
subparsers = parser.add_subparsers(dest='command', help='sub-command help')

# scan size
parser_scan = subparsers.add_parser('scan', help='scan size help')

# rsync
parser_rsync = subparsers.add_parser('rsync', help='rsync help')
parser_rsync.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
parser_rsync.add_argument("-R", "--remote",
                    required=True,
                    help="e.g icarus4")
parser_rsync.add_argument("-n", "--norun",
                    action="store_true")
args = parser.parse_args()


from common import *

apps = gen_apps_from_suite_list()
app_and_arg_list = get_app_arg_list(apps)

if args.command == 'scan':
    apps_result = scan_trace_dir(args.trace_dir)
    summary_trace_dir_result(apps_result)
    
elif args.command == 'rsync':
    app_args_list_filtered = filter_app_list(app_and_arg_list, args.app_filter)

    rsync_txt = "/tmp/rsync_dirs.txt"
    with open(rsync_txt, 'w') as f:
        for app_args in app_args_list_filtered:
            app_dir, arg_dir = app_args.split('/')
            f.write(app_dir + '\n')
            # app_config_file = os.path.join(args.trace_dir, app_args, 'app_config.py')
            # with open(app_config_file) as f:
            #     f.write(app_dir)
    cmd = f"rsync -av --files-from={rsync_txt} {args.trace_dir} {args.remote}:{args.trace_dir}"
    print(cmd)
