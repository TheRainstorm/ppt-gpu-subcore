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

def parse_app_definition_yaml(def_yml):
    ''' read app definition yaml file and return a dict containing list of app 4-element-tuple at different levels(suite, apps, count)
    '''
    apps = {}
    benchmark_yaml = yaml.load(open(def_yml), Loader=yaml.FullLoader)
    for suite in benchmark_yaml:
        apps[suite] = []
        for exec in benchmark_yaml[suite]['execs']:
            exe_name = list(exec.keys())[0]  # dict has only one key (exe name) and one value (list of args dict)
            args_list = list(exec.values())[0]
            count = 0
            
            data_dirs = benchmark_yaml[suite].get('data_dirs', "")
            if 'data_subdir' in exec:
                data_dirs = os.path.join(data_dirs, exec['data_subdir'])
            else:
                data_dirs = os.path.join(data_dirs, exe_name)
            
            args_list_not_ignore = []
            for runparms in args_list:
                if 'ignore' in runparms and runparms['ignore']:
                    continue
                args_list_not_ignore.append(runparms)
                args = runparms["args"] if runparms["args"] else ""
                if "accel-sim-mem" not in runparms:
                    runparms["accel-sim-mem"] = "4G"
                apps[suite + ":" + exe_name + ":" + str(count) ] = []
                apps[suite + ":" + exe_name + ":" + str(count) ].append( ( benchmark_yaml[suite]['exec_dir'],
                                    data_dirs,
                                    exe_name, [runparms]) )
                count += 1
            apps[suite].append(( benchmark_yaml[suite]['exec_dir'],
                                 data_dirs,
                                 exe_name, args_list_not_ignore ))
            apps[suite + ":" + exe_name] = []
            apps[suite + ":" + exe_name].append( ( benchmark_yaml[suite]['exec_dir'],
                                 data_dirs,
                                 exe_name, args_list_not_ignore ) )
    return apps

def gen_apps_from_suite_list(suite_list=""):
    global suite_info
    if suite_list == "all" or suite_list == "":
        suite_list = suite_info['suites']
    else:
        suite_list = suite_list.split('|')
    
    apps = []
    for suite in suite_list:
        if suite not in defined_apps:
            print(f"Warning: suite {suite} not found in defined apps")
            continue
        apps += defined_apps[suite]
    return apps

def get_app_arg_list(apps):
    # convert app tuple list to app_and_arg list
    app_and_arg_list = []
    for app in apps:
        exec_dir, data_dir, exe_name, args_list = app
        for argpair in args_list:
            # mem_usage = argpair["accel-sim-mem"]
            app_and_arg_list.append(os.path.join( exe_name, get_argfoldername( argpair["args"] ) ))  # backprop-rodinia-2.0-ft/4096___data_result_4096_txt
    return app_and_arg_list

def get_suite_info(def_yml):
    '''the app_and_arg loss the info of which suite it belongs to, 
    this function retuns a map from app_and_arg to suite, exe, count
    '''
    benchmark_yaml = yaml.load(open(def_yml), Loader=yaml.FullLoader)
    info = {}
    info['suites'] = []
    info['map'] = {}
    info['kernels'] = {}
    
    for suite in benchmark_yaml:
        info['suites'].append(suite)
        for exe in benchmark_yaml[suite]['execs']:
            exe_name = list(exe.keys())[0]  # dict has only one key (exe name) and one value (list of args dict)
            args_list = list(exe.values())[0]
            count = 0
            abbr = exe.get('abbr', exe_name)
            for runparms in args_list:
                args = runparms["args"]
                app_and_arg = os.path.join( exe_name, get_argfoldername( args ) )
                
                if 'kernels' in runparms:
                    info['kernels'][app_and_arg] = runparms['kernels']
                
                true_suite = benchmark_yaml[suite].get('suite', suite)
                if app_and_arg not in info['map']:
                    info['map'][app_and_arg] = ( true_suite, exe_name, str(count), abbr)
                count += 1
    return info

def filter_app_list_coord(app_arg_list, coord_str):
    '''
    filter app_arg contained in [suite]:[exe]:[count] 
    '''
    global suite_info
    def get_coord(filter_expr):
        parts = filter_expr.split(':')
        if len(parts) == 1:
            return (parts[0], None, None)
        elif len(parts) == 2:
            return (parts[0], parts[1], None)
        else:
            return (parts[0], parts[1], parts[2])
    def contain_in(c1, c2):
        if c2[0] and c1[0] not in c2[0]:  # suite 'polybench in polybench-full'
            return False
        for i in range(1, len(c2)):
            if c2[i] and c1[i] != c2[i]:
                return False
        return True
    
    coord_filter = get_coord(coord_str)
    
    new_app_arg_list = []
    for app_arg in app_arg_list:
        if not app_arg in suite_info['map']:
            continue
        if not contain_in(suite_info['map'][app_arg], coord_filter):
            continue
        new_app_arg_list.append(app_arg)
    return new_app_arg_list

def filter_app_list_re(app_arg_list, rexp):
    '''
    filter app_arg matching the regex
    '''
    new_app_arg_list = []
    for app_arg in app_arg_list:
        m = re.search(rexp, app_arg)
        if m:
            new_app_arg_list.append(app_arg)
    return new_app_arg_list

def filter_app_list(all_app_list, app_filter):
    if '|' in app_filter:
        filter_list = app_filter.split('|')
        app_list = []
        for filter in filter_list:
            curr_app_list = filter_app_list(all_app_list, filter)
            app_list += curr_app_list
        # delete duplicates, without changing the order
        app_list = list(dict.fromkeys(app_list))
        return app_list
    
    if app_filter == '':
        app_list = all_app_list
    elif app_filter.startswith('regex:'):
        regex = app_list[6:]
        app_list = filter_app_list_re(all_app_list, regex)
    elif '/' in app_filter:
        app_list = app_filter.split(',')
    else:
        coord_filter = app_filter
        app_list = filter_app_list_coord(all_app_list, coord_filter)
    return app_list

defined_apps = parse_app_definition_yaml(os.environ['apps_yaml'])
suite_info = get_suite_info(os.environ['apps_yaml'])

from datetime import datetime
def now_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
