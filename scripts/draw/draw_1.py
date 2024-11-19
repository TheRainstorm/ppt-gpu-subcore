import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

import sys, os
curr_dir = os.path.dirname(__file__)
par_dir = os.path.dirname(curr_dir)
sys.path.insert(0, os.path.abspath(par_dir))
from common import *

'''
support res format:

{
    app: [
        # kernel 1
        {k: v, k2: v2, ...},
        
        # kernel 2
        {k: v, k2: v2, ...},
        
        ...
    ],
    app2 [
        ...
    ],
    ...
}

kernel should contain 'kernel_name'
'''

# key: PPT-GPU simulation result keyword
# value: corresponding nvprof/ncu profling result keyword
key_map = {
        "achieved_occupancy": "achieved_occupancy",
        "warp_inst_executed": "inst_executed",
        
        "gpu_active_cycle_min": "active_cycles_sys",
        "gpu_active_cycle_max": "active_cycles_sys",
        "sm_active_cycles_sum": "active_cycles", # Number of cycles a multiprocessor has at least one active warp.
        "sm_elapsed_cycles_sum": "elapsed_cycles_sm", # elapsed clocks on SM
        
        "my_gpu_active_cycle_max": "active_cycles_sys",
        "my_sm_active_cycles_sum": "active_cycles",
        "my_sm_elapsed_cycles_sum": "elapsed_cycles_sm",
        
        "ipc": "ipc",
        
        "l1_hit_rate": "global_hit_rate",
        # "l1_hit_rate": "tex_cache_hit_rate",
        "l2_hit_rate": "l2_tex_hit_rate",
        "gmem_read_requests": "global_load_requests",
        "gmem_write_requests": "global_store_requests",
        "gmem_read_trans": "gld_transactions",
        "gmem_write_trans": "gst_transactions",
        "l2_read_trans": "l2_read_transactions",
        "l2_write_trans": "l2_write_transactions",
        "dram_total_trans": "dram_total_transactions",
    }

def get_kernel_stat(json_data, stat_key, app_filter='', func=None):
    '''
    construct X, Y. X: kernel name, Y: stat
    '''
    all_app_list = json_data.keys()
    app_list = filter_app_list(all_app_list, app_filter)
    
    X, Y = [], []
    for i,app in enumerate(app_list):
        try:
            kernels_res = json_data[app]
        except:
            print(f"ERROR: app {app} not found")
            continue
        
        for j,kernel_res in enumerate(kernels_res):
            app_short_name = app.replace('_', '')
            app_short_name = app_short_name[:3]+app_short_name[-10:]
            X.append(f"{app_short_name}-{j}-{kernel_res['kernel_name']}")
            if func:
                Y.append(func(kernel_res[stat_key]))
            else:
                Y.append(kernel_res[stat_key])
    return X, Y

def get_app_stat(json_data, stat_key, app_filter='', func=None, avg=False):
    '''
    construct X, Y. X: app name, Y: app stat (sum of kernel stat)
    '''
    all_app_list = json_data.keys()
    app_list = filter_app_list(all_app_list, app_filter)
    
    # sum kernel stat
    X, Y = [], []
    for app in app_list:
        try:
            kernels_res = json_data[app]
        except:
            print(f"ERROR: app {app} not found")
            continue
        
        X.append(app)
        accum = 0
        try:
            for kernel_res in kernels_res:
                if func:
                    accum += func(kernel_res[stat_key])
                else:
                    accum += kernel_res[stat_key]
        except Exception as e:
            print(f"Exception: {e}")
            print(f"ERROR: {app} {stat_key} {kernel_res}")
            exit(-1)
        if avg:
            accum /= len(kernels_res)
        Y.append(accum)
        
    return X, Y

# global var
overwrite = False
app_filter = ''
def draw_error(stat, save_img, draw_kernel=False, sim_res_func=None, error_text=True, avg=False, hw_stat="", abs=False):
    '''
    stat: the stat to compare (simulate res and hw res)
    error_text: show error text on the bar
    sim_res_func: function to process sim res
    hw_stat: don't use key map, force hw_stat key
    '''
    global overwrite, app_filter
    # save_img_path = os.path.join(args.output_dir, save_img)
    save_img_path = os.path.join(os.getcwd(), save_img)
    if os.path.exists(save_img_path) and not overwrite:
        return
    if not os.path.exists(os.path.dirname(save_img_path)):
        os.makedirs(os.path.dirname(save_img_path))
    print(f"draw error {'kernel' if draw_kernel else 'app'} {stat}: {save_img[-60:]}")
    
    def bar_overlay(ax, bars, x):
        i = 0
        for bar in bars:
            height = bar.get_height()  # 获取条形的高度（即对应的数值）
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height, f'{x[i]:.2f}', 
                    ha='center', va='bottom', fontsize=8, rotation=-90)
            i += 1
    
    hw_stat_key = key_map[stat] if not hw_stat else hw_stat
    scale = 1
    if type(hw_stat_key) == list:
        hw_stat_key, scale = hw_stat_key
    if not draw_kernel:
        x1, y1 = get_app_stat(hw_res, hw_stat_key, app_filter=app_filter, func=lambda x: x*scale, avg=avg)
        _, y2 = get_app_stat(sim_res, stat, app_filter=app_filter, func=sim_res_func, avg=avg)
    else:
        x1, y1 = get_kernel_stat(hw_res, hw_stat_key, app_filter=app_filter, func=lambda x: x*scale)
        _, y2 = get_kernel_stat(sim_res, stat, app_filter=app_filter, func=sim_res_func)

    if abs:
        error_y = np.abs(np.array(y2) - np.array(y1))/np.array(y1)
    else:
        error_y = (np.array(y2) - np.array(y1))/np.array(y1)
    MAE = np.mean(np.abs(error_y))
    t = np.array(y2) - np.array(y1)
    RMSE = np.sqrt(np.mean(t**2))
    NRMSE = RMSE/np.mean(np.abs(y1))
    NRMSE_max_min = RMSE/(np.max(y1)-np.min(y1))
    
    # correlation
    corr = np.corrcoef(y1, y2)[0, 1]
    
    fig, ax = plt.subplots()
    bars = ax.bar(x1, error_y)
    if error_text:
        bar_overlay(ax, bars, error_y)
    
    # ax.tick_params(axis='x', labelsize=14)
    plt.xticks(rotation=-90, fontsize=8)
    fig.subplots_adjust(bottom=0.4)
    ax.set_ylabel("Error")
    ax.set_xlabel("app")
    ax.set_title(f"{stat} Error, corr={corr:.2f}, MAE={MAE:.2f}, NRMSE={NRMSE:.2f}")
    # plt.show()
    fig.savefig(save_img_path)
    plt.close(fig)

def draw_side2side(stat, save_img, draw_kernel=False, sim_res_func=None, avg=True, hw_stat=""):
    global overwrite, app_filter
    # save_img_path = os.path.join(args.output_dir, save_img)
    save_img_path = os.path.join(os.getcwd(), save_img)
    if os.path.exists(save_img_path) and not overwrite:
        return
    if not os.path.exists(os.path.dirname(save_img_path)):
        os.makedirs(os.path.dirname(save_img_path))
    print(f"draw s2s {'kernel' if draw_kernel else 'app'} {stat}: {save_img[-60:]}")
    
    hw_stat_key = key_map[stat] if not hw_stat else hw_stat
    scale = 1
    if type(hw_stat_key) == list:
        hw_stat_key, scale = hw_stat_key
    if not draw_kernel:
        x1, y1 = get_app_stat(hw_res, hw_stat_key, app_filter=app_filter, func=lambda x: x*scale, avg=avg)
        _, y2 = get_app_stat(sim_res, stat, app_filter=app_filter, func=sim_res_func, avg=avg)
    else:
        x1, y1 = get_kernel_stat(hw_res, hw_stat_key, app_filter=app_filter, func=lambda x: x*scale)
        _, y2 = get_kernel_stat(sim_res, stat, app_filter=app_filter, func=sim_res_func)
    
    corr = np.corrcoef(y1, y2)[0, 1]
    MAE = np.mean(np.abs(np.array(y2) - np.array(y1))/np.array(y1))
    t = np.array(y2) - np.array(y1)
    RMSE = np.sqrt(np.mean(t**2))
    NRMSE = RMSE/np.mean(np.abs(y1))
    NRMSE_max_min = RMSE/(np.max(y1)-np.min(y1))
        
    N = len(x1)
    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35       # the width of the bars
    fig, ax = plt.subplots()
        
    rects1 = ax.bar(ind, y2, width, color='b', label="sim")
    xtra_space = 0.05
    rects2 = ax.bar(ind + width + xtra_space , y1, width, color='orange', label="hw")
    
    ax.set_xticks(ind+width+xtra_space)
    ax.set_xticklabels( x1 )
    
    # add some text for labels, title and axes ticks
    ax.set_xlabel("app")
    ax.set_ylabel(stat)
    ax.set_title(f"{stat} side by side")
    ax.set_title(f"{stat} s2s, corr={corr:.2f}, MAE={MAE:.2f}, NRMSE={NRMSE:.2f}")
    
    plt.xticks(rotation=-90)

    ax.legend()
    fig.savefig(save_img_path)
    plt.close(fig)

def truncate_kernel(sim_res, num):
    sim_res_new = {}
    for app, kernels_res in sim_res.items():
        sim_res_new[app] = kernels_res[:num]
    return sim_res_new

def find_common(sim_res, hw_res):
    # proc hw_res
    for app, kernels_res in hw_res.items():
        for kernel_res in kernels_res:
            kernel_res["dram_total_transactions"] = kernel_res["dram_read_transactions"] + kernel_res["dram_write_transactions"]

    # found common
    for app in hw_res.copy():
        if app not in sim_res:
            del hw_res[app]

    # keep same key order
    common_apps = hw_res.keys()
    sim_res = {app: sim_res[app] for app in common_apps}
    return sim_res, hw_res

def filter_res(res, app_arg_filtered_list):
    for app in res.copy():
        if app not in app_arg_filtered_list:
            del res[app]
    return res
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='draw_1 support: 1) error bar or side by side bar cmp with hw 2) apps or kernels granularity 3) all apps or part of it. we can combine them to draw what we want'
    )
    parser.add_argument("-S", "--sim_res",
                        required=True)
    parser.add_argument("-H", "--hw_res",
                        required=True)
    parser.add_argument("-o", "--output_dir",
                        default="tmp/draw_1/")
    parser.add_argument("-c", "--limit_kernel_num",
                        type=int,
                        default=300,
                        help="PPT-GPU only trace max 300 kernel, the hw trace we also truncate first 300 kernel. So GIMT also should truncate")
    parser.add_argument("-B", "--benchmark_list",
                        help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for the benchmark suite names.",
                        default="")
    parser.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
    parser.add_argument('-d', '--dir-name', default='', help='dir name to save image, default "app" and "kernel"')
    parser.add_argument("command", choices=["app", "kernel", "kernel_by_app", "app_by_bench", "single"], help="draw app or kernel. app: to get overview error of cycle, memory performance and etc. at granurality of apps. kernel: draw all error bar in granurality of kernel. single: draw seperate app in single dir, it's useful when we want to get single app info mation")

    args = parser.parse_args()
    
    print("=====================")
    print("Start draw_1")
    print("=====================")
    
    from common import *
    apps = gen_apps_from_suite_list(args.benchmark_list)
    app_and_arg_list = get_app_arg_list(apps)
    app_arg_filtered_list = filter_app_list(app_and_arg_list, args.app_filter)
    # print(f"app_arg_filtered_list: {app_arg_filtered_list}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.hw_res, 'r') as f:
        hw_res = json.load(f)

    with open(args.sim_res, 'r') as f:
        sim_res = json.load(f)
    
    sim_res = filter_res(sim_res, app_arg_filtered_list)
    hw_res = filter_res(hw_res, app_arg_filtered_list)
    sim_res = truncate_kernel(sim_res, args.limit_kernel_num)
    sim_res, hw_res = find_common(sim_res, hw_res)

    run_dir = os.getcwd()
    os.chdir(args.output_dir)
    
    if args.command=="app":
        print(f"\ncommand: {args.command}:")
        overwrite = True
        app_filter = args.app_filter
        args.dir_name = args.dir_name if args.dir_name else args.command
        os.makedirs(args.dir_name, exist_ok=True)  # save image in app dir
        os.chdir(args.dir_name)
        
        draw_error("warp_inst_executed", "error_1_warp_inst_executed.png")
        draw_error("achieved_occupancy", "error_2_app_occupancy_error.png", sim_res_func = lambda x: x/100)
        
        draw_error("ipc", "error_3_ipc.png")
        
        # draw_error("gpu_active_cycle_max", "error_3_gpu_active_cycle_max.png")
        # draw_error("sm_active_cycles_sum", "error_3_sm_active_cycles_sum.png")
        # draw_error("sm_elapsed_cycles_sum", "error_3_sm_elapsed_cycles_sum.png")
        
        draw_error("my_gpu_active_cycle_max", "error_4_my_gpu_active_cycle_max.png")
        draw_error("my_sm_active_cycles_sum", "error_4_my_sm_active_cycles_sum.png")
        draw_error("my_sm_elapsed_cycles_sum", "error_4_my_sm_elapsed_cycles_sum.png")
        
        draw_error("l1_hit_rate",           "error_6_l1_hit_rate.png")
        draw_error("l2_hit_rate",           "error_6_l2_hit_rate.png")
        draw_error("gmem_read_requests",    "error_6_gmem_read_requests.png")
        draw_error("gmem_write_requests",   "error_6_gmem_write_requests.png")
        draw_error("gmem_read_trans",       "error_6_gmem_read_trans.png")
        draw_error("gmem_write_trans",      "error_6_gmem_write_trans.png")
        draw_error("l2_read_trans",         "error_6_l2_read_trans.png")
        draw_error("l2_write_trans",        "error_6_l2_write_trans.png")
        draw_error("dram_total_trans",      "error_6_dram_total_trans.png")
        
        #### draw side2side
        draw_side2side("warp_inst_executed",    "bar_1_warp_inst_executed.png")
        draw_side2side("achieved_occupancy",    "bar_2_app_occupancy_error.png", sim_res_func = lambda x: x/100)
        
        draw_side2side("ipc",  "bar_3_ipc.png")
        
        # draw_side2side("gpu_active_cycle_max",  "bar_3_gpu_active_cycle_max.png")
        # draw_side2side("sm_elapsed_cycles_sum", "bar_3_sm_elapsed_cycles_sum.png")
        draw_side2side("my_gpu_active_cycle_max",  "bar_4_my_gpu_active_cycle_max.png")
        draw_side2side("my_sm_elapsed_cycles_sum", "bar_4_my_sm_elapsed_cycles_sum.png")
        
        draw_side2side("l1_hit_rate", "bar_6_l1_hit_rate.png")
        draw_side2side("l2_hit_rate", "bar_6_l2_hit_rate.png")
        draw_side2side("gmem_read_requests",    "bar_6_gmem_read_requests.png")
        draw_side2side("gmem_write_requests",   "bar_6_gmem_write_requests.png")
        draw_side2side("gmem_read_trans",       "bar_6_gmem_read_trans.png")
        draw_side2side("gmem_write_trans",      "bar_6_gmem_write_trans.png")
        draw_side2side("l2_read_trans",         "bar_6_l2_read_trans.png")
        draw_side2side("l2_write_trans",        "bar_6_l2_write_trans.png")
        draw_side2side("dram_total_trans",      "bar_6_dram_total_trans.png")
    elif args.command=="app_by_bench":
        overwrite = True
        args.dir_name = args.dir_name if args.dir_name else args.command
        os.makedirs(args.dir_name, exist_ok=True)  # save image in seperate dir
        os.chdir(args.dir_name)
        
        print(f"\ncommand: {args.command}:")
        # get all bench
        app_list_all = sim_res.keys()
        benchs = set()
        for app_arg in app_list_all:
            try:
                benchs.add(suite_info['map'][app_arg][0])
            except:
                print(f"Warning: {app_arg} not found in suite_info, skip")
        # set each bench as filter
        for bench in benchs:
            app_filter = bench
            draw_error("warp_inst_executed", f"{bench}_error_1_warp_inst_executed.png")
            draw_error("my_gpu_active_cycle_max", f"{bench}_error_4_my_gpu_active_cycle_max.png")
            draw_side2side("my_gpu_active_cycle_max", f"{bench}_bar_4_my_gpu_active_cycle_max.png")
            draw_error("l1_hit_rate",           f"{bench}_error_6_l1_hit_rate.png")
            draw_side2side("l1_hit_rate", f"{bench}_bar_6_l1_hit_rate.png")
            draw_error("l2_hit_rate",           f"{bench}_error_6_l2_hit_rate.png")
            draw_side2side("l2_hit_rate", f"{bench}_bar_6_l2_hit_rate.png")
    
    elif args.command == 'kernel':
        print(f"\ncommand: {args.command}:")
        args.dir_name = args.dir_name if args.dir_name else args.command
        os.makedirs(args.dir_name, exist_ok=True)  # save image in seperate dir
        os.chdir(args.dir_name)
        
        overwrite = True
        app_filter = args.app_filter
        draw_error("my_gpu_active_cycle_max", "error_4_my_gpu_active_cycle_max.png", draw_kernel=True)
        draw_side2side("my_gpu_active_cycle_max", f"bar_4_my_gpu_active_cycle_max.png", draw_kernel=True)
    elif args.command == 'kernel_by_app':
        print(f"\ncommand: {args.command}:")
        args.dir_name = args.dir_name if args.dir_name else args.command
        os.makedirs(args.dir_name, exist_ok=True)  # save image in seperate dir
        os.chdir(args.dir_name)
        
        app_list_all = sim_res.keys()
        app_list = filter_app_list(app_list_all, args.app_filter)
        for i,app_arg in enumerate(app_list):
            app_filter=app_arg  # set global filter to single app
            
            app_name_safe = app_arg.replace('/', '_')
            draw_error("my_gpu_active_cycle_max", f"error_{app_name_safe}_4_my_gpu_active_cycle_max.png", draw_kernel=True)
            draw_side2side("my_gpu_active_cycle_max", f"bar_{app_name_safe}_4_my_gpu_active_cycle_max.png", draw_kernel=True)
            draw_error("warp_inst_executed", f"error_{app_name_safe}_1_warp_inst_executed.png", draw_kernel=True)
            draw_side2side("warp_inst_executed", f"bar_{app_name_safe}_1_my_warp_inst_executed.png", draw_kernel=True)
    elif args.command == 'single':
        print(f"\ncommand: {args.command}:")
        overwrite = True
        app_list_all = sim_res.keys()
        app_list = filter_app_list(app_list_all, args.app_filter)  # convert coord filter to app_and_arg filter
        print(f"will draw: {app_list}")
        for i,app_arg in enumerate(app_list):
            os.chdir(args.output_dir)
            os.makedirs(app_arg, exist_ok=True)
            os.chdir(app_arg)
            
            app_filter=app_arg  # set global filter to single app
            draw_error("my_gpu_active_cycle_max", f"error_4_my_gpu_active_cycle_max.png", draw_kernel=True)
            draw_side2side("my_gpu_active_cycle_max", f"bar_4_my_gpu_active_cycle_max.png", draw_kernel=True)
            
            draw_error("achieved_occupancy", f"error_2_app_occupancy.png", draw_kernel=True, sim_res_func = lambda x: x/100)
            draw_side2side("achieved_occupancy", f"bar_2_app_occupancy.png", draw_kernel=True, sim_res_func = lambda x: x/100)
            draw_error("warp_inst_executed", f"error_1_warp_inst_executed.png", draw_kernel=True)
            draw_side2side("warp_inst_executed", f"bar_1_my_warp_inst_executed.png", draw_kernel=True)
            
            draw_side2side("l1_hit_rate", f"bar_6_l1_hit_rate.png", draw_kernel=True)
            draw_side2side("l2_hit_rate", f"bar_6_l2_hit_rate.png", draw_kernel=True)
    os.chdir(run_dir)