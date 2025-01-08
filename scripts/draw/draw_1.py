import argparse
import json
import operator
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
        
        "gpu_active_cycle_max": "active_cycles_sys",
        "my_sm_active_cycles_sum": "active_cycles",
        "sm_elapsed_cycles_sum": "elapsed_cycles_sm",
        
        "ipc": "ipc",
        
        # umem
        "l1_hit_rate": "tex_cache_hit_rate",
        # "l1_hit_rate": ["tex_cache_hit_rate", '/', '100'],
        # "l1_hit_rate_ld": ["global_hit_rate_ld", '/', '100'],
        # "l1_hit_rate_st": ["global_hit_rate_st", '/', '100'],
        # global
        "l1_hit_rate_g": ["global_hit_rate", '/', '100'],
        "l1_hit_rate_ldg": ["global_hit_rate_ld", '/', '100'],
        "l1_hit_rate_stg": ["global_hit_rate_st", '/', '100'],
        
        "l2_hit_rate": "l2_tex_hit_rate",
        # "l2_hit_rate": ["l2_tex_hit_rate", '/', '100'],
        "l2_hit_rate_ld": ["l2_tex_read_hit_rate", '/', '100'],
        "l2_hit_rate_st": ["l2_tex_write_hit_rate", '/', '100'],
        
        "gmem_read_requests": "global_load_requests",
        "gmem_write_requests": "global_store_requests",
        "gmem_read_trans": "gld_transactions",
        "gmem_write_trans": "gst_transactions",
        "l2_read_trans": "l2_read_transactions",
        "l2_write_trans": "l2_write_transactions",
        "dram_total_trans": "dram_total_transactions",
        
        "gmem_tot_reqs": ["global_load_requests", '+', "global_store_requests"],
        "gmem_ld_reqs": "global_load_requests",
        "gmem_st_reqs": "global_store_requests",
        "gmem_tot_sectors": ["gld_transactions", '+', "gst_transactions"],
        "gmem_ld_sectors": "gld_transactions",
        "gmem_st_sectors": "gst_transactions",
        "gmem_ld_diverg": "gld_transactions_per_request",
        
        "l2_tot_trans": ["l2_read_transactions", '+', "l2_write_transactions"],
        "l2_ld_trans": "l2_read_transactions",
        "l2_st_trans": "l2_write_transactions",
        "dram_tot_trans": ["dram_read_transactions", '+', "dram_write_transactions"],
        "dram_ld_trans": "dram_read_transactions",
        "dram_st_trans": "dram_write_transactions",
    }

draw_list = ["l1_hit_rate", "l1_hit_rate_g", "l1_hit_rate_ldg", "l1_hit_rate_stg", "l2_hit_rate", "l2_hit_rate_ld", "l2_hit_rate_st",
            "l2_ld_trans","l2_st_trans","l2_tot_trans","dram_ld_trans","dram_st_trans","dram_tot_trans",
            "gmem_tot_reqs", "gmem_ld_sectors", "gmem_st_sectors", "gmem_tot_sectors", "gmem_ld_diverg"]
draw_list = ["l2_ld_trans","l2_st_trans","l2_tot_trans", "dram_ld_trans","dram_st_trans","dram_tot_trans"]
# 定义操作符的优先级
precedence = {
    '+': 1,
    '-': 1,
    '*': 2,
    '/': 2
}

# 定义每个操作符的操作
operations = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv
}

def infix_to_postfix(tokens):
    """将中缀表达式转换为后缀表达式"""
    output = []
    operators = []
    
    for token in tokens:
        if token.isdigit():  # 操作数直接加入输出
            output.append(token)
        elif token in precedence:  # 操作符
            while (operators and operators[-1] in precedence and
                   precedence[operators[-1]] >= precedence[token]):
                output.append(operators.pop())
            operators.append(token)
        elif token == '(':  # 左括号入栈
            operators.append(token)
        elif token == ')':  # 右括号出栈直到遇到左括号
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()  # 弹出左括号
        else: # symbol
            output.append(token)
    
    while operators:  # 处理剩余的操作符
        output.append(operators.pop())
    
    return output

def evaluate_postfix(postfix, data):
    """计算后缀表达式的值"""
    stack = []
    for token in postfix:
        if token.isdigit():  # 操作数直接入栈
            stack.append(int(token))
        elif token in operations:  # 操作符，弹出两个操作数进行计算
            b = stack.pop()
            a = stack.pop()
            result = operations[token](a, b)
            stack.append(result)
        else: # symbol
            stack.append(data[token])
    return stack[0]

def get_kernel_stat(json_data, stat_key, app_filter='', func=None, key_is_expression=False):
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
            if 'kernel_name' not in kernel_res:
                X.append(f"{app}-{j}")
            else:
                app_short_name = app.replace('_', '')
                app_short_name = app_short_name[:3]+app_short_name[-10:]
                X.append(f"{app_short_name}-{j}-{kernel_res['kernel_name']}")
            # app_short_name = app.replace('_', '')
            # app_short_name = app_short_name[:3]+app_short_name[-10:]
            # X.append(f"{app_short_name}-{j}-{kernel_res['kernel_name']}")
            if key_is_expression:
                value = evaluate_postfix(infix_to_postfix(stat_key), kernel_res)
            else:
                value = kernel_res[stat_key]
            if func:
                value = func(value)
            Y.append(value)

    return np.array(X), np.array(Y)

def get_app_stat(json_data, stat_key, app_filter='', func=None, avg=False, key_is_expression=False):
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
                if key_is_expression:
                    value = evaluate_postfix(infix_to_postfix(stat_key), kernel_res)
                else:
                    value = kernel_res[stat_key]
                if func:
                    value = func(value)
                accum += value
        except Exception as e:
            # print(f"Exception: {e}")
            print(f"error in get_app_stat: {app} {stat_key}")
            raise e
        if avg:
            accum /= len(kernels_res)
        Y.append(accum)
        
    return np.array(X), np.array(Y)

# global var
overwrite = False
app_filter = ''

def draw_helper(msg, stat, save_img, draw_kernel=False, sim_res_func=None, avg=False, hw_stat="", verbose=True):
    global overwrite, app_filter
    save_img_path = os.path.join(os.getcwd(), save_img)
    if os.path.exists(save_img_path) and not overwrite:
        return False, None
    if not os.path.exists(os.path.dirname(save_img_path)):
        os.makedirs(os.path.dirname(save_img_path))
    if verbose:
        print(f"draw {msg} {stat}: {save_img[-60:]}")
    
    hw_stat_key = key_map[stat] if not hw_stat else hw_stat
    key_is_expression = False
    if type(hw_stat_key) == list:
        key_is_expression = True
    if not draw_kernel:
        x1, y1 = get_app_stat(hw_res, hw_stat_key, app_filter=app_filter, avg=avg, key_is_expression=key_is_expression)
        _, y2 = get_app_stat(sim_res, stat, app_filter=app_filter, func=sim_res_func, avg=avg)
    else:
        x1, y1 = get_kernel_stat(hw_res, hw_stat_key, app_filter=app_filter, key_is_expression=key_is_expression)
        _, y2 = get_kernel_stat(sim_res, stat, app_filter=app_filter, func=sim_res_func)

    non_zero_idxs = y1 != 0
    MAE = np.mean(np.abs(y1[non_zero_idxs] - y2[non_zero_idxs])/y1[non_zero_idxs])
    t = y2 - y1
    RMSE = np.sqrt(np.mean(t**2))
    NRMSE = RMSE/np.mean(np.abs(y1))
    NRMSE_max_min = RMSE/(np.max(y1)-np.min(y1))
    # correlation
    corr = np.corrcoef(y1, y2)[0, 1]
    
    return True, [x1, y1, y2, MAE, NRMSE, corr, save_img_path]
    
def draw_error(stat, save_img, draw_kernel=False, sim_res_func=None, error_text=True, avg=False, hw_stat="", abs=False):
    '''
    stat: the stat to compare (simulate res and hw res)
    error_text: show error text on the bar
    sim_res_func: function to process sim res
    hw_stat: don't use key map, force hw_stat key
    '''
    flag, data = draw_helper("error", stat, save_img, draw_kernel, sim_res_func, avg, hw_stat)
    if not flag:
        return
    x1, y1, y2, MAE, NRMSE, corr, save_img_path = data
    # save_img_path = os.path.join(os.getcwd(), save_img)
    # if os.path.exists(save_img_path) and not overwrite:
    #     return
    # if not os.path.exists(os.path.dirname(save_img_path)):
    #     os.makedirs(os.path.dirname(save_img_path))
    # print(f"draw error {'kernel' if draw_kernel else 'app'} {stat}: {save_img[-60:]}")
    
    # hw_stat_key = key_map[stat] if not hw_stat else hw_stat
    # key_is_expression = False
    # if type(hw_stat_key) == list:
    #     key_is_expression = True
    # if not draw_kernel:
    #     x1, y1 = get_app_stat(hw_res, hw_stat_key, app_filter=app_filter, avg=avg, key_is_expression=key_is_expression)
    #     _, y2 = get_app_stat(sim_res, stat, app_filter=app_filter, func=sim_res_func, avg=avg)
    # else:
    #     x1, y1 = get_kernel_stat(hw_res, hw_stat_key, app_filter=app_filter, key_is_expression=key_is_expression)
    #     _, y2 = get_kernel_stat(sim_res, stat, app_filter=app_filter, func=sim_res_func)

    def bar_overlay(ax, bars, x):
        i = 0
        for bar in bars:
            height = bar.get_height()  # 获取条形的高度（即对应的数值）
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height, f'{x[i]:.2f}', 
                    ha='center', va='bottom', fontsize=8, rotation=-90)
            i += 1
    
    error_y = []
    for i in range(len(y1)):
        if y1[i] == 0:
            error_y.append(0)
        else:
            if abs:
                error_y.append(np.abs(y2[i] - y1[i])/y1[i])
            else:
                error_y.append((y2[i] - y1[i])/y1[i])

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

def draw_side2side(stat, save_img, draw_kernel=False, sim_res_func=None, avg=True, hw_stat="", verbose=True):
    flag, data = draw_helper("bar", stat, save_img, draw_kernel, sim_res_func, avg, hw_stat, verbose=verbose)
    if not flag:
        return
    x1, y1, y2, MAE, NRMSE, corr, save_img_path = data
    
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
    fig.subplots_adjust(bottom=0.4)
    ax.set_xlabel("app")
    ax.set_ylabel(stat)
    ax.set_title(f"{stat} side by side")
    ax.set_title(f"{stat} s2s, corr={corr:.2f}, MAE={MAE:.2f}, NRMSE={NRMSE:.2f}")
    
    plt.xticks(rotation=-90)

    ax.legend()
    fig.savefig(save_img_path)
    plt.close(fig)

def draw_correl(stat, save_img, draw_kernel=False, sim_res_func=None, avg=True, hw_stat=""):
    flag, data = draw_helper("correl", stat, save_img, draw_kernel, sim_res_func, avg, hw_stat)
    if not flag:
        return
    x1, y1, y2, MAE, NRMSE, corr, save_img_path = data
        
    fig, ax = plt.subplots()
    ax.scatter(y1, y2, label=f"corr={corr:.2f} error={MAE:.2f}", color='blue')
    
    min_val = min(y1.min(), y2.min())
    max_val = max(y1.max(), y2.max())
    min_val -= 0.1*(max_val - min_val)
    max_val += 0.1*(max_val - min_val)
    
    ax.plot([min_val, max_val], [min_val, max_val], color='red')
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    if max_val / (min_val+1e-6) > 1000:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_aspect('equal', adjustable='box')
    # add some text for labels, title and axes ticks
    ax.set_xlabel(f"HW {stat}")
    ax.set_ylabel(f"Sim {stat}")
    ax.set_title(f"{stat} Correl, corr={corr:.2f}, MAE={MAE:.2f}, NRMSE={NRMSE:.2f}")
    ax.legend()
    fig.savefig(save_img_path)
    plt.close(fig)
    return {
        stat: {
            'MAE': MAE,
            'NRMSE': NRMSE,
            'corr': corr,
        }
    }
    
def truncate_kernel(sim_res, num):
    sim_res_new = {}
    for app, kernels_res in sim_res.items():
        sim_res_new[app] = kernels_res[:num]
    return sim_res_new

def find_common(sim_res, hw_res):
    # proc hw_res
    try:
        for app, kernels_res in hw_res.items():
            for kernel_res in kernels_res:
                kernel_res["dram_total_transactions"] = kernel_res["dram_read_transactions"] + kernel_res["dram_write_transactions"]
    except:
        pass

    # found common
    for app in hw_res.copy():
        if app not in sim_res or len(sim_res[app]) != len(hw_res[app]):
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
    parser.add_argument("command",
                        # choices=["app", "kernel", "kernel_by_app", "app_by_bench", "single", "memory"],
                        help="draw app or kernel. app: to get overview error of cycle, memory performance and etc. at granurality of apps. kernel: draw all error bar in granurality of kernel. single: draw seperate app in single dir, it's useful when we want to get single app info mation")
    parser.add_argument('--gtx1080ti', action='store_true', help='1080ti l1 hit should use tex_cache_hit_rate')
    parser.add_argument('-N', '--not-overwrite', dest='overwrite', action='store_false', help='not overwrite')

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
    
    overwrite = args.overwrite
    if args.command=="app":
        print(f"\ncommand: {args.command}:")
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
        
        draw_error("gpu_active_cycle_max", "error_4_gpu_active_cycle_max.png")
        draw_error("sm_active_cycles_sum", "error_4_my_sm_active_cycles_sum.png")
        draw_error("sm_elapsed_cycles_sum", "error_4_sm_elapsed_cycles_sum.png")
        
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
        draw_side2side("gpu_active_cycle_max",  "bar_4_gpu_active_cycle_max.png")
        draw_side2side("sm_elapsed_cycles_sum", "bar_4_sm_elapsed_cycles_sum.png")
        
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
            draw_error("gpu_active_cycle_max", f"{bench}_error_4_gpu_active_cycle_max.png")
            draw_side2side("gpu_active_cycle_max", f"{bench}_bar_4_gpu_active_cycle_max.png")
            draw_error("l1_hit_rate",           f"{bench}_error_6_l1_hit_rate.png")
            draw_side2side("l1_hit_rate", f"{bench}_bar_6_l1_hit_rate.png")
            draw_error("l2_hit_rate",           f"{bench}_error_6_l2_hit_rate.png")
            draw_side2side("l2_hit_rate", f"{bench}_bar_6_l2_hit_rate.png")
    
    elif args.command == 'kernel':
        print(f"\ncommand: {args.command}:")
        args.dir_name = args.dir_name if args.dir_name else args.command
        os.makedirs(args.dir_name, exist_ok=True)  # save image in seperate dir
        os.chdir(args.dir_name)
        
        app_filter = args.app_filter
        draw_error("gpu_active_cycle_max", "error_4_gpu_active_cycle_max.png", draw_kernel=True)
        draw_side2side("gpu_active_cycle_max", f"bar_4_gpu_active_cycle_max.png", draw_kernel=True)
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
            draw_error("gpu_active_cycle_max", f"error_{app_name_safe}_4_gpu_active_cycle_max.png", draw_kernel=True)
            draw_side2side("gpu_active_cycle_max", f"bar_{app_name_safe}_4_gpu_active_cycle_max.png", draw_kernel=True)
            draw_error("warp_inst_executed", f"error_{app_name_safe}_1_warp_inst_executed.png", draw_kernel=True)
            draw_side2side("warp_inst_executed", f"bar_{app_name_safe}_1_my_warp_inst_executed.png", draw_kernel=True)
    elif args.command == 'single':
        print(f"\ncommand: {args.command}:")
        app_list_all = sim_res.keys()
        app_list = filter_app_list(app_list_all, args.app_filter)  # convert coord filter to app_and_arg filter
        print(f"will draw: {app_list}")
        for i,app_arg in enumerate(app_list):
            os.chdir(args.output_dir)
            os.makedirs(app_arg, exist_ok=True)
            os.chdir(app_arg)
            
            app_filter=app_arg  # set global filter to single app
            draw_error("gpu_active_cycle_max", f"error_4_gpu_active_cycle_max.png", draw_kernel=True)
            draw_side2side("gpu_active_cycle_max", f"bar_4_gpu_active_cycle_max.png", draw_kernel=True)
            
            draw_error("achieved_occupancy", f"error_2_app_occupancy.png", draw_kernel=True, sim_res_func = lambda x: x/100)
            draw_side2side("achieved_occupancy", f"bar_2_app_occupancy.png", draw_kernel=True, sim_res_func = lambda x: x/100)
            draw_error("warp_inst_executed", f"error_1_warp_inst_executed.png", draw_kernel=True)
            draw_side2side("warp_inst_executed", f"bar_1_my_warp_inst_executed.png", draw_kernel=True)
            
            draw_side2side("l1_hit_rate", f"bar_6_l1_hit_rate.png", draw_kernel=True)
            draw_side2side("l2_hit_rate", f"bar_6_l2_hit_rate.png", draw_kernel=True)
    elif args.command=='memory':
        print(f"\ncommand: {args.command}:")
        args.dir_name = args.dir_name if args.dir_name else args.command
        os.makedirs(args.dir_name, exist_ok=True)  # save image in seperate dir
        os.chdir(args.dir_name)
        # get all bench
        app_list_all = sim_res.keys()
        benchs = set()
        for app_arg in app_list_all:
            try:
                benchs.add(suite_info['map'][app_arg][0])
            except:
                print(f"Warning: {app_arg} not found in suite_info, skip")
        
        # total app
        draw_correl("l1_hit_rate", f"6_l1_hit_rate_correl.png")
        draw_correl("l2_hit_rate", f"6_l2_hit_rate_correl.png")
        draw_correl("l1_hit_rate", f"6_l1_hit_rate_correl_all_kernel.png", draw_kernel=True)
        draw_correl("l2_hit_rate", f"6_l2_hit_rate_correl_all_kernel.png", draw_kernel=True)
        
        for i, stat in enumerate(["gmem_tot_reqs", "gmem_tot_sectors", "l1_hit_rate", "l1_hit_rate_ldg",
                                  "l2_ld_trans", "l2_st_trans", "l2_tot_trans", "l2_hit_rate", "l2_hit_rate_ld",
                                    "dram_ld_trans", "dram_st_trans", "dram_tot_trans"]):
            draw_side2side(stat, f"all_bar_{i}_{stat}.png")
            draw_correl(stat, f"all_corr_{i}_{stat}.png")
        
        # by bench
        # MAE_res = {'apps': {}, 'kernels': {}}
        for bench in benchs:
            # apps_res = {}  # app level
            # kernels_res = {} # kernel level
            # MAE_res['apps'][bench] = apps_res
            # MAE_res['kernels'][bench] = kernels_res
            
            # set each bench as filter
            app_filter = bench
            
            for i, stat in enumerate(draw_list):
                try:
                    draw_side2side(stat, f"{bench}_{i}_{stat}_bar.png")
                    app_res = draw_correl(stat, f"{bench}_{i}_{stat}_correl.png")
                    # kernel_res = draw_correl(stat, f"{bench}_{i}_{stat}_correl_all_kernel.png", draw_kernel=True)  # not avg
                    
                    # apps_res.update(app_res)
                    # kernels_res.update(kernel_res)
                except:
                    print(f"ERROR: draw memory {app_arg} {stat} failed")
                    continue
        
        # import csv
        # # write csv
        # csvfile = open('memory_res.csv', 'w', newline='')
        # csv_writer = csv.writer(csvfile, delimiter=',',
        #                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # csv_writer.writerow(['level', 'bench', 'stat', 'MAE', 'NRMSE', 'corr'])
        # for level in ['apps', 'kernels']:
        #     for bench in benchs:
        #         for stat in MAE_res[level][bench].keys():
        #             csv_writer.writerow([level, bench, stat, MAE_res[level][bench][stat]['MAE'], MAE_res[level][bench][stat]['NRMSE'], MAE_res[level][bench][stat]['corr']])
        # csvfile.close()
        
    elif args.command == 'memory_kernels':
        print(f"\ncommand: {args.command}:")
        args.dir_name = args.dir_name if args.dir_name else args.command
        os.makedirs(args.dir_name, exist_ok=True)  # save image in seperate dir
        os.chdir(args.dir_name)
        cwd = os.getcwd()
        
        app_list_all = sim_res.keys()
        for i,app_arg in enumerate(app_list_all):
            app_filter=app_arg  # set global filter to single app
            
            app_name_safe = app_arg.replace('/', '_')
            prefix = f"{i:2d}_{app_name_safe}"
            os.chdir(cwd)
            os.makedirs(prefix, exist_ok=True)  # save image in seperate dir
            os.chdir(prefix)
            
            for i, stat in enumerate(draw_list):
                try:
                    draw_side2side(stat, f"{i}_{stat}_bar.png", draw_kernel=True, verbose=False)
                    # draw_correl(stat, f"{stat}_correl.png", draw_kernel=True)
                except InterruptedError as e:
                    print("InterruptedError: ", e)
                    exit()
                except:
                    print(f"Warnning: {app_arg} {stat} failed")
                    continue
    else:
        print(f"ERROR: command {args.command} not supported")
    os.chdir(run_dir)