
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys, os
curr_dir = os.path.dirname(__file__)
par_dir = os.path.dirname(curr_dir)
sys.path.insert(0, os.path.abspath(par_dir))
from common import *

def bar_overlay(ax, bars, x):
    i = 0
    for bar in bars:
        height = bar.get_height()  # 获取条形的高度（即对应的数值）
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height, f'{x[i]:.2f}', 
                ha='center', va='bottom', fontsize=8, rotation=-90)
        i += 1
    
def draw_bar(fig, ax, x, y_list, labels, legend=True):
    '''
    x: x label
    y_list: list of y
    labels: y label
    errors: error on each bar
    '''
    
    # stack bar
    accum = np.zeros(len(y_list[0]))
    for i,y in enumerate(y_list):
        bars = ax.bar(x, y, bottom=accum, label=labels[i])
        accum += np.array(y)
    
    if legend:
        ax.legend(
            fontsize='small',       # 缩小字体
            markerscale=0.7,        # 缩小标记的大小
            borderpad=0.5,          # 减少图例框内的填充
            labelspacing=0.3,       # 缩小标签之间的间距
        )
    
    ax.set_title(f'CPI stack')

    # xticks
    ax.set_xlabel('kernel')
    ax.set_ylabel('CPI')
    # tick_labels = ax.get_xticklabels()
    # ax.set_xticklabels(tick_labels, rotation=-90)
    plt.xticks(rotation=-90)
    # fig.subplots_adjust(bottom=0.4)

    # fig.tight_layout()

def draw_bar_side2side(fig, ax, x1, y1_list, x2, y2_list, labels, legend=True):
    N = len(x1)
    ind = np.arange(N) + .15 # the x locations for the groups
    width = 0.35       # the width of the bars
    
    # stack bar
    color_list = []
    accum1 = np.zeros(N)
    for i,y1 in enumerate(y1_list):
        bars = ax.bar(ind, y1, bottom=accum1, width=width, label=labels[i])
        color_list.append(bars[0].get_facecolor())
        accum1 += np.array(y1)

    xtra_space = 0.05
    accum2 = np.zeros(N)
    for i,y2 in enumerate(y2_list):
        bars = ax.bar(ind + width + xtra_space, y2, bottom=accum2, width=width, color=color_list[i], label=labels[i])
        accum2 += np.array(y2)
    
    # Set x-ticks and labels
    ax.set_xticks(ind + (width + xtra_space) / 2)
    ax.set_xticklabels(x1)
    
    if legend:
        ax.legend(labels, framealpha=0)
    
    # xticks
    ax.set_xlabel('kernel')
    ax.set_ylabel('CPI')
    # tick_labels = ax.get_xticklabels()
    # ax.set_xticklabels(tick_labels, rotation=-90)
    plt.xticks(rotation=-90)
    # fig.subplots_adjust(bottom=0.4)

def get_stack_data(json_data, app_list=''):
    all_app_list = json_data.keys()
    app_list = filter_app_list(all_app_list, app_list)
    
    multiple_subcore = False
    num_subplots = 1
    x = []
    labels = []
    app_res = json_data[app_list[0]]
    for j,kernel_res in enumerate(app_res):
        if type(kernel_res) == list:
            multiple_subcore = True
            num_subplots = len(kernel_res)
        kernel_first_subcore_res = kernel_res[0] if multiple_subcore else kernel_res
        # get labels
        for k, v in kernel_first_subcore_res.items():
            if type(v) != dict:  # avoid debug info dict
                labels.append(k)
        break
    
    def get(d, y_list):
        for k, v in d.items():
            if type(v) != dict:
                y_list[labels.index(k)].append(v)
                        
    Y_list = [[[] for i in range(len(labels))] for j in range(num_subplots)]
    for i,app_arg in enumerate(app_list):
        app = app_arg.split('/')[0]
        app_res = json_data[app_arg]
        for j,kernel_res in enumerate(app_res):
            x.append(f"{app}-{j}")
            if multiple_subcore:
                for k,subcore_res in enumerate(kernel_res):
                    get(subcore_res, Y_list[k])
            else:
                get(kernel_res, Y_list[0])
    return x, Y_list, labels

def draw_cpi_stack(save_img, app_list='', draw_error=False, draw_subcore=False):
    global overwrite
    save_img_path = os.path.join(os.getcwd(), save_img)
    if os.path.exists(save_img_path) and not overwrite:
        return
    if not os.path.exists(os.path.dirname(save_img_path)):
        os.makedirs(os.path.dirname(save_img_path))
    
    x, Y_list, labels = get_stack_data(sim_res, app_list=app_list)
    print(f"{save_img} len(x): {len(x)}")
    
    is_subcore = len(Y_list)>1
    num_subplots = len(Y_list)
    
    fig,ax = plt.subplots()
    if is_subcore:
        draw_bar(fig,ax, x, Y_list[-1], labels)
    else:
        draw_bar(fig,ax, x, Y_list[0], labels)
    image_path = save_img
    fig.savefig(image_path)
    plt.close(fig)
    
    if is_subcore and draw_subcore:
        fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)

        # draw subplots
        save_img = save_img.replace('.png', '_subplots.png')
        for i in range(4):
            ax = axs[i//2, i%2]
            draw_bar(fig, ax, x, Y_list[i], labels, legend=(i==0))
        image_path = save_img
        fig.savefig(image_path)
        plt.close(fig)

def draw_cpi_stack_side2side(save_img, app_list='', draw_error=False, draw_subcore=False):
    global overwrite
    save_img_path = os.path.join(os.getcwd(), save_img)
    if os.path.exists(save_img_path) and not overwrite:
        return
    if not os.path.exists(os.path.dirname(save_img_path)):
        print(f"Make dir {os.path.dirname(save_img_path)}")
        os.makedirs(os.path.dirname(save_img_path))
    x, Y_list, labels = get_stack_data(sim_res, app_list=app_list)
    x2, Y_list2, _ = get_stack_data(sim_res2, app_list=app_list)
    y_list2 = Y_list2[0]
    print(f"draw cpi stack {save_img[-60:]}: {len(x)}")
    
    
    is_subcore = len(Y_list)>1
    num_subplots = len(Y_list)
    
    fig,ax = plt.subplots()
    if is_subcore:
        draw_bar_side2side(fig,ax, x, Y_list[-1], x2, y_list2, labels)
    else:
        draw_bar_side2side(fig,ax, x, Y_list[0], x2, y_list2, labels)
    image_path = save_img
    fig.savefig(image_path)
    plt.close(fig)
    
    if is_subcore and draw_subcore:
        fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)

        # draw subplots
        save_img = save_img.replace('.png', '_subplots.png')
        for i in range(4):
            ax = axs[i//2, i%2]
            draw_bar_side2side(fig,ax, x, Y_list[i], x2, y_list2, labels)
        image_path = save_img
        fig.savefig(image_path)
        plt.close(fig)

def draw_cpi_stack_subplot_s2s(save_img, app_list='', draw_error=False):
    global overwrite
    save_img_path = os.path.join(os.getcwd(), save_img)
    if os.path.exists(save_img_path) and not overwrite:
        return
    if not os.path.exists(os.path.dirname(save_img_path)):
        print(f"Make dir {os.path.dirname(save_img_path)}")
        os.makedirs(os.path.dirname(save_img_path))
    x, Y_list, labels1 = get_stack_data(sim_res, app_list=app_list)
    x2, Y_list2, labels2 = get_stack_data(sim_res2, app_list=app_list)
    y_list2 = Y_list2[0]
    print(f"draw subplot sidebyside {save_img[-60:]}: {len(x)}")

    fig, axs = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)

    draw_bar(fig, axs[0], x, Y_list[-1], labels1)
    draw_bar(fig, axs[1], x2, y_list2, labels2)

    image_path = save_img
    fig.savefig(image_path)
    plt.close(fig)
    
def check_app_kernel_num(res, print_num=False):
    for app, app_res in res.items():
        if len(app_res) > args.limit_kernel_num:
            print(f"{app} kernel num {len(app_res)} > {args.limit_kernel_num}")
            exit(1)
        if print_num:
            print(f"{app}: {len(app_res)}")

from draw_1 import truncate_kernel,get_kernel_stat,find_common,filter_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script draw CPI stack image, it support: 1) draw single or, side by side campare with second result'
    )
    parser.add_argument("-S", "--sim_res",
                        help="first sim result (left)")
    parser.add_argument("-R", "--sim_res2",
                        help="second sim result (right)")
    parser.add_argument("-H", "--hw_res",
                        help="hw result, used to caculate error")
    parser.add_argument("-o", "--output_dir",
                        default="tmp/draw_cpi_stack/")
    parser.add_argument("-F", "--app-filter", default="", help="filter apps. e.g. regex:.*-rodinia-2.0-ft, [suite]:[exec]:[count]")
    parser.add_argument("-c", "--limit_kernel_num",
                        type=int,
                        default=300,
                        help="PPT-GPU only trace max 300 kernel, the hw trace we also truncate first 300 kernel. So GIMT also should truncate")
    # parser.add_argument("--subdir",
    #                     default="cpi_single",
    #                     help="subdir to save the image (used when not draw side by side)")
    parser.add_argument("--s2s", action="store_true", help="draw side by side")
    parser.add_argument("--draw-subcore", action="store_true", help="draw subcore if have")
    
    parser.add_argument("--seperate-dir", action="store_true", help="draw app in sperate folder, useful when draw single")
    parser.add_argument("--not-seperate-dir-by-bench", dest='seperate_dir_by_bench', action="store_false", help="not draw app in sperate bench folder")
    parser.add_argument("--subplot-s2s", action="store_true", help="draw subplot side by side, used when two stack figure have different labels")
    parser.add_argument("--subdir", default='cpi_warp', help="draw single cpi stack, we store in subdir to distinguish")
    
    args = parser.parse_args()
    
    print("Start draw cpi stack py")
    
    from common import *
    apps = gen_apps_from_suite_list()
    app_and_arg_list = get_app_arg_list(apps)
    app_arg_filtered_list = filter_app_list(app_and_arg_list, args.app_filter)
    # print(f"app_arg_filtered_list: {app_arg_filtered_list}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.sim_res, 'r') as f:
        sim_res = json.load(f)
        sim_res = truncate_kernel(sim_res, args.limit_kernel_num)
        print("\nsim res1 info:")
        check_app_kernel_num(sim_res, print_num=False)

    if args.sim_res2:
        with open(args.sim_res2, 'r') as f:
            sim_res2 = json.load(f)
        sim_res2 = truncate_kernel(sim_res2, args.limit_kernel_num)
        print("\nsim res2 info:")
        check_app_kernel_num(sim_res2, print_num=False)
    
    sim_res = filter_res(sim_res, app_arg_filtered_list)
    sim_res2 = filter_res(sim_res2, app_arg_filtered_list)
    
    print("\nDraw:")
    run_dir = os.getcwd()
    os.chdir(args.output_dir)
    
    overwrite = True
    app_list_all = sim_res.keys()
    app_list = filter_app_list(app_list_all, args.app_filter)  # convert coord filter to app_and_arg filter
    # print(app_list)
        
    if args.s2s:
        if not args.sim_res2:
            print("s2s need two sim res")
            exit(1)

        for i,app_arg in enumerate(app_list):
            if args.seperate_dir_by_bench:
                try:
                    bench = suite_info['map'][app_arg][0]
                except:
                    print(f"Warning: {app_arg} not found in suite_info, skip")
                    continue
                os.chdir(args.output_dir)
                sep_dir = f"cpi_s2s_{bench}"
                os.makedirs(sep_dir, exist_ok=True)
                os.chdir(sep_dir)
            
            if args.seperate_dir:
                os.chdir(args.output_dir)
                os.makedirs(app_arg, exist_ok=True)
                os.chdir(app_arg)
                
            app_name_safe = app_arg.replace('/', '_')
            if args.subplot_s2s:
                draw_cpi_stack_subplot_s2s(f"cpi_s2s_detail/cpi_s2s_{i}_{app_name_safe}.png", app_list=app_arg)
            else:
                draw_cpi_stack_side2side(f"cpi_s2s/cpi_s2s_{i}_{app_name_safe}.png", app_list=app_arg, draw_subcore=args.draw_subcore)
    else:
        # single result cpi stack
        for i,app_arg in enumerate(app_list):
            if args.seperate_dir:
                os.chdir(args.output_dir)
                os.makedirs(app_arg, exist_ok=True)
                os.chdir(app_arg)

            app_name_safe = app_arg.replace('/', '_')
            draw_cpi_stack(f"{args.subdir}/{args.subdir}_{app_name_safe}.png", app_list=app_arg, draw_subcore=args.draw_subcore)

    os.chdir(run_dir)
