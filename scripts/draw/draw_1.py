import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


'''
{
    app: {
        [
            # kernel 1
            {k: v, k2: v2, ...},
            # kernel 2
        ]
    }
}

kernel should contain 'kernel_name'
'''

key_map_dict = {
    "PPT-GPU": {
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
    },
    "GIMT-titanv": {
        "warp_inst_executed": "inst_executed",
        
        "sm0_elapsed_cycles": "active_cycles_sys",
        "sm_elapsed_cycles_sum": "elapsed_cycles_sm", # elapsed clocks on SM
        
        "l1_hit_rate": "global_hit_rate",  # 实际应该和这个对比
        # "l1_hit_rate": "tex_cache_hit_rate",
        "l2_hit_rate": "l2_tex_hit_rate",
        "l1_read_requests": "global_load_requests",
        "l1_write_requests": "global_store_requests",
        "l1_read_trans": "gld_transactions",
        "l1_write_trans": "gst_transactions",
        "l2_read_trans": "l2_read_transactions",
        "l2_write_trans": "l2_write_transactions",
        "dram_total_trans": "dram_total_transactions",
    },
    "GIMT-1080ti": {
        "warp_inst_executed": "inst_executed",
        
        "sm0_elapsed_cycles": ["elapsed_cycles_sm", 1/28], # elapsed_cycles_sm/sm
        "sm_elapsed_cycles_sum": "elapsed_cycles_sm",
        
        # "l1_hit_rate": "global_hit_rate",  # 1080ti 采到的都是 0
        "l1_hit_rate": "tex_cache_hit_rate",
        "l2_hit_rate": "l2_tex_hit_rate",
        "l1_read_requests": "global_load_requests",
        "l1_write_requests": "global_store_requests",
        "l1_read_trans": "gld_transactions",
        "l1_write_trans": "gst_transactions",
        "l2_read_trans": "l2_read_transactions",
        "l2_write_trans": "l2_write_transactions",
        "dram_total_trans": "dram_total_transactions",
    }
}

def get_kernel_stat(json_data, stat_key, app_list='all', func=None):
    '''
    construct X, Y. X: kernel name, Y: stat
    '''
    if app_list == 'all':
        app_list = json_data.keys()
    elif app_list.startswith('['):  # python slice, e.g [0: 10] mean first 10
        local_namespace = {'json_data':json_data}
        exec(f"res = list(json_data.keys()){app_list}", globals(), local_namespace)
        app_list = local_namespace['res']
    else:
        app_list = app_list.split(',')
    
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

def get_app_stat(json_data, stat_key, app_list='all', func=None, avg=False):
    '''
    construct X, Y. X: app name, Y: app stat (sum of kernel stat)
    '''
    if app_list == 'all':
        app_list = json_data.keys()
    elif app_list.startswith('['):  # python slice, e.g [0: 10] mean first 10
        local_namespace = {'json_data':json_data}
        exec(f"res = list(json_data.keys()){app_list}", globals(), local_namespace)
        app_list = local_namespace['res']
    else:
        app_list = app_list.split(',')
    
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

def draw_error(stat, save_img, app_list='all', draw_kernel=False, sim_res_func=None, error_text=True, avg=False, hw_stat="", abs=False):
    '''
    stat: the stat to compare (simulate res and hw res)
    error_text: show error text on the bar
    sim_res_func: function to process sim res
    hw_stat: don't use key map, force hw_stat key
    '''
    global overwrite
    save_img_path = os.path.join(args.output_dir, save_img)
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
        x1, y1 = get_app_stat(hw_res, hw_stat_key, app_list=app_list, func=lambda x: x*scale, avg=avg)
        _, y2 = get_app_stat(sim_res, stat, app_list=app_list, func=sim_res_func, avg=avg)
    else:
        x1, y1 = get_kernel_stat(hw_res, hw_stat_key, app_list=app_list, func=lambda x: x*scale)
        _, y2 = get_kernel_stat(sim_res, stat, app_list=app_list, func=sim_res_func)

    if abs:
        error_y = np.abs(np.array(y2) - np.array(y1))/np.array(y1)
    else:
        error_y = (np.array(y2) - np.array(y1))/np.array(y1)
    avg_error = np.mean(np.abs(error_y))
    
    fig, ax = plt.subplots()
    bars = ax.bar(x1, error_y)
    if error_text:
        bar_overlay(ax, bars, error_y)
    
    # ax.tick_params(axis='x', labelsize=14)
    plt.xticks(rotation=-90, fontsize=8)
    fig.subplots_adjust(bottom=0.4)
    ax.set_ylabel("Error")
    ax.set_xlabel("app")
    ax.set_title(f"{stat} Error, avg abs error={avg_error:.2f}")
    # plt.show()
    fig.savefig(save_img_path)
    plt.close(fig)

def draw_side2side(stat, save_img, app_list='all', draw_kernel=False, sim_res_func=None, avg=True, hw_stat=""):
    global overwrite
    save_img_path = os.path.join(args.output_dir, save_img)
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
        x1, y1 = get_app_stat(hw_res, hw_stat_key, app_list=app_list, func=lambda x: x*scale, avg=avg)
        _, y2 = get_app_stat(sim_res, stat, app_list=app_list, func=sim_res_func, avg=avg)
    else:
        x1, y1 = get_kernel_stat(hw_res, hw_stat_key, app_list=app_list, func=lambda x: x*scale)
        _, y2 = get_kernel_stat(sim_res, stat, app_list=app_list, func=sim_res_func)
    
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

if __name__ == "__main__":
    SELECT = list(key_map_dict.keys())
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument("-S", "--sim_res",
                    default="tmp/res_GIMT.json")
    parser.add_argument("-H", "--hw_res",
                    default="tmp/res_hw.json")
    parser.add_argument("-o", "--output_dir",
                    default="tmp/draw_1/")
    parser.add_argument("-D", "--draw_select",
                        choices=SELECT,
                        default="GIMT-titanv",
                        help=f"draw which res: {SELECT}")
    parser.add_argument("-c", "--limit_kernel_num",
                        type=int,
                        default=300,
                        help="PPT-GPU only trace max 300 kernel, the hw trace we also truncate first 300 kernel. So GIMT also should truncate")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    key_map = key_map_dict[args.draw_select]

    with open(args.hw_res, 'r') as f:
        hw_res = json.load(f)

    with open(args.sim_res, 'r') as f:
        sim_res = json.load(f)
    
    sim_res = truncate_kernel(sim_res, args.limit_kernel_num)
    sim_res, hw_res = find_common(sim_res, hw_res)

    if "GIMT" in args.draw_select:
        overwrite = False
        # very import
        draw_error("warp_inst_executed", "error_warp_inst_executed.png")
        draw_error("sm0_elapsed_cycles", "error_sm0_elapsed_cycles.png")
        draw_error("sm_elapsed_cycles_sum", "error_sm_elapsed_cycles_sum.png")
        draw_error("l1_hit_rate", "error_l1_hit_rate.png", sim_res_func = lambda x: x*100)
        draw_error("l1_hit_rate", "error_l1_hit_rate_tex.png", hw_stat="tex_cache_hit_rate", sim_res_func = lambda x: x*100)
        draw_error("l2_hit_rate", "error_l2_hit_rate.png", sim_res_func = lambda x: x*100)
        draw_error("l1_read_requests", "error_l1_read_requests.png")
        draw_error("l1_write_requests", "error_l1_write_requests.png")
        draw_error("l1_read_trans", "error_l1_read_trans.png")
        draw_error("l1_write_trans", "error_l1_write_trans.png")
        draw_error("l2_read_trans", "error_l2_read_trans.png")
        draw_error("l2_write_trans", "error_l2_write_trans.png")
        draw_error("dram_total_trans", "error_dram_total_trans.png")

        #### draw side2side
        # overwrite = True
        draw_side2side("sm0_elapsed_cycles", "bar_sm0_elapsed_cycles.png")
        draw_side2side("sm_elapsed_cycles_sum", "bar_sm_elapsed_cycles_sum.png")
        
        draw_side2side("l1_hit_rate", "bar_l1_hit_rate.png", sim_res_func = lambda x: x*100)
        draw_side2side("l1_hit_rate", "bar_l1_hit_rate_tex.png", hw_stat="tex_cache_hit_rate", sim_res_func = lambda x: x*100)
        draw_side2side("l2_hit_rate", "bar_l2_hit_rate.png", sim_res_func = lambda x: x*100)
        
        draw_side2side("warp_inst_executed", "bar_warp_inst_executed.png")
        
        #### draw kernel
        # overwrite = True
        # draw_side2side("sm0_elapsed_cycles", "bar-kernel-bfs1-gpu_active_cycle_max.png", app_list='[2:3]', draw_kernel=True)
        # draw_error("sm0_elapsed_cycles", "kernel-b+tree_backprop-gpu_active_cycle_max.png", app_list='[0:2]', draw_kernel=True, abs=False)
        # draw_error("sm0_elapsed_cycles", "kernel-bfs12-gpu_active_cycle_max.png", app_list='[2:4]', draw_kernel=True, abs=False)
        # draw_error("sm0_elapsed_cycles", "kernel-bfs3-gpu_active_cycle_max.png", app_list='[4:5]', draw_kernel=True, abs=False)
        # draw_error("sm0_elapsed_cycles", "kernel-dwt1-gpu_active_cycle_max.png", app_list='[6:7]', draw_kernel=True, abs=False)
        # draw_error("sm0_elapsed_cycles", "kernel-dwt2-gpu_active_cycle_max.png", app_list='[7:8]', draw_kernel=True, abs=False)
        # draw_error("sm0_elapsed_cycles", "kernel-hotspot12-gpu_active_cycle_max.png", app_list='[12:14]', draw_kernel=True, abs=False)
        # draw_error("sm0_elapsed_cycles", "kernel-particle_float-gpu_active_cycle_max.png", app_list='particlefilter_float-rodinia-3.1/_x_128__y_128__z_10__np_1000', draw_kernel=True, abs=False)
        # draw_error("sm0_elapsed_cycles", "kernel-particle_naive-gpu_active_cycle_max.png", app_list='particlefilter_naive-rodinia-3.1/_x_128__y_128__z_10__np_1000', draw_kernel=True, abs=False)
        # draw_error("sm0_elapsed_cycles", "kernel-pathfinder-gpu_active_cycle_max.png", app_list='pathfinder-rodinia-3.1/100000_100_20___result_txt', draw_kernel=True, abs=False)
 
        # all simple kernel app
        overwrite = True
        best_app_list = \
            "backprop-rodinia-2.0-ft/4096___data_result_4096_txt,"\
            "hotspot-rodinia-2.0-ft/30_6_40___data_result_30_6_40_txt,"\
            "heartwall-rodinia-2.0-ft/__data_test_avi_1___data_result_1_txt,"\
            "srad_v2-rodinia-2.0-ft/__data_matrix128x128_txt_0_127_0_127__5_2___data_result_matrix128x128_1_150_1_100__5_2_txt,"\
            "backprop-rodinia-3.1/65536,"\
            "bfs-rodinia-3.1/__data_graph4096_txt,"\
            "dwt2d-rodinia-3.1/__data_192_bmp__d_192x192__f__5__l_3,"\
            "heartwall-rodinia-3.1/__data_test_avi_1,"\
            "hotspot-rodinia-3.1/512_2_2___data_temp_512___data_power_512_output_out,"\
            "hotspot-rodinia-3.1/1024_2_2___data_temp_1024___data_power_1024_output_out,"\
            "pathfinder-rodinia-3.1/100000_100_20___result_txt,"\
            "srad_v1-rodinia-3.1/100_0_5_502_458"
        draw_error("sm0_elapsed_cycles", "kernel_error_sm0_elapsed_cycles.png", app_list=best_app_list)
        draw_error("sm_elapsed_cycles_sum", "kernel_error_sm_elapsed_cycles_sum.png", app_list=best_app_list)
        draw_error("l1_hit_rate", "kernel_error_sm_l1_hit_rate.png", app_list=best_app_list)
        draw_error("l2_hit_rate", "kernel_error_sm_l2_hit_rate.png", app_list=best_app_list)
        
        # app_list = "b+tree-rodinia-3.1/file___data_mil_txt_command___data_command_txt,backprop-rodinia-3.1/65536,"\
        #     "dwt2d-rodinia-3.1/__data_192_bmp__d_192x192__f__5__l_3,dwt2d-rodinia-3.1/__data_rgb_bmp__d_1024x1024__f__5__l_3,"\
        #     "gaussian-rodinia-3.1/_f___data_matrix4_txt,"\
        #     "hotspot-rodinia-3.1/512_2_2___data_temp_512___data_power_512_output_out,hotspot-rodinia-3.1/1024_2_2___data_temp_1024___data_power_1024_output_out,"\
        #     "nn-rodinia-3.1/__data_filelist_4__r_5__lat_30__lng_90,"\
        #     "particlefilter_naive-rodinia-3.1/_x_128__y_128__z_10__np_1000,"\
        #     "pathfinder-rodinia-3.1/100000_100_20___result_txt"
        # draw_error("sm0_elapsed_cycles", "kernel-simple-gpu_active_cycle_max.png", app_list=app_list, draw_kernel=True)
        # draw_error("sm_elapsed_cycles_sum", "kernel-simple-sm_elapsed_cycles_sum.png", app_list=app_list, draw_kernel=True)
        # draw_error("l1_hit_rate", "kernel-simple-l1_hit_rate.png", sim_res_func = lambda x: x*100, app_list=app_list, draw_kernel=True)
        # draw_error("l1_hit_rate", "kernel-simple-l1_hit_rate_global_hit_rate.png", sim_res_func = lambda x: x*100, app_list=app_list, draw_kernel=True)
        # draw_error("l2_hit_rate", "kernel-simple-l2_hit_rate.png", sim_res_func = lambda x: x*100, app_list=app_list, draw_kernel=True)
        # draw_error("l1_read_requests", "kernel-simple-l1_read_requests.png", app_list=app_list, draw_kernel=True)
        # draw_error("l1_write_requests", "kernel-simple-l1_write_requests.png", app_list=app_list, draw_kernel=True)
        # draw_error("l1_read_trans", "kernel-simple-l1_read_trans.png", app_list=app_list, draw_kernel=True)
        # draw_error("l1_write_trans", "kernel-simple-l1_write_trans.png", app_list=app_list, draw_kernel=True)
        # draw_error("l2_read_trans", "kernel-simple-l2_read_trans.png", app_list=app_list, draw_kernel=True)
        # draw_error("l2_write_trans", "kernel-simple-l2_write_trans.png", app_list=app_list, draw_kernel=True)
        # draw_error("dram_total_trans", "kernel-simple-dram_total_trans.png", app_list=app_list, draw_kernel=True)

    if args.draw_select=="PPT-GPU":
        overwrite = False
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
        
        # draw_error("l2_hit_rate", "kernel-bar_6_l2_hit_rate", app_list="gaussian-rodinia-3.1/_f___data_matrix208_txt", draw_kernel=True)
        #### Kernel
        overwrite = False
        # mkdir
        app_list_all = sim_res.keys()
        for i,app_arg in enumerate(app_list_all):
            app_name_safe = app_arg.replace('/', '_')
            draw_error("my_gpu_active_cycle_max", f"4_my_gpu_active_cycle_max/error_4_my_gpu_active_cycle_max_{i}_{app_name_safe}.png", app_list=app_arg, draw_kernel=True)
            draw_side2side("my_gpu_active_cycle_max", f"4_my_gpu_active_cycle_max/bar_4_my_gpu_active_cycle_max_{i}_{app_name_safe}.png", app_list=app_arg, draw_kernel=True)
            
            draw_error("achieved_occupancy", f"2_app_occupancy/error_2_app_occupancy_{i}_{app_name_safe}.png", app_list=app_arg, draw_kernel=True, sim_res_func = lambda x: x/100)
            draw_side2side("achieved_occupancy", f"2_app_occupancy/bar_2_app_occupancy_{i}_{app_name_safe}.png", app_list=app_arg, draw_kernel=True, sim_res_func = lambda x: x/100)
            
            draw_side2side("l1_hit_rate", f"6_l1_hit_rate/bar_6_l1_hit_rate_{i}_{app_name_safe}.png", app_list=app_arg, draw_kernel=True)
            draw_side2side("l2_hit_rate", f"6_l2_hit_rate/bar_6_l2_hit_rate_{i}_{app_name_safe}.png", app_list=app_arg, draw_kernel=True)
        
        # draw_error("my_gpu_active_cycle_max", "kernel_rodinia2_error_4_my_gpu_active_cycle_max.png", app_list='[0:7]', draw_kernel=True)
        # draw_side2side("my_gpu_active_cycle_max", "kernel_rodinia2_bar_4_my_gpu_active_cycle_max.png", app_list='[0:7]', draw_kernel=True)
        # draw_error("achieved_occupancy", "kernel_rodinia2_error_2_app_occupancy_error.png", app_list='[0:7]', draw_kernel=True, sim_res_func = lambda x: x/100)
        # draw_side2side("achieved_occupancy", "kernel_rodinia2_bar_2_app_occupancy_error.png", app_list='[0:7]', draw_kernel=True, sim_res_func = lambda x: x/100)
        