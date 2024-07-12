import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description=''
)
parser.add_argument("--sim_res",
                 default="sim_res.json")
parser.add_argument("--hw_res",
                 default="hw_res.json")
args = parser.parse_args()


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
'''
# X: kernel
# Y: stat

# X: app
# Y: stat (sum of kernel)

key_map = {
    "achieved_occupancy": "achieved_occupancy",
    "warp_inst_executed": "inst_executed",
    
    "gpu_active_cycle_max": "active_cycles_sys",
    # "gpu_active_cycle_max": "elapsed_cycles_sys",
    
    "sm_active_cycles_sum": "active_cycles_pm", # Number of cycles a multiprocessor has at least one active warp.
    
    "sm_elapsed_cycles_sum": "elapsed_cycles_pm", # elapsed clocks on SM
    # "sm_elapsed_cycles_sum": "elapsed_cycles_sm", # elapsed clocks
    
    # "l1_hit_rate": "global_hit_rate",
    "l1_hit_rate": "tex_cache_hit_rate",
    "l2_hit_rate": "l2_tex_hit_rate",
    "gmem_read_requests": "global_load_requests",
    "gmem_write_requests": "global_store_requests",
    "gmem_read_trans": "gld_transactions",
    "gmem_write_trans": "gst_transactions",
    "l2_read_trans": "l2_read_transactions",
    "l2_write_trans": "l2_write_transactions",
    "dram_total_trans": "dram_total_transactions",
}

with open(args.hw_res, 'r') as f:
    hw_res = json.load(f)

with open(args.sim_res, 'r') as f:
    sim_res = json.load(f)

# proc hw_res
for app, kernels_res in hw_res.items():
    for kernel_res in kernels_res:
        kernel_res["dram_total_transactions"] = kernel_res["dram_read_transactions"] + kernel_res["dram_write_transactions"]

def get_kernel_stat(json_data, kernel_name_key, stat_key, func=None):
    X = []
    Y = []
    for app, kernels_res in json_data.items():
        for kernel_res in kernels_res:
            X.append(f"{app[:6]}_{kernel_res[kernel_name_key]}")
            if func:
                Y.append(func(kernel_res[stat_key]))
            else:
                Y.append(kernel_res[stat_key])
    return X, Y

def get_app_stat(json_data, stat_key, func=None):
    # sum kernel stat
    X = []
    Y = []
    for app, kernels_res in json_data.items():
        X.append(app)
        accum = 0
        for kernel_res in kernels_res:
            if func:
                accum += func(kernel_res[stat_key])
            else:
                accum += kernel_res[stat_key]
        Y.append(accum)
    return X, Y

def draw_error(stat, save_img_path, sim_res_func=None, error_text=True, overwrite=True):
    print(f"draw {stat}")
    if os.path.exists(save_img_path) and not overwrite:
        return
    def bar_overlay(ax, bars, x):
        i = 0
        for bar in bars:
            height = bar.get_height()  # 获取条形的高度（即对应的数值）
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height, f'{x[i]:.2f}', 
                    ha='center', va='bottom')
            i += 1
        
    x1, y1 = get_app_stat(hw_res, key_map[stat])
    _, y2 = get_app_stat(sim_res, stat, sim_res_func)

    error_y = np.abs(np.array(y1) - np.array(y2))/np.array(y1)
    avg_error = np.mean(error_y)
    
    fig, ax = plt.subplots()
    bars = ax.bar(x1, error_y)
    if error_text:
        bar_overlay(ax, bars, error_y)
    
    plt.xticks(rotation=-90)
    fig.subplots_adjust(bottom=0.4)
    ax.set_ylabel("Error")
    ax.set_xlabel("app")
    ax.set_title(f"{stat} Error, avg={avg_error:.2f}")
    # plt.show()
    fig.savefig(save_img_path)

draw_error("warp_inst_executed", "tmp/warp_inst_executed.png")
draw_error("gpu_active_cycle_max", "tmp/gpu_active_cycle_max.png")
draw_error("sm_active_cycles_sum", "tmp/sm_active_cycles_sum.png")
draw_error("achieved_occupancy", "tmp/app_occupancy_error.png", sim_res_func = lambda x: x/100)
draw_error("l1_hit_rate", "tmp/l1_hit_rate.png")
draw_error("l2_hit_rate", "tmp/l2_hit_rate.png")
draw_error("gmem_read_requests", "tmp/gmem_read_requests.png")
draw_error("gmem_read_trans", "tmp/gmem_read_trans.png")
draw_error("l2_read_trans", "tmp/l2_read_trans.png")
draw_error("dram_total_trans", "tmp/dram_total_trans.png")
