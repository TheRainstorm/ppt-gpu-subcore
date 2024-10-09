import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def draw(data, img_path, show=False):
    fig, ax = plt.subplots()
    z_list = []
    for block_size, (x, y) in data.items():
        z = np.polyfit(x, y, 1)
        z_list.append(z)
        
        ax.scatter(x, y, label=f"{block_size}: {z[0]:.3}x + {int(z[1])}")
        ax.legend()
    
    print(z_list)
    avg_z = np.mean(z_list, axis=0)
    print(avg_z)

    ax.set_ylabel("Cycle")
    ax.set_xlabel("Grid size")
    ax.set_title(f"Empty Kernel {avg_z[0]:.3}x + {int(avg_z[1])}")
    if show:
        plt.show()
    fig.savefig(img_path)
    plt.close(fig)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='draw_1 support: 1) error bar or side by side bar cmp with hw 2) apps or kernels granularity 3) all apps or part of it. we can combine them to draw what we want'
    )
    parser.add_argument("-i", "--res", required=True, help="input json file")
    parser.add_argument("-o", "--output", default="kernel_cycle.png", help="draw  image")
    args = parser.parse_args()
    
    with open(args.res) as f:
        res = json.load(f)
    
    data = {}
    data_le = {}
    data_gt = {}
    for app_arg, app_res in res.items():
        kernel_name, params = app_arg.split("/")
        if 'kernel_lat' not in kernel_name:
            continue
        m = re.search(r'(\d+)_+(\d+)', params)
        if m:
            grid_size, block_size = map(int, m.groups())
        else:
            exit(1)
            
        if grid_size < 118:
            cycle = int(app_res[0]['gpc__cycles_elapsed.max'])
            if block_size not in data_le:
                data_le[block_size] = [[], []]
            data_le[block_size][0].append(grid_size)
            data_le[block_size][1].append(int(cycle))
        else:
            cycle = int(app_res[0]['gpc__cycles_elapsed.max'])
            if block_size not in data_gt:
                data_gt[block_size] = [[], []]
            data_gt[block_size][0].append(grid_size)
            data_gt[block_size][1].append(int(cycle))
    
    def merge(data1, data2):
        for k, v in data2.items():
            if k not in data1:
                data1[k] = [[], []]
            data1[k][0].extend(v[0])
            data1[k][1].extend(v[1])
        return data1
    draw(data_le, args.output.replace(".png", "_le.png"))
    draw(data_gt, args.output)
    draw(merge(data_le, data_gt), args.output.replace(".png", "_all.png"), show=True)
    