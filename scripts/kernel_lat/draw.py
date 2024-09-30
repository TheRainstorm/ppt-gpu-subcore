import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def draw(data, img_path):
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
    # plt.show()
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
    for app_arg, app_res in res.items():
        kernel_name, params = app_arg.split("/")
        m = re.search(r'(\d+)_+(\d+)', params)
        if m:
            grid_size, block_size = map(int, m.groups())
        else:
            exit(1)
        
        cycle1 = int(app_res[0]['gpc__cycles_elapsed.avg'])
        cycle2 = int(app_res[0]['sys__cycles_active.sum'])
        cycle3 = int(app_res[0]['gpc__cycles_elapsed.max'])
        if block_size not in data:
            data[block_size] = [[], []]
        data[block_size][0].append(grid_size)
        data[block_size][1].append(int(cycle3))
    
    draw(data, args.output)
    