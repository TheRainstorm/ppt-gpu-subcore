import json
import subprocess
import re
import matplotlib.pyplot as plt

def run_command_and_collect_data(command, block_sizes, loops):
    """
    运行命令并收集输出数据
    """
    data = []
    for block_size in block_sizes:
        for loop in loops:
            full_command = f"{command} {block_size} {loop}"
            print(f"Running: {full_command}")
            try:
                # 运行命令并获取输出
                output = subprocess.check_output(full_command, shell=True, text=True)
                # 提取 Cycles (per thread) 的值
                match = re.search(r"Cycles \(per thread\) = (\d+)", output)
                if match:
                    cycles = int(match.group(1))
                    data.append((block_size, loop, cycles))
                    print(f"Block Size: {block_size}, Loop: {loop}, Cycles: {cycles}")
                else:
                    print(f"Could not find 'Cycles (per thread)' in output: {output}")
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
    return data

def plot_data(data):
    """
    绘制循环 (loop) 与 Cycles (per thread) 的曲线
    """
    # 将数据按 block_size 分组
    grouped_data = {}
    for block_size, loop, cycles in data:
        if block_size not in grouped_data:
            grouped_data[block_size] = []
        grouped_data[block_size].append((loop, cycles))

    # 绘制每个 block_size 的曲线
    for block_size, values in grouped_data.items():
        values.sort()  # 按 loop 排序
        loops, cycles = zip(*values)
        plt.plot(loops, cycles, label=f"Block Size: {block_size}")

    plt.xlabel("Loop")
    plt.ylabel("Cycles (per thread)")
    plt.title("Loop vs Cycles (per thread)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 配置要测试的 block_size 和 lo
    block_sizes = [32, 64, 128, 160, 192, 224, 256]
    loops = list(range(1, 15))
    # loops = list(range(1, 2))
    
    data = run_command_and_collect_data('/staff/fyyuan/repo/GPGPUs-Workloads/Benchmarks/micro/ITVAL/ITVAL_F', block_sizes, loops)
    out = 'CPE_test.json'
    with open(out, 'w') as f:
        json.dump(data, f)
    
    # plot_data(data)