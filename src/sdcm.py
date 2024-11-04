import argparse
import json
import math
import os
import matplotlib.pyplot as plt

import time
from functools import wraps

from sortedcontainers import SortedList


def timeit(func):
    """
    装饰器，用于统计函数执行时间。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds")
        return result
    return wrapper

def process_trace(block_trace, l1_cache_line_size, inst_count=None):
    def get_line_adresses(addresses, l1_cache_line_size):
        '''
        coalescing the addresses of the warp
        '''
        line_idx = int(math.log(l1_cache_line_size,2))
        sector_size = 32
        sector_idx = int(math.log(sector_size,2))
        line_mask = ~(2**line_idx - 1)
        sector_mask = ~(2**sector_idx - 1)
        
        cache_line_set = set()
        sector_set = set()  # count sector number
        
        for addr in addresses:
            # 排除 0 ？
            if addr:
                addr = int(addr, base=16)
                cache_line = addr >> line_idx
                cache_line_set.add(cache_line)
                sector = addr >> sector_idx
                sector_set.add(sector)
        
        return list(cache_line_set), list(sector_set)

    cache_line_access = []
    for trace_line in block_trace:
        trace_line_splited = trace_line.strip().split(' ')
        inst = trace_line_splited[0]
        
        if not ("LDG" in inst or "STG" in inst or "LDL" in inst or "STL" in inst):
            if inst_count is not None:
                inst_count[inst] = inst_count.get(inst, 0) + 1
            continue
        
        addrs = trace_line_splited[1:]
        line_addrs, sector_addrs = get_line_adresses(addrs, l1_cache_line_size)
        
        for individual_addrs in line_addrs:
            cache_line_access.append([0, 0, 0, individual_addrs])
    
    # print(f"[INFO]: {trace_file} req,warp_inst,ratio: {len(cache_line_access)},{len(block_trace)},{len(cache_line_access)/len(block_trace)}")
    return cache_line_access
# @timeit
def get_cache_line_access_from_raw_trace(trace_file, l1_cache_line_size):
    block_trace = open(trace_file,'r').readlines()
    return process_trace(block_trace, l1_cache_line_size)

# @timeit
def get_cache_line_access_from_file(file_path):
    cache_line_access = []
    with open(file_path, 'r') as f:
        # assert(fscanf(fp, "%s %s %s %s", inst_id, mem_id, warp_id, address) != EOF);
        for line in f.readlines():
            cache_line_access.append(line.split())
    
    return cache_line_access

def get_stack_distance(cache_line_access):
    stack = SortedList()
    last_position = {}
    SD = []

    for i, (inst_id, mem_id, warp_id, address) in enumerate(cache_line_access):
        if address in last_position:
            # 获取上次访问的位置
            previous_index = last_position[address]
            # 计算 stack distance 为当前排序列表的长度减去之前的位置
            sd = len(stack) - stack.bisect_left(previous_index) - 1
            SD.append(sd)
            stack.remove(previous_index)  # 移除上次的访问位置
        else:
            SD.append(-1)
        
        # 记录当前访问位置
        last_position[address] = i
        stack.add(i)

    return SD

# @timeit
def get_stack_distance_1(cache_line_access):
    '''caculate sd for each reference
    rs: reference stream, cache line address list
    return: stack distance of each reference
    - sd
    '''
    stack = []
    SD = []
    for inst_id, mem_id, warp_id, address in cache_line_access:
        if address in stack:
            sd = len(stack) - stack.index(address) - 1
            SD.append(sd)
            stack.remove(address)
        else:
            SD.append(-1)
        stack.append(address)
    return SD

# @timeit
def get_csdd(SD):
    '''get cumulative stack distance distribution (csdd)
    '''
    T = len(SD)
    sd_counter = {}
    for sd in SD:
        if sd not in sd_counter:
            sd_counter[sd] = 0
        sd_counter[sd] += 1
    max_sd = max(sd_counter.keys())
    sd_counter[max_sd+1] = sd_counter[-1]
    del sd_counter[-1]
    
    # sort
    sd_histogram = sorted(sd_counter.items())
    
    # get sdd
    sdd = [(sd, count, count/T) for sd, count in sd_histogram]
    
    csdd = []
    accum = 0
    for sd, count in sd_histogram:
        accum += count
        csdd.append((sd, accum))

    # avg
    csdd_avg = [(sd, accum/T) for sd, accum in csdd]
    return sdd, csdd_avg

def calculate_p_hit(A, B, D):
    p_hit = 0.0
    for a in range(A):
        # 计算组合数 C(D, a)
        comb = math.comb(D, a)
        # 计算公式中的各项
        term1 = (A / B) ** a
        term2 = ((B - A) / B) ** (D - a)
        # 累加到 P_HIT
        p_hit += comb * term1 * term2
    return p_hit

from scipy import special as sp

def qfunc(arg):
    return 0.5-0.5*sp.erf(arg/1.41421)

def calculate_p_hit_approx(A, B, D):
    mean = D * (A/B)
    variance = mean * ((B-A)/B)
    p_hit = 1 - qfunc( abs(A-1-mean) / math.sqrt(variance) )
    return p_hit

# @timeit
def sdcm(sdd, cache_line_size, cache_size, associativity, use_approx=False):
    '''calculate the stack distance cache miss rate (SDCM)
    sdd: stack distance distribution
    cache_line_size: cache line size
    cache_size: cache size
    associativity: cache associativity
    '''
    # cache line number
    B = cache_size // cache_line_size
    A = associativity
    
    hit_rate = 0
    for sd, _, p_sd in sdd[:-1]:  # the last one is -1
        if sd==0:
            p_hit = 1
        elif sd==-1:
            p_hit = 0
        else:
            try:
                if use_approx:
                    p_hit = calculate_p_hit_approx(A, B, sd)
                else:
                    p_hit = calculate_p_hit(A, B, sd)
            except:
                print(f"[ERROR]: A: {A}, B: {B}, sd: {sd}")
                exit(-1)
        hit_rate += p_hit * p_sd
    return hit_rate

def sdcm_model(cache_line_access, cache_parameter, use_approx=True):
    SD = get_stack_distance(cache_line_access)
    sdd, csdd = get_csdd(SD)
    hit_rate = sdcm(sdd, cache_parameter['cache_line_size'], cache_parameter['capacity'], cache_parameter['associativity'], use_approx=use_approx)
    # print(f"hit rate: {hit_rate}")
    return hit_rate
    
def draw_csdd(csdd, img_path):
    import matplotlib.pyplot as plt
    
    x = [sd for sd, _ in csdd]
    y = [accum for _, accum in csdd]
    
    fig, ax = plt.subplots()

    ax.plot(x, y)
    
    # add some text for labels, title and axes ticks
    ax.set_xlabel("Stack Distance")
    ax.set_ylabel("CSDD")
    fig.savefig(img_path)
    plt.close(fig)

def draw_csdd_list(csdd_list, labels, img_path):
    fig, ax = plt.subplots()
    
    # 设置不同的标记样式
    markers = ['o', 's', '^', 'd']
    linestyles = ['-', '--', '-.', ':']
    
    for i, csdd in enumerate(csdd_list):
        x = [sd for sd, _ in csdd]
        y = [accum for _, accum in csdd]
        ax.plot(x, y, label=labels[i], marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)])
    
    # 设置对数坐标轴
    ax.set_xscale('log')
    ax.set_yscale('linear')

    ax.legend()
    ax.set_xlabel("Stack Distance")
    ax.set_ylabel("CSDD")
    fig.savefig(img_path)
    plt.close(fig)

def draw_sdd(sdd, img_path):
    x = [sd for sd, _, _ in sdd]
    y = [count for _, _, count in sdd]
    
    fig, ax = plt.subplots()

    ax.plot(x, y)
    
    # add some text for labels, title and axes ticks
    ax.set_xlabel("Stack Distance")
    ax.set_ylabel("percentage")
    fig.savefig(img_path)
    plt.close(fig)

def draw_SD(sdd, img_path):
    x = [sd for sd, _, _ in sdd]
    y = [count for _, _, count in sdd]
    
    fig, ax = plt.subplots()

    ax.hist(x, weights=y, alpha=0.7, color='blue', edgecolor='black')
    
    # add some text for labels, title and axes ticks
    ax.set_xlabel("Stack Distance")
    ax.set_ylabel("percentage")
    fig.savefig(img_path)
    plt.close(fig)

def read_sdd_from_pardax_output(file):
    sdd = []
    with open(file, 'r') as f:
        for line in f.readlines():
            line_split = line.split(',')
            sd = int(line_split[0])
            prop = float(line_split[1])
            cnt = int(line_split[2])
            if sd== -1:
                sd = sdd[-1][0] + 1
            sdd.append((sd, cnt, prop))

    return sdd
if __name__ == "__main__2":
    # sdd = read_sdd('K1_GMEM_SM0_lds.rp')
    sdd = read_sdd_from_pardax_output('K1_UMEM_SM0.rp')
    hit_rate = sdcm(sdd, 32, 32*1024, 64, use_approx=True)
    print(hit_rate)
    hit_rate = sdcm(sdd, 32, 32*1024, 64, use_approx=False)
    print(hit_rate)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument("-f", "--trace-files",
                        nargs="+",
                        required=True )
    parser.add_argument("-l", "--labels",
                        nargs="+",
                        required=True )
    parser.add_argument("-o", "--output-dir", default="draw")
    parser.add_argument("-a", "--approx", action="store_true")
    parser.add_argument("-r", "--raw-trace", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    SD_list = []
    sdd_list = []
    csdd_list = []
    labels = args.labels
    draw_cmp = True if len(args.trace_files) > 1 else False
    cache_size = 32*1024
    cache_line_size = 32
    cache_associativity = 64
    
    if len(labels) != len(args.trace_files):
        print("The number of labels must be equal to the number of trace files")
        exit(-1)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    debug = {}
    if args.debug and os.path.exists('debug.json'):
        with open('debug.json') as f:
            debug = json.load(f)
            SD_list = debug['SD_list']
            sdd_list = debug['sdd_list']
            csdd_list = debug['csdd_list']
            labels = debug['labels']
    else:
        for trace_file in args.trace_files:
            if args.raw_trace:
                cache_line_access = get_cache_line_access_from_raw_trace(trace_file, cache_line_size)
            else:
                cache_line_access = get_cache_line_access_from_file(trace_file)
            
            SD = get_stack_distance(cache_line_access)
            sdd, csdd = get_csdd(SD)
            SD_list.append(SD)
            sdd_list.append(sdd)
            csdd_list.append(csdd)
        
        debug['SD_list'] = SD_list
        debug['sdd_list'] = sdd_list
        debug['csdd_list'] = csdd_list
        debug['labels'] = labels
    
    with open('debug.json', 'w') as f:
        json.dump(debug, f, indent=4)
    
    for i, sdd in enumerate(sdd_list):
        # check sdd
        accum = 0
        for sd, _, prop in sdd:
            accum += prop
        print(f"[INFO]: {labels[i]}: total request: {len(SD_list[i])}, distinct sd: {len(sdd)}")
        print(f"[INFO]: {labels[i]}: accum: {accum}")
        
        hit_rate = sdcm(sdd, cache_line_size, cache_size, cache_associativity, use_approx=args.approx)
        print(f"[INFO]: {labels[i]}: hit rate: {hit_rate}")
    
    for i, csdd in enumerate(csdd_list):
        sdd = sdd_list[i]
        draw_SD(sdd, os.path.join(args.output_dir, f"{labels[i]}_SD.png"))
        draw_sdd(sdd, os.path.join(args.output_dir, f"{labels[i]}_sdd.png"))
        draw_csdd(csdd,  os.path.join(args.output_dir, f"{labels[i]}_csdd.png"))
    
    if draw_cmp:
        draw_csdd_list(csdd_list, labels, os.path.join(args.output_dir, f"{'-'.join(labels)}_csdd_cmp.png"))