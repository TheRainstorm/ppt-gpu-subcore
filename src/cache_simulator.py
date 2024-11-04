import argparse
import json
import math
import os
import matplotlib.pyplot as plt
import math
import time
from functools import wraps

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

class Node:
    # 提高访问属性的速度，并节省内存
    __slots__ = 'prev', 'next', 'key', 'value'

    def __init__(self, key=0):
        self.key = key

class LRUCache:
    def __init__(self, cache_parameter) -> None:
        self.associativity = int(cache_parameter['associativity'])
        self.capacity = int(cache_parameter['capacity'])
        self.cache_line_size = int(cache_parameter['cache_line_size'])
        self.cache_set_num = self.capacity // self.associativity // self.cache_line_size
        
        # helper
        self.blk_bits = int(math.log2(self.cache_line_size))
        self.idx_bits = int(math.log2(self.cache_set_num))
        
        self.idx_mask = (1 << self.idx_bits) - 1
        
        # cache
        self.cache_set_list = [{} for i in range(self.cache_set_num)]
        self.dummy_list = []
        for i in range(self.cache_set_num):
            e = Node()
            e.prev = e
            e.next = e
            self.dummy_list.append(e) 
        
        # statistic
        self.clear_statics()
    
    def access(self, mem_width, write, addr):
        idx = (addr >> self.blk_bits) & self.idx_mask
        tag = addr >> (self.blk_bits + self.idx_bits)
        
        self.read_cnt += 0 if write else 1
        self.write_cnt += 1 if write else 0
        
        node = self.get_node(idx, tag)
        hit = True if node else False
        if not hit:
            self.read_miss += 0 if write else 1
            self.write_miss += 1 if write else 0
            self.put_node(idx, tag)
        
        return hit

    def get_node(self, idx, tag):
        if tag not in self.cache_set_list[idx]:
            return None
        node = self.cache_set_list[idx][tag]
        # update LRU
        self.remove(node)
        self.push_front(idx, node)
        return node
    
    def put_node(self, idx, tag):
        node = self.get_node(idx, tag)
        if node:
            # update node value. We don' care value
            return
        node = Node(tag)
        self.cache_set_list[idx][tag] = node
        self.push_front(idx, node)
        if len(self.cache_set_list[idx]) > self.associativity:
            # remove LRU
            node = self.dummy_list[idx].prev
            del self.cache_set_list[idx][node.key]
            self.remove(node)
    
    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def push_front(self, idx, node):
        head = self.dummy_list[idx]
        node.next = head.next
        node.prev = head
        head.next.prev = node
        head.next = node
    
    def clear_statics(self):
        self.read_cnt, self.read_miss, self.write_cnt, self.write_miss = 0, 0, 0, 0
        
    def get_hit_info(self):
        total_access = self.read_cnt + self.write_cnt
        def divide_safe(a, b):
            return a / b if b else 0
        read_miss_raito = divide_safe(self.read_miss, self.read_cnt)
        write_miss_raito = divide_safe(self.write_miss, self.write_cnt)
        miss_ratio = divide_safe(self.read_miss + self.write_miss, total_access)
        
        return {
            "total_access": total_access,
            "read_miss_raito": read_miss_raito,
            "write_miss_raito": write_miss_raito,
            "miss_ratio": miss_ratio,
            "hit_ratio": 1 - miss_ratio,
            "read_cnt": self.read_cnt,
            "write_cnt": self.write_cnt,
            "total_access": total_access
        }

# @timeit
def get_cache_line_access_from_raw_trace(trace_file, l1_cache_line_size):
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
    # block_trace = open(trace_file,'r').read().strip().split("\n=====\n")
    block_trace = open(trace_file,'r').readlines()
    for trace_line in block_trace:
        trace_line_splited = trace_line.strip().split(' ')
        inst = trace_line_splited[0]
        addrs = trace_line_splited[1:]
        line_addrs, sector_addrs = get_line_adresses(addrs, l1_cache_line_size)
        
        for individual_addrs in line_addrs:
            cache_line_access.append([0, 0, 0, individual_addrs])
    print(f"[INFO]: {trace_file} req,warp_inst,ratio: {len(cache_line_access)},{len(block_trace)},{len(cache_line_access)/len(block_trace)}")
    return cache_line_access

def interleave_trace(block_trace_list):
    '''
    block_trace_list: list of block trace
    return interleaved list
    '''
    if len(block_trace_list) == 1:
        return block_trace_list[0]
    
    max_len = len(max(block_trace_list, key=len))
    interleaved_trace = []
    for i in range(max_len):
        for j in range(len(block_trace_list)):
            if i < len(block_trace_list[j]):
                interleaved_trace.append(block_trace_list[j][i])

    return interleaved_trace

# @timeit
def get_merged_line_access_from_raw_trace_list(trace_file_list, l1_cache_line_size):
    block_trace_list = []
    for trace_file in trace_file_list:
        block_trace_list.append(get_cache_line_access_from_raw_trace(trace_file, l1_cache_line_size))
    
    sm_cache_line_access = interleave_trace(block_trace_list)
    return sm_cache_line_access


def run(trace_files, ):
    l1_cache_parameter = {
        "capacity": 32 * 1024,
        "cache_line_size": 32,
        "associativity": 64,
    }
    cache_line_access = get_merged_line_access_from_raw_trace_list(trace_files, l1_cache_parameter['cache_line_size'])
    with open('cache_line_access.txt', 'w') as f:
        for item in cache_line_access:
            f.write("%s\n" % item)
    return cache_simulate(cache_line_access, l1_cache_parameter)

# @timeit
def cache_simulate(cache_line_access, cache_parameter):
    cache = LRUCache(cache_parameter)
    
    for inst_id, mem_id, warp_id, address in cache_line_access:
        mem_width = 4
        write = inst_id == '1'
        addr = address * cache_parameter['cache_line_size']

        hit = cache.access(mem_width, write, addr)
        # print(hit)
    
    # print(json.dumps(cache.get_hit_info(), indent=4))
    return cache.get_hit_info()['hit_ratio']
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument("-f", "--trace-files",
                        nargs="+",
                        required=True )
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    
    run(args.trace_files)
