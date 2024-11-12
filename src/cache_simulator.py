import argparse
import json
import math
import os
import math
import time
from functools import wraps
import bisect

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

def generate_primes(limit):
    """生成不超过 limit 的素数列表"""
    is_prime = [True] * (limit + 1)
    is_prime[0], is_prime[1] = False, False
    primes = []
    
    for num in range(2, limit + 1):
        if is_prime[num]:
            primes.append(num)
            for multiple in range(num * 2, limit + 1, num):
                is_prime[multiple] = False
    return primes

# 生成前100万的素数
primes_list = generate_primes(1200000)

def find_nearest_prime(primes, number):
    """找到最接近给定数字的素数"""
    pos = bisect.bisect_left(primes, number)
    
    # 如果数字在素数列表的范围之外，返回边界值
    if pos == 0:
        return primes[0]
    if pos == len(primes):
        return primes[-1]
    
    # 找到最接近的素数
    before = primes[pos - 1]
    return before
    # after = primes[pos]
    # if abs(number - before) <= abs(after - number):
    #     return before
    # else:
    #     return after

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
        cache_set = self.capacity // self.associativity // self.cache_line_size
        self.cache_set_num = find_nearest_prime(primes_list, cache_set)  # prime to avoid conflict on single set
        
        # helper
        # self.blk_bits = int(math.log2(self.cache_line_size))
        # self.idx_bits = ceil(math.log2(self.cache_set_num))  # 向上取整
        # self.idx_mask = (1 << self.idx_bits) - 1
        # 不用像硬件一样存储 tag（避免存储冗余的部分），直接存储整个 addr 即可
        # idx 部分直接取模 cache set 数即可
        
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
        idx = (addr // self.cache_line_size) % self.cache_set_num
        tag = addr
        
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

from sdcm import get_cache_line_access_from_raw_trace
from memory_model import interleave_trace
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
    
    L2_req = []
    
    read_g, write_g = 0, 0
    read_hit_g, write_hit_g = 0, 0
    
    cnt = 0
    # debug_file = open('debug2.csv', 'w')
    for is_store, is_local, warp_id, address in cache_line_access:
        cnt += 1
        mem_width = 4
        # addr = address * cache_parameter['cache_line_size']
        addr = address * 32  # sector size 32B

        if not is_local:
            read_g += 0 if is_store else 1
            write_g += 1 if is_store else 0
        
        hit = cache.access(mem_width, is_store, addr)
        # debug_file.write(f"{cnt},{is_store},{is_local},{warp_id},{address},{hit}\n")
        
        if hit:
            if not is_local:
                read_hit_g += 0 if is_store else 1
                write_hit_g += 1 if is_store else 0
            
            if is_store:  # write through
                L2_req.append([is_store, is_local, warp_id, address])
        else:
            L2_req.append([is_store, is_local, warp_id, address])
        
    # print(json.dumps(cache.get_hit_info(), indent=4))
    hit_rate_dict = {}
    cache_info = cache.get_hit_info()
    hit_rate_dict['tot'] = cache_info['hit_ratio']
    hit_rate_dict['ld'] = 1 - cache_info['read_miss_raito']
    hit_rate_dict['st'] = 1 - cache_info['write_miss_raito']
    hit_rate_dict['ldg'] = (read_hit_g/read_g) if read_g else 0
    hit_rate_dict['stg'] = (write_hit_g/write_g) if write_g else 0
    
    return hit_rate_dict, L2_req
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument("-f", "--trace-files",
                        nargs="+",
                        required=True )
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    
    run(args.trace_files)
