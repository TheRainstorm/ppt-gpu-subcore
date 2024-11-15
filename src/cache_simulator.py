import argparse
from enum import IntEnum
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
    __slots__ = 'prev', 'next', 'key', 'value', 'sectors_valid', 'sectors_dirty'

    def __init__(self, key=0):
        self.key = key
        self.sectors_valid = [0] * 32  # assume max 32 sector in one cache line
        self.sectors_dirty = [0] * 32
        self.next = self.prev = self

class W(IntEnum):
    write_back = 0
    write_through = 1

class LRUCache:
    def __init__(self, cache_parameter, keep_traffic=False) -> None:
        self.associativity = int(cache_parameter['associativity'])
        self.capacity = int(cache_parameter['capacity'])
        self.cache_line_size = int(cache_parameter['cache_line_size'])
        self.sector_size = cache_parameter.get('sector_size', self.cache_line_size)  # default non sector
        self.write_allocate = cache_parameter.get('write_allocate', True)
        self.write_strategy = cache_parameter.get('write_strategy', W.write_back)
        
        self.keep_traffic = keep_traffic
        # cache
        cache_set = self.capacity // self.associativity // self.cache_line_size
        self.cache_set_num = find_nearest_prime(primes_list, cache_set)  # prime to avoid conflict on single set
        self.sectors = self.cache_line_size // self.sector_size
        
        self.cache_set_list = [{} for i in range(self.cache_set_num)]
        self.dummy_list = []
        for i in range(self.cache_set_num):
            e = Node()
            e.prev = e
            e.next = e
            self.dummy_list.append(e)
        
        # statistic
        self.clear_statics()
        self.data = {}
        
    def inc(self, key):
        if self.scope not in self.data:
            self.data[self.scope] = {}
        if key not in self.data[self.scope]:
            self.data[self.scope][key] = 0
        self.data[self.scope][key] += 1
        
    def parse_addr(self, addr):
        cache_line_idx = (addr // self.cache_line_size) % self.cache_set_num
        sector_idx = (addr // self.sector_size) % (self.cache_line_size // self.sector_size)
        tag = addr // self.cache_line_size  # also include cache_line_idx
        return cache_line_idx, sector_idx, tag
    
    def read(self, mem_width, addr):
        self.read_cnt += 1
        self.inc('read_cnt')
        self.traffics = []
        
        cache_line_idx, sector_idx, tag = self.parse_addr(addr)
        code, node = self.get_node(cache_line_idx, sector_idx, tag)
        hit = code==0
        if not hit:
            self.read_miss += 1
            self.inc('read_miss')
            if code==1:
                self.read_tag_miss += 1
                self.inc('read_tag_miss')
            self.read_req += 1 # read from memory
            self.inc('read_req')
            if self.keep_traffic:
                self.traffics.append([0, self.sector_size, addr])
            self.put_node(cache_line_idx, sector_idx, tag, node) # write to cache, omit write value
        return hit, self.traffics
    
    def write(self, mem_width, addr):
        self.write_cnt += 1
        self.inc('write_cnt')
        self.traffics = []
        
        cache_line_idx, sector_idx, tag = self.parse_addr(addr)
        code, node = self.get_node(cache_line_idx, sector_idx, tag)
        hit = code==0
        
        if code!=0:
            self.write_miss += 1
            self.inc('write_miss')
            if code==1:
                self.write_tag_miss += 1
                self.inc('write_tag_miss')
                
        if self.write_strategy == W.write_through:
            self.write_through += 1
            self.inc('write_through')
            if self.keep_traffic:
                self.traffics.append([1, self.sector_size, addr])
        elif self.write_strategy == W.write_back:
            if hit:
                node.sectors_dirty[sector_idx] = 1
            else:
                if self.write_allocate:
                    self.put_node(cache_line_idx, sector_idx, tag, node, dirty=1)
                else:
                    # don't allocate cache, write to memory, 
                    self.write_nonallocate += 1
                    self.inc('write_nonallocate')
                    if self.keep_traffic:
                        self.traffics.append([1, self.sector_size, addr])
        return hit, self.traffics
        
    def access(self, mem_width, write, addr):
        if write:
            return self.write(mem_width, addr)
        else:
            return self.read(mem_width, addr)
    def flush_dirty(self):
        if self.write_strategy == W.write_back:
            for cache_set in self.cache_set_list:
                for node in cache_set.values():
                    for i in range(self.sectors):
                        if node.sectors_valid[i] == 1 and node.sectors_dirty[i] == 1:
                            node.sectors_dirty[i] = 0
                            self.write_flush += 1
                            if self.keep_traffic:
                                self.traffics.append([1, self.sector_size, node.key * self.cache_line_size + i * self.sector_size])
                        
    def get_node(self, cache_line_idx, sector_idx, tag):
        '''check tag and data, return node if hit
        '''
        if tag not in self.cache_set_list[cache_line_idx]:
            return 1, None  # tag miss
        node = self.cache_set_list[cache_line_idx][tag]
        if node.sectors_valid[sector_idx] == 0:
            return 2, node # data miss
        # update LRU
        self.remove(node)
        self.push_front(cache_line_idx, node)
        return 0, node
    
    def put_node(self, cache_line_idx, sector_idx, tag, node, dirty=0):
        '''allocate new node to cache (only tag, omit data, since we only care about hit/miss)
        '''
        if not node: # it's possible tag hit, sector miss
            node = Node(tag)
        node.sectors_valid[sector_idx] = 1
        node.sectors_dirty[sector_idx] = dirty  # read miss: clean, write miss: dirty
        self.cache_set_list[cache_line_idx][tag] = node
        self.remove(node)
        self.push_front(cache_line_idx, node)
        if len(self.cache_set_list[cache_line_idx]) > self.associativity:
            # remove LRU
            node = self.dummy_list[cache_line_idx].prev
            # evict dirty data
            for i in range(self.sectors):
                if node.sectors_valid[i] == 1 and node.sectors_dirty[i] == 1:
                    self.write_evict += 1
                    if self.keep_traffic:
                        self.traffics.append([1, self.sector_size, tag * self.cache_line_size + i * self.sector_size])
            del self.cache_set_list[cache_line_idx][node.key]
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
        self.read_cnt, self.read_miss, self.read_tag_miss, self.write_cnt, self.write_miss, self.write_tag_miss, = 0, 0, 0, 0, 0, 0
        self.read_req = 0
        self.write_through, self.write_evict, self.write_nonallocate, self.write_flush = 0, 0, 0, 0
        
    def get_cache_info(self):
        return caculate(self.read_cnt, self.read_miss, self.read_tag_miss, self.write_cnt, self.write_miss, self.write_tag_miss,
                        self.read_req, self.write_through, self.write_evict, self.write_nonallocate, self.write_flush)

def caculate(read_cnt, read_miss, read_tag_miss, write_cnt, write_miss, write_tag_miss, read_req, write_through, write_evict, write_nonallocate, write_flush):
    total_access = read_cnt + write_cnt
    def divide_safe(a, b):
        return a / b if b else 0
    read_miss_ratio = divide_safe(read_miss, read_cnt)
    write_miss_ratio = divide_safe(write_miss, write_cnt)
    miss_ratio = divide_safe(read_miss + write_miss, total_access)
    
    read_tag_miss_ratio = divide_safe(read_tag_miss, read_cnt)
    write_tag_miss_ratio = divide_safe(write_tag_miss, write_cnt)
    tag_miss_ratio = divide_safe(read_tag_miss + write_tag_miss, total_access)
    
    return {
        "read_miss_ratio": read_miss_ratio,
        "write_miss_ratio": write_miss_ratio,
        "miss_ratio": miss_ratio,
        "hit_ratio": 1 - miss_ratio,
        
        "read_tag_miss_ratio": read_tag_miss_ratio,
        "write_tag_miss_ratio": write_tag_miss_ratio,
        "tag_miss_ratio": tag_miss_ratio,
        
        "read_cnt": read_cnt,
        "write_cnt": write_cnt,
        "total_access": total_access,
        
        "read_req": read_req,
        "write_req": write_through + write_evict + write_nonallocate + write_flush,
        "write_through": write_through,
        "write_evict": write_evict,
        "write_nonallocate": write_nonallocate,
    }
            
from src.sdcm import get_cache_line_access_from_raw_trace
from src.memory_model import interleave_trace
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
def cache_simulate(cache_line_access, cache_parameter, dump_trace='', keep_traffic=False):
    cache = LRUCache(cache_parameter, keep_traffic=keep_traffic)
    
    req_nextlv = []
    
    cnt = 0
    if dump_trace:
        debug_file = open(dump_trace, 'w')
    
    for is_store, is_local, warp_id, address in cache_line_access:
        scope = 'local' if is_local else 'global'
        cache.scope = scope
        
        cnt += 1
        mem_width = 4
        # addr = address * cache_parameter['sector_size']
        addr = address

        hit, traffics = cache.access(mem_width, is_store, addr)
        if dump_trace:
            debug_file.write(f"{cnt},{is_store},{is_local},{warp_id},{address},{hit}\n")
        
        for traffic in traffics:
            req_nextlv.append([traffic[0], is_local, warp_id, traffic[2]])
    
    if cache.write_evict > 0:
        print(f"Info: write evict {cache.write_evict} before flush")
    cache.flush_dirty()
    
    if dump_trace:
        debug_file.close()
    
    cache_info_scopes = {}
    for scope, scope_data in cache.data.items():
        cache_info = caculate(scope_data.get('read_cnt', 0), scope_data.get('read_miss', 0), scope_data.get('read_tag_miss', 0),
                            scope_data.get('write_cnt', 0), scope_data.get('write_miss', 0), scope_data.get('write_tag_miss', 0),
                            scope_data.get('read_req', 0), scope_data.get('write_through', 0), scope_data.get('write_evict', 0),
                            scope_data.get('write_nonallocate', 0), scope_data.get('write_flush', 0))
        cache_info_scopes[scope] = cache_info
    
    cache_info = cache.get_cache_info()
    hit_rate_dict = {}
    hit_rate_dict['tot'] = cache_info['hit_ratio']
    hit_rate_dict['ld'] = 1 - cache_info['read_miss_ratio']
    hit_rate_dict['st'] = 1 - cache_info['write_miss_ratio']
    hit_rate_dict['ldg'] = 1 - cache_info_scopes['global']['read_miss_ratio']
    hit_rate_dict['stg'] = 1 - cache_info_scopes['global']['write_miss_ratio']
    
    # extra info
    hit_rate_dict['tot_tag'] = 1 - cache_info['tag_miss_ratio']
    hit_rate_dict['ld_tag'] = 1 - cache_info['read_tag_miss_ratio']
    hit_rate_dict['st_tag'] = 1 - cache_info['write_tag_miss_ratio']
    hit_rate_dict['sectors_ld'] = cache_info['read_cnt']
    hit_rate_dict['sectors_st'] = cache_info['write_cnt']
    hit_rate_dict['sectors_ld_nextlv'] = cache_info['read_req']
    hit_rate_dict['sectors_st_nextlv'] = cache_info['write_req']
    
    return hit_rate_dict, req_nextlv
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument("-f", "--trace-files",
                        nargs="+",
                        required=True )
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    
    run(args.trace_files)
