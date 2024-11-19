

import math
import os


def get_line_adresses(addresses, l1_cache_line_size, sector_size=32):
    '''
    coalescing the addresses of the warp
    '''
    line_idx = int(math.log(l1_cache_line_size,2))
    sector_idx = int(math.log(sector_size,2))
    # line_mask = ~(2**line_idx - 1)
    # sector_mask = ~(2**sector_idx - 1)
    
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

def process_trace(block_trace, l1_cache_line_size, sector_size=32):
    S = {}
    S["gmem_ld_reqs"] = 0
    S["gmem_st_reqs"] = 0
    S["gmem_ld_trans"] = 0
    S["gmem_st_trans"] = 0
    S["lmem_ld_reqs"] = 0
    S["lmem_st_reqs"] = 0
    S["lmem_ld_trans"] = 0
    S["lmem_st_trans"] = 0
    S["lmem_used"] = False
    S["atom_reqs"] = 0
    S["red_reqs"] = 0
    S["atom_red_trans"] = 0
    
    S["gmem_ld_sectors"] = 0
    S["gmem_st_sectors"] = 0
    S["umem_ld_sectors"] = 0
    S["umem_st_sectors"] = 0

    warp_id = 0 ## warp counter for l2 inclusion
    is_store = 0 ## LD=0 - ST=1
    is_local = 0  ## Global=0 - Local=1

    cache_line_access = []
    sector_access = []
    for items in block_trace:
        addrs = items.strip().split(" ")
        access_type = addrs[0]
        line_addrs, sector_addrs = get_line_adresses(addrs[1:], l1_cache_line_size, sector_size)

        ## global reduction operations
        if "RED" in access_type:
            S["red_reqs"] += 1
            S["atom_red_trans"] += len(sector_addrs)
            continue

        ## global atomic operations
        if "ATOM" in access_type:
            S["atom_reqs"] += 1
            S["atom_red_trans"] += len(sector_addrs)
            continue

        warp_id += 1
        ## global memory access
        if "LDG" in access_type or "STG" in access_type: 
            is_local = 0
            if "LDG" in access_type:
                is_store = 0
                S["gmem_ld_reqs"] += 1
                S["gmem_ld_trans"] += len(line_addrs)
                S["gmem_ld_sectors"] += len(sector_addrs)
            elif "STG" in access_type:
                is_store = 1
                S["gmem_st_reqs"] += 1
                S["gmem_st_trans"] += len(line_addrs)
                S["gmem_st_sectors"] += len(sector_addrs)
        
        ## local memory access
        elif "LDL" in access_type or "STL" in access_type:
            is_local = 1
            S["lmem_used"] = True
            if "LDL" in access_type:
                is_store = 0
                S["lmem_ld_reqs"] += 1
                S["lmem_ld_trans"] += len(line_addrs)
            elif "STL" in access_type:
                is_store = 1
                S["lmem_st_reqs"] += 1
                S["lmem_st_trans"] += len(line_addrs)
        
        if is_store == 1:
            S["umem_st_sectors"] += len(sector_addrs)
        else:
            S["umem_ld_sectors"] += len(sector_addrs)
            
        for individual_addrs in line_addrs:
            cache_line_access.append([is_store, is_local, warp_id, individual_addrs])
        for individual_addrs in sector_addrs:
            sector_access.append([is_store, is_local, warp_id, individual_addrs])
    S["umem_ld_reqs"] =  S["gmem_ld_reqs"]  + S["lmem_ld_reqs"] # + tex + surface
    S["umem_st_reqs"] =  S["gmem_st_reqs"]  + S["lmem_st_reqs"] 
    S["umem_ld_trans"] = S["gmem_ld_trans"] + S["lmem_ld_trans"]
    S["umem_st_trans"] = S["gmem_st_trans"] + S["lmem_st_trans"]
    # print(f"[INFO]: {trace_file} req,warp_inst,ratio: {len(cache_line_access)},{len(block_trace)},{len(cache_line_access)/len(block_trace)}")
    return S, sector_access



# trace_path = os.path.join(trace_dir, 'memory_traces', f"kernel_{kernel_id}_sm_{smi}.mem")
trace_path = '/staff/fyyuan/hw_trace01/ppt-gpu-titanv/11.0/polybench-2mm/NO_ARGS/memory_traces/kernel_1_sm_0.mem'
with open(trace_path,'r') as f:
    smi_blocks_interleave = f.readlines()

sm_stats, smi_trace = process_trace(smi_blocks_interleave, 128, 32) # warp level to cache line level

cache_line_access = smi_trace
cnt = 0
debug_file = open('trace_idx.csv', 'w')
for is_store, is_local, warp_id, address in cache_line_access:
    cnt += 1
    mem_width = 4
    # addr = address * cache_parameter['cache_line_size']
    addr = address * 32  # sector size 32B
    idx1 = (addr//32) % 512
    idx2 = (addr//128) % 128
    debug_file.write(f"{cnt},{is_store},{is_local},{warp_id},{addr},{address},{hex(addr)},{hex(address)},{idx1},{idx2}\n")

debug_file.close()
