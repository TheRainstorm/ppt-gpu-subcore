import argparse
from enum import IntEnum
import json
import os
import random
import sys
import prettytable as pt

curr_dir = os.path.dirname(__file__)
par_dir = os.path.dirname(curr_dir)
sys.path.insert(0, os.path.abspath(par_dir))

from src.memory_model import interleave_trace, get_memory_perf
from src.kernels import get_max_active_block_per_sm
from ppt import get_gpu_config, get_kernels_launch_params

from src.sdcm import sdcm_model, process_trace
from src.cache_simulator import LRUCache, cache_simulate, W
import concurrent.futures
import multiprocessing

def divide_or_zero(a, b):
    if b == 0:
        return 0
    return a/b

class BlockMapping(IntEnum):
    mod_block_mapping = 1
    random_block_mapping = 2
    sm_block_mapping = 3

def ppt_gpu_model_warpper(kernel_id, trace_dir,
                         launch_params,
                         max_blocks_per_sm, 
                         gpu_config, # gpu config
                         granularity = 2,
                        ):
        
    avg_block_per_sm = (launch_params['grid_size'] + gpu_config['num_SMs'] - 1) // gpu_config['num_SMs']
    if granularity == 1:
        block_per_sm_simulate = 1
    elif granularity == 2:
        block_per_sm_simulate = max_blocks_per_sm
    elif granularity == 3:
        block_per_sm_simulate = avg_block_per_sm
        
    mem_traces_dir_path = os.path.join(trace_dir, 'memory_traces')

    gmem_reqs = 1  # 
    memory_stats = get_memory_perf(kernel_id, mem_traces_dir_path, launch_params['grid_size'], gpu_config['num_SMs'],\
                                gpu_config['l1_cache_size'], gpu_config['l1_cache_line_size'], gpu_config['l1_cache_associativity'],\
                                gpu_config['l2_cache_size'], gpu_config['l2_cache_line_size'], gpu_config['l2_cache_associativity'],\
                                gmem_reqs, avg_block_per_sm, block_per_sm_simulate)
    # rename
    memory_stats['l1_hit_rate'] = memory_stats['gmem_hit_rate']
    memory_stats['l2_hit_rate'] = memory_stats['hit_rate_l2']
    memory_stats['l2_hit_rate_ld'] = memory_stats['l2_hit_rate_st'] = memory_stats['hit_rate_l2']
    
    memory_stats['l1_hit_rate_ld'] = memory_stats['gmem_hit_rate_lds']
    memory_stats['l1_hit_rate_st'] = divide_or_zero(memory_stats['l1_hit_rate']*memory_stats['gmem_tot_trans'] - memory_stats['gmem_hit_rate_lds']*memory_stats['gmem_ld_trans'], memory_stats['gmem_st_trans'])
    memory_stats['gmem_ld_sectors'] = memory_stats['gmem_ld_trans']  # ppt-gpu has no sector level
    memory_stats['gmem_st_sectors'] = memory_stats['gmem_st_trans']
    memory_stats['gmem_tot_sectors'] = memory_stats['gmem_tot_trans']
    
    memory_stats['l2_ld_trans'] = memory_stats['l2_ld_trans_gmem']
    memory_stats['l2_st_trans'] = memory_stats['l2_st_trans_gmem']
    memory_stats['l2_tot_trans'] = memory_stats['l2_tot_trans_gmem']
    memory_stats['dram_ld_trans'] = memory_stats['dram_ld_trans_gmem']
    memory_stats['dram_st_trans'] = memory_stats['dram_st_trans_gmem']
    memory_stats['dram_tot_trans'] = memory_stats['dram_tot_trans_gmem']
    # gmem_tot_diverg
    
    return memory_stats

def process_dict_list(dict_list, op='sum', scale=1):
    dict_new = {}
    for key in dict_list[0].keys():
        try:
            if op=='avg':
                dict_new[key] = sum([d[key] for d in dict_list]) / len(dict_list) * scale
            elif op=='sum':
                dict_new[key] = sum([d[key] for d in dict_list]) * scale
        except:
            dict_new[key] = dict_list[0][key]
        
    return dict_new

def get_block(trace_dir, kernel_id, block_list):
    smi_blocks = []
    for bidx in block_list:
        block_trace_path = os.path.join(trace_dir, 'memory_traces', f"kernel_{kernel_id}_block_{bidx}.mem")
        if not os.path.exists(block_trace_path):
            smi_blocks.append([])
        else:
            with open(block_trace_path,'r') as f:
                block_trace = f.readlines()
            smi_blocks.append(block_trace)
    return smi_blocks

def run_L1(smi, trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm, gpu_config, is_sdcm, use_approx, granularity,
           filter_L2=False, block_mapping=BlockMapping.mod_block_mapping, sm_map=None, l1_dump_trace=False):
    # mapping block trace to SM
    if block_mapping != BlockMapping.sm_block_mapping:
        smi_blocks = get_block(trace_dir, kernel_id, sm_map[smi])
        smi_blocks_interleave = interleave_trace(smi_blocks)
    else:
        if smi >= len(sm_map):
            smi_blocks_interleave = []
        else:
            trace_path = os.path.join(trace_dir, 'memory_traces', f"kernel_{kernel_id}_sm_{sm_map[smi][0]}.mem")
            with open(trace_path,'r') as f:
                smi_blocks_interleave = f.readlines()
        # print(f"smi: {smi}, {sm_map[smi]}, len: {len(smi_blocks_interleave)}")
    
    sm_stats, smi_trace = process_trace(smi_blocks_interleave, gpu_config['l1_cache_line_size'], gpu_config['l1_sector_size']) # warp level to cache line level
    
    flag_active = False
    if smi_trace:
        flag_active = True
        if l1_dump_trace:
            with open(f'smi_trace_{smi}.json', 'w') as f:
                json.dump(smi_trace, f)
        # use sector size as cache line size
        if is_sdcm:
            l1_param = {'cache_line_size': gpu_config['l1_sector_size'],
                        'capacity': gpu_config['l1_cache_size'], 'associativity': gpu_config['l1_cache_associativity'],
                        'write_allocate': False, 'write_strategy': W.write_through}
            hit_rate_dict, L2_req = sdcm_model(smi_trace, l1_param,
                            use_approx=use_approx, granularity=granularity)
        else:
            l1_param = {'cache_line_size': gpu_config['l1_cache_line_size'], 'sector_size': gpu_config['l1_sector_size'],
                        'capacity': gpu_config['l1_cache_size'], 'associativity': gpu_config['l1_cache_associativity'],
                        'write_allocate': False, 'write_strategy': W.write_through}
            hit_rate_dict, L2_req = cache_simulate(smi_trace, l1_param, keep_traffic=True)
        if filter_L2:
            smi_trace = L2_req
        sm_stats['l1_hit_rate'] = hit_rate_dict['tot']
        sm_stats['gmem_ld_sectors_hit'] = sm_stats['gmem_ld_sectors'] * hit_rate_dict['ldg']
        sm_stats['gmem_st_sectors_hit'] = sm_stats['gmem_st_sectors'] * hit_rate_dict['stg']
        sm_stats['umem_ld_sectors_hit'] = sm_stats['umem_ld_sectors'] * hit_rate_dict['ld']
        sm_stats['umem_st_sectors_hit'] = sm_stats['umem_st_sectors'] * hit_rate_dict['st']
    
    return flag_active, sm_stats, smi_trace

def sdcm_model_warpper_parallel(kernel_id, trace_dir,
                launch_params,
                max_blocks_per_sm, 
                gpu_config, # gpu config
                is_sdcm=True,
                use_approx=True,
                granularity=2,
                filter_L2=False,
                l1_write_through=True,
                block_mapping=BlockMapping.mod_block_mapping,
                l1_dump_trace=False,l2_dump_trace=''):
    grid_size = launch_params['grid_size']
    num_SMs = gpu_config['num_SMs']
    active_sm = min(num_SMs, grid_size)
    
    # reuse distance model for L1
    sm_stats_list = []
    sm_traces = []

    avg_block_per_sm = (launch_params['grid_size'] + gpu_config['num_SMs'] - 1) // gpu_config['num_SMs']
    if granularity == 1:
        block_per_sm_simulate = 1
    elif granularity == 2:
        block_per_sm_simulate = max_blocks_per_sm
    elif granularity == 3:
        block_per_sm_simulate = avg_block_per_sm
    # scale
    scale = avg_block_per_sm / block_per_sm_simulate
    
    sm_map = [] # record block indexes mapped to each SM (or sm idx for sm trace)
    if block_mapping==BlockMapping.sm_block_mapping:
        # get all sm trace
        for trace_file in os.listdir(os.path.join(trace_dir, 'memory_traces')):
            if trace_file.startswith(f"kernel_{kernel_id}_sm_"):
                smi = int(trace_file.split('_')[-1].split('.')[0])
                sm_map.append([smi])
        scale = 1  # sm trace is already at sm level
    elif block_mapping==BlockMapping.mod_block_mapping:
        for smi in range(active_sm):
            smi_blocks = []
            for bidx in range(grid_size):
                if bidx % num_SMs == smi:
                    smi_blocks.append(bidx)
                    if len(smi_blocks) >= max_blocks_per_sm:
                        break
            sm_map.append(smi_blocks)
    elif block_mapping==BlockMapping.random_block_mapping:
        # all blocks index
        idx = list(range(grid_size))
        random.shuffle(idx)
        # each sm get block_per_sm_simulate block
        blocks_parts = [idx[i:i+block_per_sm_simulate] for i in range(0, len(idx), block_per_sm_simulate)]
        sm_map = blocks_parts[:active_sm]
    else:
        print("sm block mapping not supported")
        exit(1)
    # print(sm_map)
    
    num_jobs = min(active_sm, multiprocessing.cpu_count())
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(run_L1, i, trace_dir, kernel_id, grid_size, num_SMs, block_per_sm_simulate,
                                   gpu_config, is_sdcm, use_approx, granularity, filter_L2, block_mapping, sm_map, l1_dump_trace)
                   for i in range(active_sm)]
        for future in concurrent.futures.as_completed(futures):
            flag, sm_stats, smi_trace = future.result()
            if flag:
                sm_traces.append(smi_trace)
                sm_stats_list.append(sm_stats)
                
    # # serial for debugging
    # for i in range(active_sm):
    #     flag, sm_stats, smi_trace = run_L1(i, trace_dir, kernel_id, grid_size, num_SMs, block_per_sm_simulate,
    #                                        gpu_config, is_sdcm, use_approx, granularity, filter_L2, block_mapping, sm_map, l1_dump_trace)
    #     if flag:
    #         sm_traces.append(smi_trace)
    #         sm_stats_list.append(sm_stats)
    
    K = process_dict_list(sm_stats_list, op='sum', scale=scale)
    # caculate average L1 hit rate for all SMs
    K['l1_hit_rate_list'] = [sm_stats['l1_hit_rate'] for sm_stats in sm_stats_list]
    K['l1_hit_rate_ld'] = divide_or_zero(K['umem_ld_sectors_hit'], K['umem_ld_sectors'])
    K['l1_hit_rate_st'] = divide_or_zero(K['umem_st_sectors_hit'], K['umem_st_sectors'])
    K['l1_hit_rate'] = divide_or_zero((K['umem_ld_sectors_hit'] + K['umem_st_sectors_hit']), (K['umem_ld_sectors'] + K['umem_st_sectors']))
    K['l1_hit_rate_ldg'] = divide_or_zero(K['gmem_ld_sectors_hit'], K['gmem_ld_sectors'])
    K['l1_hit_rate_stg'] = divide_or_zero(K['gmem_st_sectors_hit'], K['gmem_st_sectors'])
    K['l1_hit_rate_g'] = divide_or_zero((K['gmem_ld_sectors_hit'] + K['gmem_st_sectors_hit']), (K['gmem_ld_sectors'] + K['gmem_st_sectors']))
    
    # reuse distance model for L2
    l2_trace = interleave_trace(sm_traces)
    if l2_dump_trace:
        with open(l2_dump_trace.replace('.csv', '.json'), 'w') as f:
            json.dump(l2_trace, f)
    
    l2_param = {'capacity': gpu_config['l2_cache_size'], 'cache_line_size': gpu_config['l2_cache_line_size'], 'sector_size': gpu_config['l2_sector_size'], 'associativity': gpu_config['l2_cache_associativity'],
                'write_allocate': True, 'write_strategy': W.write_back}
    if is_sdcm:
        l2_hit_rate_dict, _ = sdcm_model(l2_trace, l2_param, dump_trace=l2_dump_trace)
        K['l2_hit_rate'] = l2_hit_rate_dict['tot']
        K['l2_hit_rate_ld'] = l2_hit_rate_dict['ld']
        K['l2_hit_rate_st'] = l2_hit_rate_dict['st']
        K['l2_ld_reqs'] = 0
        K['l2_st_reqs'] =0
    else:
        l2_hit_rate_dict, _ = cache_simulate(l2_trace, l2_param, dump_trace=l2_dump_trace)
        # K['l2_hit_rate'] = l2_hit_rate_dict['tot']
        # K['l2_hit_rate_ld'] = l2_hit_rate_dict['ld']
        # K['l2_hit_rate_st'] = l2_hit_rate_dict['st']
        K['l2_hit_rate'] = l2_hit_rate_dict['tot_tag']
        K['l2_hit_rate_ld'] = l2_hit_rate_dict['ld_tag']
        K['l2_hit_rate_st'] = l2_hit_rate_dict['st_tag']
        K['l2_hit_rate_tag'] = l2_hit_rate_dict['tot_tag']
        K['l2_hit_rate_ld_tag'] = l2_hit_rate_dict['ld_tag']
        K['l2_hit_rate_st_tag'] = l2_hit_rate_dict['st_tag']
        K['l2_ld_reqs'] = 0
        K['l2_st_reqs'] = 0
    
    # L1/TEX
    K['gmem_tot_reqs'] = K['gmem_ld_reqs'] + K['gmem_st_reqs']
    # K['gmem_tot_trans'] = K['gmem_ld_trans'] + K['gmem_st_trans']
    K['gmem_tot_sectors'] = K['gmem_ld_sectors'] + K['gmem_st_sectors']
    K['gmem_ld_diverg'] = K['gmem_ld_sectors'] / K['gmem_ld_reqs'] if K['gmem_ld_reqs'] > 0 else 0
    K['gmem_st_diverg'] = K['gmem_st_sectors'] / K['gmem_st_reqs'] if K['gmem_st_reqs'] > 0 else 0
    K['gmem_tot_diverg'] = K['gmem_tot_sectors'] / K['gmem_tot_reqs'] if K['gmem_tot_reqs'] > 0 else 0
    
    K['umem_tot_reqs'] = K['umem_ld_reqs'] + K['umem_st_reqs']
    # K['umem_tot_trans'] = K['umem_ld_trans'] + K['umem_st_trans']
    K['umem_tot_sectors'] = K['umem_ld_sectors'] + K['umem_st_sectors']
    
    # L2
    if is_sdcm:
        K['l2_ld_trans'] = K['umem_ld_sectors'] * (1 - K['l1_hit_rate_ld'])
        if l1_write_through:
            K['l2_st_trans'] = K['umem_st_sectors']
        else:
            K['l2_st_trans'] = K['umem_st_sectors'] * (1 - K['l1_hit_rate_st'])
    else:
        K['l2_ld_trans'] = l2_hit_rate_dict['sectors_ld'] * scale
        K['l2_st_trans'] = l2_hit_rate_dict['sectors_st'] * scale
    
    K['l2_tot_trans'] = K['l2_ld_trans'] + K['l2_st_trans']
    
    # DRAM
    if is_sdcm:
        K["dram_tot_trans"] = K["l2_tot_trans"] * (1 - K["l2_hit_rate"])
        K["dram_ld_trans"] = K["l2_ld_trans"] * (1 - K["l2_hit_rate_ld"])
        K["dram_st_trans"] = K["l2_st_trans"] * (1 - K["l2_hit_rate_st"])
    else:
        K['dram_ld_trans'] = l2_hit_rate_dict['sectors_ld_nextlv'] * scale
        K['dram_st_trans'] = l2_hit_rate_dict['sectors_st_nextlv'] * scale
        K['dram_tot_trans'] = K['dram_ld_trans'] + K['dram_st_trans']
    K['scale'] = scale

    # debug print
    if kernel_id==1:
        print("L1/TEX Cache")
        tb = pt.PrettyTable()
        tb.field_names = ["Type", "Instr/Requests", "Sectors", "Sectors/Req", "Hit Rate", "Bytes", "Sector Misses to L2"]
        tb.add_row(["Global Load", K['gmem_ld_reqs'], K['gmem_ld_sectors'], K['gmem_ld_diverg'], K['l1_hit_rate_ld'], K['gmem_ld_sectors']*gpu_config['l1_sector_size'], 0])
        tb.add_row(["Global Store", K['gmem_st_reqs'], K['gmem_st_sectors'], K['gmem_st_diverg'], K['l1_hit_rate_st'], K['gmem_st_sectors']*gpu_config['l1_sector_size'], 0])
        tb.add_row(["Total", K['gmem_tot_reqs'], K['gmem_tot_sectors'], K['gmem_tot_diverg'], K['l1_hit_rate'], K['gmem_tot_sectors']*gpu_config['l1_sector_size'], 0])
        print(tb)
        print("L2 Cache")
        tb = pt.PrettyTable()
        tb.field_names = ["Type", "Requests", "Sectors", "Sectors/Req", "Hit Rate", "Bytes", "Sector Misses to Device"]
        tb.add_row(["L1/TEX Load", K['l2_ld_reqs'], K['l2_ld_trans'],  divide_or_zero(K['l2_ld_trans'],K['l2_ld_reqs']), K['l2_hit_rate_ld'], K['l2_ld_trans']*gpu_config['l2_sector_size'], 0])
        tb.add_row(["L1/TEX Store", K['l2_st_reqs'], K['l2_st_trans'], divide_or_zero(K['l2_st_trans'],K['l2_st_reqs']), K['l2_hit_rate_st'], K['l2_st_trans']*gpu_config['l2_sector_size'], 0])
        tb.add_row(["L1/TEX Total", K['l2_ld_reqs']+K['l2_st_reqs'], K['l2_tot_trans'], divide_or_zero(K['l2_tot_trans'],K['l2_ld_reqs']+K['l2_st_reqs']), K['l2_hit_rate'], K['l2_tot_trans']*gpu_config['l2_sector_size'], 0])
        print(tb)
        print("Device Memory")
        tb = pt.PrettyTable()
        tb.field_names = ["Type", "Sectors", "Bytes"]
        tb.add_row(["Load", K['dram_ld_trans'],  K['dram_ld_trans']*32])
        tb.add_row(["Store", K['dram_st_trans'], K['dram_st_trans']*32])
        tb.add_row(["Total", K['dram_tot_trans'],K['dram_tot_trans']*32])
        print(tb)
    
    return K

def memory_model_warpper(gpu_model, app_path, model, kernel_id=-1, granularity=2,
                         use_approx=True, filter_L2=False, block_mapping=BlockMapping.mod_block_mapping,
                         l1_dump_trace=False, l2_dump_trace=''):
    gpu_config = get_gpu_config(gpu_model).uarch
    kernels_launch_params = get_kernels_launch_params(app_path)

    app_res = []
    
    if kernel_id != -1:
        kernels_launch_params = [kernels_launch_params[kernel_id-1]]
    
    for kernel_param in kernels_launch_params:
        occupancy_res = get_max_active_block_per_sm(gpu_config['cc_configs'], kernel_param, gpu_config['num_SMs'], gpu_config['shared_mem_size'])
        
        # print(f"kernel {kernel_param['kernel_id']} start")
        
        if model == 'ppt-gpu':
            kernel_res = ppt_gpu_model_warpper(kernel_param['kernel_id'], app_path, kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config,
                                        granularity=granularity)
        elif model == 'sdcm':
            kernel_res = sdcm_model_warpper_parallel(kernel_param['kernel_id'], app_path, kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config,
                                        granularity=granularity, block_mapping=block_mapping, use_approx=use_approx, filter_L2=filter_L2,
                                        l1_dump_trace=l1_dump_trace, l2_dump_trace=l2_dump_trace)
        elif model == 'simulator':
            kernel_res = sdcm_model_warpper_parallel(kernel_param['kernel_id'], app_path, kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config,
                                        is_sdcm=False, granularity=granularity, block_mapping=block_mapping, filter_L2=filter_L2,
                                        l1_dump_trace=l1_dump_trace, l2_dump_trace=l2_dump_trace)
        else:
            raise ValueError(f"model {model} is not supported")
        
        kernel_res.update(kernel_param) # kernel name, grid size etc
        app_res.append(kernel_res)
    
    return app_res, gpu_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ppt-gpu memory model'
    )
    parser.add_argument('-a', "--app", dest="app_path",
                    required=True,
                    help='your application trace path')
    parser.add_argument('-c', "--config",
                    required=True,
                    help='target GPU hardware configuration')
    parser.add_argument("--granularity",
                    type=int,
                    default=2,
                    help='1=One Thread Block per SM or 2=Active Thread Blocks per SM or 3=All Thread Blocks per SM')
    parser.add_argument('-k', "--kernel", dest="kernel_id",
                    type=int,
                    default=-1,
                    help='(1 based index) To choose a specific kernel, add the kernel id')
    parser.add_argument('-M', "--model",
                    choices=['ppt-gpu', 'sdcm', 'simulator'],
                    default='ppt-gpu',
                    help='change memory model')
    parser.add_argument('--use-approx', 
                    action='store_true',
                    help='sdcm use approx')
    parser.add_argument('--filter-l2', 
                        action='store_true',
                        help='L1 hit bypass L2')
    parser.add_argument('--use-sm-trace', 
                        action='store_true',
                        help='use sm level trace')
    parser.add_argument('--block-mapping',
                    type=int,
                    default=BlockMapping.mod_block_mapping,
                    help='chose different block mapping strategy. 1: mod, 2: randoom, 3: use sm trace instead(hw true mapping)')
    args = parser.parse_args()
    if args.use_sm_trace:
        args.block_mapping = BlockMapping.sm_block_mapping
    
    l1_dump_trace, l2_dump_trace = False, ''
    # l1_dump_trace, l2_dump_trace = True, 'l2_trace.csv'
    app_res, _ = memory_model_warpper(args.config, args.app_path, args.model, kernel_id=args.kernel_id, granularity=args.granularity, use_approx=args.use_approx,
                        filter_L2=args.filter_l2, block_mapping=args.block_mapping,
                        l1_dump_trace=l1_dump_trace, l2_dump_trace=l2_dump_trace)
    print(app_res)
    print("Done")

if __name__ == "__main__2":
    with open('l2_trace.json') as f:
        smi_trace_ = json.load(f)
    smi_trace = smi_trace_
    
    # cache_parameter = {'capacity':  64*1024,  'cache_line_size': 32,'associativity': 4}
    cache_parameter = {'capacity':  4.5 * 1024*1024,  'cache_line_size': 128,'associativity': 32}
    
    # hit_rate_dict1 = sdcm_model(smi_trace, cache_parameter)
    # print(hit_rate_dict1)
    hit_rate_dict2, L2_req = cache_simulate(smi_trace, cache_parameter)
    print(hit_rate_dict2)
    print("Done")