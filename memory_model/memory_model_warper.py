import argparse
import os
import sys

curr_dir = os.path.dirname(__file__)
par_dir = os.path.dirname(curr_dir)
sys.path.insert(0, os.path.abspath(par_dir))

from src.memory_model import interleave_trace, get_memory_perf
from src.kernels import get_max_active_block_per_sm
from ppt import get_gpu_config, get_kernels_launch_params

from src.sdcm import sdcm_model, process_trace
from src.cache_simulator import LRUCache, cache_simulate
import concurrent.futures
import multiprocessing

def ppt_gpu_model_warpper(kernel_id, trace_dir,
                         launch_params,
                         max_blocks_per_sm, 
                         gpu_config, # gpu config
                         granularity = 2,
                         use_sm_trace=False,
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
    memory_stats['l1_hit_rate'] = memory_stats['umem_hit_rate'] * 100
    memory_stats['l2_hit_rate'] = memory_stats['hit_rate_l2'] * 100
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

def get_block(smi, trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm):
    smi_blocks = []
    for bidx in range(grid_size):
        if bidx % num_SMs == smi:
            block_trace_path = os.path.join(trace_dir, 'memory_traces', f"kernel_{kernel_id}_block_{bidx}.mem")
            if not os.path.exists(block_trace_path):
                smi_blocks.append([])
            else:
                with open(block_trace_path,'r') as f:
                    block_trace = f.readlines()
                smi_blocks.append(block_trace)
            if len(smi_blocks) >= max_blocks_per_sm:
                break
    return smi_blocks

def run_L1(smi, trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm, gpu_config, is_sdcm, use_approx, granularity, filter_L2=False, use_sm_trace=False):
    # mapping block trace to SM
    if not use_sm_trace:
        smi_blocks = get_block(smi, trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm)
        smi_blocks_interleave = interleave_trace(smi_blocks)
    else:
        trace_path = os.path.join(trace_dir, 'memory_traces', f"kernel_{kernel_id}_sm_{smi}.mem")
        if not os.path.exists(trace_path):
            smi_blocks_interleave = []
        else:
            with open(trace_path,'r') as f:
                smi_blocks_interleave = f.readlines()
    
    inst_count = {}
    sm_stats, smi_trace = process_trace(smi_blocks_interleave, gpu_config['l1_cache_line_size']) # warp level to cache line level
    
    flag_active = False
    hit_rate = 0
    if smi_trace:
        flag_active = True
        if is_sdcm:
            hit_rate_dict = sdcm_model(smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']},
                            use_approx=use_approx, granularity=granularity)
            sm_stats['l1_hit_rate'] = hit_rate_dict['tot']
            sm_stats['l1_hit_rate_ld'] = hit_rate_dict['ld']
            sm_stats['l1_hit_rate_st'] = hit_rate_dict['st']
            sm_stats['umem_ld_sectors_hit'] = sm_stats['umem_ld_sectors'] * hit_rate_dict['ld']
            sm_stats['umem_ld_sectors_miss'] = sm_stats['umem_ld_sectors'] - sm_stats['umem_ld_sectors_hit']
            sm_stats['umem_st_sectors_hit'] = sm_stats['umem_st_sectors'] * hit_rate_dict['st']
            sm_stats['umem_st_sectors_miss'] = sm_stats['umem_st_sectors'] - sm_stats['umem_st_sectors_hit']
        else:
            hit_rate_dict, L2_req = cache_simulate(smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']})
            if filter_L2:
                smi_trace = L2_req
            sm_stats['l1_hit_rate'] = hit_rate_dict['tot']
            sm_stats['l1_hit_rate_ld'] = hit_rate_dict['ld']
            sm_stats['l1_hit_rate_st'] = hit_rate_dict['st']
            sm_stats['umem_ld_sectors_hit'] = sm_stats['umem_ld_sectors'] * hit_rate_dict['ld']
            sm_stats['umem_ld_sectors_miss'] = sm_stats['umem_ld_sectors'] - sm_stats['umem_ld_sectors_hit']
            sm_stats['umem_st_sectors_hit'] = sm_stats['umem_st_sectors'] * hit_rate_dict['st']
            sm_stats['umem_st_sectors_miss'] = sm_stats['umem_st_sectors'] - sm_stats['umem_st_sectors_hit']
            
    return flag_active, sm_stats, smi_trace

def sdcm_model_warpper_parallel(kernel_id, trace_dir,
                launch_params,
                max_blocks_per_sm, 
                gpu_config, # gpu config
                is_sdcm=True,
                use_approx=True,
                granularity=2,
                filter_L2=False,
                use_sm_trace=False,
                l1_write_through=True):
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
    
    num_jobs = min(active_sm, multiprocessing.cpu_count())
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(run_L1, i, trace_dir, kernel_id, grid_size, num_SMs, block_per_sm_simulate, gpu_config, is_sdcm, use_approx, granularity, filter_L2, use_sm_trace)
                   for i in range(active_sm)]
        for future in concurrent.futures.as_completed(futures):
            flag, sm_stats, smi_trace = future.result()
            if flag:
                sm_traces.append(smi_trace)
                sm_stats_list.append(sm_stats)
    # serial for debugging
    # for i in range(active_sm):
    #     flag, sm_stats, smi_trace = run_L1(i, trace_dir, kernel_id, grid_size, num_SMs, block_per_sm_simulate, gpu_config, is_sdcm, use_approx, granularity, filter_L2, use_sm_trace)
    #     if flag:
    #         sm_traces.append(smi_trace)
    #         sm_stats_list.append(sm_stats)
    
    K = process_dict_list(sm_stats_list, op='sum', scale=scale)
    K['l1_hit_rate_list'] = [sm_stats['l1_hit_rate'] for sm_stats in sm_stats_list]
    K['l1_hit_rate_ld'] = K['umem_ld_sectors_hit'] / K['umem_ld_sectors'] if K['umem_ld_sectors'] > 0 else 0
    K['l1_hit_rate_st'] = K['umem_st_sectors_hit'] / K['umem_st_sectors'] if K['umem_st_sectors'] > 0 else 0
    K['l1_hit_rate'] = (K['umem_ld_sectors_hit'] + K['umem_st_sectors_hit']) / (K['umem_ld_sectors'] + K['umem_st_sectors']) if K['umem_ld_sectors'] + K['umem_st_sectors'] > 0 else 0
    
    # reuse distance model for L2
    l2_trace = interleave_trace(sm_traces)
    
    l2_param = {'capacity': gpu_config['l2_cache_size'], 'cache_line_size': gpu_config['l2_cache_line_size'], 'associativity': gpu_config['l2_cache_associativity']}
    if is_sdcm:
        l2_hit_rate_dict = sdcm_model(l2_trace, l2_param)
        K['l2_hit_rate'] = l2_hit_rate_dict['tot']
    else:
        l2_hit_rate_dict, _ = cache_simulate(l2_trace, l2_param)
        K['l2_hit_rate'] = l2_hit_rate_dict['tot']
    
    # global memory
    K['gmem_tot_reqs'] = K['gmem_ld_reqs'] + K['gmem_st_reqs']
    K['gmem_tot_trans'] = K['gmem_ld_trans'] + K['gmem_st_trans']
    K['umem_tot_reqs'] = K['umem_ld_reqs'] + K['umem_st_reqs']
    K['umem_tot_trans'] = K['umem_ld_trans'] + K['umem_st_trans']
    K['gmem_ld_diverg'] = K['gmem_ld_trans'] / K['gmem_ld_reqs'] if K['gmem_ld_reqs'] > 0 else 0
    K['gmem_st_diverg'] = K['gmem_st_trans'] / K['gmem_st_reqs'] if K['gmem_st_reqs'] > 0 else 0
    K['gmem_tot_diverg'] = K['gmem_tot_trans'] / K['gmem_tot_reqs'] if K['gmem_tot_reqs'] > 0 else 0
    # l2
    K['l2_ld_trans_gmem'] = K['gmem_ld_trans'] * (1 - K['l1_hit_rate_ld'])
    if l1_write_through:
        K['l2_st_trans_gmem'] = K['gmem_st_trans']                                 # write through
    else:
        write_ratio = K['gmem_st_trans'] / K['gmem_tot_trans']
        l1_hit_rate_st = (K['l1_hit_rate'] - K['l1_hit_rate_ld'] * (1 - write_ratio))/write_ratio
        K['l2_st_trans_gmem'] = K['gmem_st_trans'] * (1 - l1_hit_rate_st)
    K['l2_tot_trans_gmem'] = K['l2_ld_trans_gmem'] + K['l2_st_trans_gmem']
    # DRAM
    K["dram_tot_trans_gmem"] = K["l2_tot_trans_gmem"] * (1 - K["l2_hit_rate"])
    K["dram_ld_trans_gmem"] =  K["l2_ld_trans_gmem"] * (1 - K["l2_hit_rate"])
    K["dram_st_trans_gmem"] =  K["l2_st_trans_gmem"] * (1 - K["l2_hit_rate"])
    
    K['l1_hit_rate'] *= 100
    K['l2_hit_rate'] *= 100
    return K

def memory_model_warpper(gpu_model, app_path, model, kernel_id=-1, granularity=2, use_sm_trace=False, 
                         use_approx=True, filter_L2=False):
    gpu_config = get_gpu_config(gpu_model).uarch
    kernels_launch_params = get_kernels_launch_params(app_path)

    app_res = []
    
    if kernel_id != -1:
        kernels_launch_params = [kernels_launch_params[kernel_id-1]]
    
    for kernel_param in kernels_launch_params:
        occupancy_res = get_max_active_block_per_sm(gpu_config['cc_configs'], kernel_param, gpu_config['num_SMs'], gpu_config['shared_mem_size'])
        
        print(f"kernel {kernel_param['kernel_id']} start")
        
        if model == 'ppt-gpu':
            kernel_res = ppt_gpu_model_warpper(kernel_param['kernel_id'], app_path, kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config,
                                        granularity=granularity, use_sm_trace=use_sm_trace)
        elif model == 'sdcm':
            kernel_res = sdcm_model_warpper_parallel(kernel_param['kernel_id'], app_path, kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config,
                                        granularity=granularity, use_sm_trace=use_sm_trace, use_approx=use_approx)
        elif model == 'simulator':
            kernel_res = sdcm_model_warpper_parallel(kernel_param['kernel_id'], app_path, kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config,
                                        is_sdcm=False, granularity=granularity, use_sm_trace=use_sm_trace, filter_L2=filter_L2)
        else:
            raise ValueError(f"model {model} is not supported")
        
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
    parser.add_argument('--use-approx', 
                    action='store_true',
                    help='sdcm use approx')
    parser.add_argument('-k', "--kernel", dest="kernel_id",
                    type=int,
                    default=-1,
                    help='(1 based index) To choose a specific kernel, add the kernel id')
    parser.add_argument('-M', "--model",
                    choices=['ppt-gpu', 'sdcm', 'simulator'],
                    default='ppt-gpu',
                    help='change memory model')
    parser.add_argument('--use-sm-trace', 
                    action='store_true',
                    help='use sm level trace')
    args = parser.parse_args()
    
    app_res, _ = memory_model_warpper(args.config, args.app_path, args.model, kernel_id=args.kernel_id, granularity=args.granularity, use_sm_trace=args.use_sm_trace)
    print(app_res)
    print("Done")
    