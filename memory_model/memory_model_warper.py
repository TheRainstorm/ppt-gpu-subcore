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
    # print(memory_stats)
    return [memory_stats['umem_hit_rate'], memory_stats['hit_rate_l2']]

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

def run_L1(smi, trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm, gpu_config, is_sdcm, use_approx, granularity, filter_L2=False):
    # mapping block trace to SM
    smi_blocks = get_block(smi, trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm)
    
    inst_count = {}
    smi_blocks_interleave = interleave_trace(smi_blocks)
    smi_trace = process_trace(smi_blocks_interleave, gpu_config['l1_cache_line_size'], inst_count=inst_count) # warp level to cache line level
    
    flag_active = False
    hit_rate = 0
    if smi_trace:
        flag_active = True
        if is_sdcm:
            hit_rate = sdcm_model(smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']},
                            use_approx=use_approx, granularity=granularity)
        else:
            hit_rate, L2_req = cache_simulate(smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']})
            if filter_L2:
                smi_trace = L2_req
            
    return flag_active, hit_rate, smi_trace

def sdcm_model_warpper_parallel(kernel_id, trace_dir,
                launch_params,
                max_blocks_per_sm, 
                gpu_config, # gpu config
                is_sdcm=True,
                use_approx=True,
                granularity=2,
                filter_L2=False):
    grid_size = launch_params['grid_size']
    num_SMs = gpu_config['num_SMs']
    active_sm = min(num_SMs, grid_size)
    
    # reuse distance model for L1
    l1_hit_rate_list = []
    sm_traces = []

    avg_block_per_sm = (launch_params['grid_size'] + gpu_config['num_SMs'] - 1) // gpu_config['num_SMs']
    if granularity == 1:
        block_per_sm_simulate = 1
    elif granularity == 2:
        block_per_sm_simulate = max_blocks_per_sm
    elif granularity == 3:
        block_per_sm_simulate = avg_block_per_sm
    
    num_jobs = min(active_sm, multiprocessing.cpu_count())
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(run_L1, i, trace_dir, kernel_id, grid_size, num_SMs, block_per_sm_simulate, gpu_config, is_sdcm, use_approx, granularity, filter_L2)
                   for i in range(active_sm)]
        for future in concurrent.futures.as_completed(futures):
            flag, hit_rate, smi_trace = future.result()
            if flag:
                sm_traces.append(smi_trace)
                l1_hit_rate_list.append(hit_rate)
        
    avg_l1_hit_rate = sum(l1_hit_rate_list) / len(l1_hit_rate_list)

    # reuse distance model for L2
    l2_trace = interleave_trace(sm_traces)
    
    l2_param = {'capacity': gpu_config['l2_cache_size'], 'cache_line_size': gpu_config['l2_cache_line_size'], 'associativity': gpu_config['l2_cache_associativity']}
    if is_sdcm:
        l2_hit_rate = sdcm_model(l2_trace, l2_param)
    else:
        l2_hit_rate, _ = granularity=granularity(l2_trace, l2_param)
    
    return [avg_l1_hit_rate, l2_hit_rate]

def memory_model_warpper(gpu_model, app_path, model, kernel_id=-1,
                         use_approx=True, granularity=2, filter_L2=False):
    gpu_config = get_gpu_config(gpu_model).uarch
    kernels_launch_params = get_kernels_launch_params(app_path)

    app_res = []
    
    if kernel_id != -1:
        kernels_launch_params = [kernels_launch_params[kernel_id-1]]
    
    for kernel_param in kernels_launch_params:
        occupancy_res = get_max_active_block_per_sm(gpu_config['cc_configs'], kernel_param, gpu_config['num_SMs'], gpu_config['shared_mem_size'])
        
        print(f"kernel {kernel_param['kernel_id']} start")
        
        if model == 'ppt-gpu':
            l1_hit_rate, l2_hit_rate = ppt_gpu_model_warpper(kernel_param['kernel_id'], app_path, kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config)
        elif model == 'sdcm':
            l1_hit_rate, l2_hit_rate = sdcm_model_warpper_parallel(kernel_param['kernel_id'], app_path, kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config,
                                        use_approx=use_approx, granularity=granularity)
        elif model == 'simulator':
            l1_hit_rate, l2_hit_rate = sdcm_model_warpper_parallel(kernel_param['kernel_id'], app_path, kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config,
                                        is_sdcm=False, granularity=granularity, filter_L2=filter_L2)
        else:
            raise ValueError(f"model {model} is not supported")
        
        app_res.append({"l1_hit_rate": l1_hit_rate*100, "l2_hit_rate": l2_hit_rate*100})
    
    return app_res
    
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
    args = parser.parse_args()
    
    app_res = memory_model_warpper(args.config, args.app_path, args.model, kernel_id=args.kernel_id, granularity=args.granularity)
    print(app_res)
    print("Done")
    