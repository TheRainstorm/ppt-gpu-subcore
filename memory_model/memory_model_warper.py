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
from src.cache_simulator import cache_simulate

def ppt_gpu_model_warpper(kernel_id, trace_dir,
                         launch_params,
                         max_blocks_per_sm, 
                         gpu_config, # gpu config
                        ):
    mem_traces_dir_path = os.path.join(trace_dir, 'memory_traces')

    gmem_reqs = 1  # 
    avg_block_per_sm = (launch_params['grid_size'] + gpu_config['num_SMs'] - 1) // gpu_config['num_SMs']
    memory_stats = get_memory_perf(kernel_id, mem_traces_dir_path, launch_params['grid_size'], gpu_config['num_SMs'],\
                                gpu_config['l1_cache_size'], gpu_config['l1_cache_line_size'], gpu_config['l1_cache_associativity'],\
                                gpu_config['l2_cache_size'], gpu_config['l2_cache_line_size'], gpu_config['l2_cache_associativity'],\
                                gmem_reqs, avg_block_per_sm, max_blocks_per_sm)
    # print(memory_stats)
    return [memory_stats['umem_hit_rate'], memory_stats['hit_rate_l2']]

# import concurrent.futures
# import multiprocessing
# from joblib import Parallel, delayed

def map_block_to_sm(trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm):
    active_sm = min(num_SMs, grid_size)

    sm_blocks = []
    for smi in range(active_sm):
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
        sm_blocks.append(smi_blocks)
    return sm_blocks

def sdcm_model_warpper(kernel_id, trace_dir,
                launch_params,
                max_blocks_per_sm, 
                gpu_config, # gpu config
                model=sdcm_model):
    grid_size = launch_params['grid_size']
    num_SMs = gpu_config['num_SMs']
    active_sm = min(num_SMs, grid_size)
    # mapping block trace to SM
    sm_blocks = map_block_to_sm(trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm)
    
    # reuse distance model for L1
    l1_hit_rate_list = []
    sm_traces = []
    inst_count = {}
    for smi in range(active_sm):
        smi_blocks = sm_blocks[smi]
        smi_blocks_interleave = interleave_trace(smi_blocks)  # interleave at warp level
        smi_trace = process_trace(smi_blocks_interleave, gpu_config['l1_cache_line_size'], inst_count=inst_count) # warp level to cache line level
        if not smi_trace:
            continue   # don't count zero trace SM
        else:
            sm_traces.append(smi_trace)
            hit_rate = model(smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']})
            l1_hit_rate_list.append(hit_rate)
    
    # num_jobs = min(active_sm, multiprocessing.cpu_count())
    # # l1_hit_rate_list = Parallel(n_jobs=num_jobs, prefer="processes")(delayed(model)(smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']}) for i in range(active_sm))
    
    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
    #     futures = [executor.submit(model, smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']}) for i in range(active_sm)]
    #     for future in concurrent.futures.as_completed(futures):
    #         l1_hit_rate_list.append(future.result())
        
    # print(l1_hit_rate_list)
    print(f"kernel {kernel_id} inst_count: {inst_count}")
    avg_l1_hit_rate = sum(l1_hit_rate_list) / len(l1_hit_rate_list)
    
    # reuse distance model for L2
    l2_trace = interleave_trace(sm_traces)
    l2_hit_rate = model(l2_trace, {'capacity': gpu_config['l2_cache_size'], 'cache_line_size': gpu_config['l2_cache_line_size'], 'associativity': gpu_config['l2_cache_associativity']})
    # print(l2_hit_rate)
    
    return [avg_l1_hit_rate, l2_hit_rate]

def simulator_warpper(kernel_id, trace_dir,
                launch_params,
                max_blocks_per_sm, 
                gpu_config, # gpu config
                ):
    return sdcm_model_warpper(kernel_id, trace_dir, launch_params, max_blocks_per_sm, gpu_config, model=cache_simulate)

def memory_model_warpper(gpu_model, app_path, model, kernel_id=-1):
    gpu_config = get_gpu_config(gpu_model).uarch
    kernels_launch_params = get_kernels_launch_params(app_path)
    
    # select model
    if model == 'ppt-gpu':
        memory_model = ppt_gpu_model_warpper
    elif model == 'sdcm':
        memory_model = sdcm_model_warpper
    elif model == 'simulator':
        memory_model = simulator_warpper
    else:
        raise ValueError("Invalid model")
    
    app_res = []
    
    if kernel_id != -1:
        kernels_launch_params = [kernels_launch_params[kernel_id-1]]
    
    for kernel_param in kernels_launch_params:
        occupancy_res = get_max_active_block_per_sm(gpu_config['cc_configs'], kernel_param, gpu_config['num_SMs'], gpu_config['shared_mem_size'])
        
        print(f"kernel {kernel_param['kernel_id']} start")
        l1_hit_rate, l2_hit_rate = memory_model(kernel_param['kernel_id'], app_path, kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config)
        
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
                    default="2",
                    choices=["1", "2", "3"],
                    help='1=One Thread Block per SM or 2=Active Thread Blocks per SM or 3=All Thread Blocks per SM')
    parser.add_argument('-k', "--kernel", dest="kernel_id",
                    type=int,
                    default=-1,
                    help='(1 based index) To choose a specific kernel, add the kernel id')
    parser.add_argument('-M', "--model",
                    choices=['ppt-gpu', 'sdcm', 'simulator'],
                    default='ppt-gpu',
                    help='change memory model')
    args = parser.parse_args()
    
    app_res = memory_model_warpper(args.config, args.app_path, args.model, kernel_id=args.kernel_id)
    print(app_res)
    print("Done")
    