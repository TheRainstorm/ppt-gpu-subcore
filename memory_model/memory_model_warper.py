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

def sdcm_model_warpper(kernel_id, trace_dir,
                launch_params,
                max_blocks_per_sm, 
                gpu_config, # gpu config
                model=sdcm_model,
                is_sdcm=True,
                use_approx=True,
                granularity=2):
    grid_size = launch_params['grid_size']
    num_SMs = gpu_config['num_SMs']
    active_sm = min(num_SMs, grid_size)
    # mapping block trace to SM
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
            if is_sdcm:
                hit_rate = model(smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']},
                                use_approx=use_approx, granularity=granularity)
            else:
                hit_rate = model(smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']})
            
            l1_hit_rate_list.append(hit_rate)

    print(f"kernel {kernel_id} inst_count: {inst_count}")
    avg_l1_hit_rate = sum(l1_hit_rate_list) / len(l1_hit_rate_list)
    
    # reuse distance model for L2
    l2_trace = interleave_trace(sm_traces)
    l2_hit_rate = model(l2_trace, {'capacity': gpu_config['l2_cache_size'], 'cache_line_size': gpu_config['l2_cache_line_size'], 'associativity': gpu_config['l2_cache_associativity']})
    return [avg_l1_hit_rate, l2_hit_rate]

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

def run_L1(smi, trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm, gpu_config, is_sdcm, use_approx, granularity, model):
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
            hit_rate = model(smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']},
                            use_approx=use_approx, granularity=granularity)
        else:
            hit_rate = model(smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']})
    
    return flag_active, hit_rate, smi_trace

def sdcm_model_warpper_parallel(kernel_id, trace_dir,
                launch_params,
                max_blocks_per_sm, 
                gpu_config, # gpu config
                model=sdcm_model,
                is_sdcm=True,
                use_approx=True,
                granularity=2):
    grid_size = launch_params['grid_size']
    num_SMs = gpu_config['num_SMs']
    active_sm = min(num_SMs, grid_size)
    
    import concurrent.futures
    import multiprocessing

    # reuse distance model for L1
    l1_hit_rate_list = []
    sm_traces = []
    inst_count = {}
    
    num_jobs = min(active_sm, multiprocessing.cpu_count())
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(run_L1, i, trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm, gpu_config, is_sdcm, use_approx, granularity, model)
                   for i in range(active_sm)]
        for future in concurrent.futures.as_completed(futures):
            flag, hit_rate, smi_trace = future.result()
            if flag:
                sm_traces.append(smi_trace)
                l1_hit_rate_list.append(hit_rate)
        
    avg_l1_hit_rate = sum(l1_hit_rate_list) / len(l1_hit_rate_list)

    # reuse distance model for L2
    l2_trace = interleave_trace(sm_traces)
    l2_hit_rate = model(l2_trace, {'capacity': gpu_config['l2_cache_size'], 'cache_line_size': gpu_config['l2_cache_line_size'], 'associativity': gpu_config['l2_cache_associativity']})
    
    return [avg_l1_hit_rate, l2_hit_rate]

def run_L1_filter(smi, trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm, gpu_config, is_sdcm, use_approx, granularity, model):
    smi_blocks = get_block(smi, trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm)
    
    smi_blocks_interleave = interleave_trace(smi_blocks)
    
    smi_trace = process_trace(smi_blocks_interleave, gpu_config['l1_cache_line_size']) # warp level to cache line level
    
    flag_active = False
    hit_rate = 0
    L2_req = []
    if smi_trace:
        flag_active = True
        cache_parameter = {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']}
        cache = LRUCache(cache_parameter)
        for inst_id, mem_id, warp_id, address in smi_trace:
            mem_width = 4
            write = inst_id == '1'
            addr = address * cache_parameter['cache_line_size']

            hit = cache.access(mem_width, write, addr)
            if not hit:
                L2_req.append([inst_id, mem_id, warp_id, address])
        
        hit_rate = cache.get_hit_info()['hit_ratio']
    return flag_active, hit_rate, L2_req

def sdcm_model_warpper_l2_filter(kernel_id, trace_dir,
                launch_params,
                max_blocks_per_sm, 
                gpu_config, # gpu config
                model=sdcm_model,
                is_sdcm=True,
                use_approx=True,
                granularity=2):
    grid_size = launch_params['grid_size']
    num_SMs = gpu_config['num_SMs']
    active_sm = min(num_SMs, grid_size)
    
    import concurrent.futures
    import multiprocessing

    # reuse distance model for L1
    l1_hit_rate_list = []
    sm_traces = []
    inst_count = {}
    
    num_jobs = min(active_sm, multiprocessing.cpu_count())
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(run_L1_filter, i, trace_dir, kernel_id, grid_size, num_SMs, max_blocks_per_sm, gpu_config, is_sdcm, use_approx, granularity, model)
                   for i in range(active_sm)]
        for future in concurrent.futures.as_completed(futures):
            flag, hit_rate, smi_trace = future.result()
            if flag:
                sm_traces.append(smi_trace)
                l1_hit_rate_list.append(hit_rate)
        
    avg_l1_hit_rate = sum(l1_hit_rate_list) / len(l1_hit_rate_list)

    # reuse distance model for L2
    l2_trace = interleave_trace(sm_traces)
    l2_hit_rate = model(l2_trace, {'capacity': gpu_config['l2_cache_size'], 'cache_line_size': gpu_config['l2_cache_line_size'], 'associativity': gpu_config['l2_cache_associativity']})
    
    return [avg_l1_hit_rate, l2_hit_rate]

def simulator_warpper(kernel_id, trace_dir,
                launch_params,
                max_blocks_per_sm, 
                gpu_config, # gpu config
                ):
    return sdcm_model_warpper_parallel(kernel_id, trace_dir, launch_params, max_blocks_per_sm, gpu_config,
                              model=cache_simulate, is_sdcm=False)

def simulator_warpper_l2_filter(kernel_id, trace_dir,
                launch_params,
                max_blocks_per_sm, 
                gpu_config, # gpu config
                ):
    return sdcm_model_warpper_l2_filter(kernel_id, trace_dir, launch_params, max_blocks_per_sm, gpu_config,
                              model=cache_simulate, is_sdcm=False)

def memory_model_warpper(gpu_model, app_path, model, kernel_id=-1, use_approx=True, granularity=2):
    gpu_config = get_gpu_config(gpu_model).uarch
    kernels_launch_params = get_kernels_launch_params(app_path)
    
    # select model
    if model == 'ppt-gpu':
        memory_model = ppt_gpu_model_warpper
    elif model == 'sdcm':
        memory_model = sdcm_model_warpper_parallel
    elif model == 'simulator':
        memory_model = simulator_warpper
    elif model == 'simulator_l2_filter':
        memory_model = simulator_warpper_l2_filter
    else:
        raise ValueError("Invalid model")
    
    app_res = []
    
    if kernel_id != -1:
        kernels_launch_params = [kernels_launch_params[kernel_id-1]]
    
    for kernel_param in kernels_launch_params:
        occupancy_res = get_max_active_block_per_sm(gpu_config['cc_configs'], kernel_param, gpu_config['num_SMs'], gpu_config['shared_mem_size'])
        
        print(f"kernel {kernel_param['kernel_id']} start")
        
        if model == 'sdcm':
            l1_hit_rate, l2_hit_rate = memory_model(kernel_param['kernel_id'], app_path, kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config,
                                        use_approx=use_approx, granularity=granularity)
        else:
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
    