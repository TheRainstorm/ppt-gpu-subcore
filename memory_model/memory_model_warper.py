import argparse
import os
import sys

curr_dir = os.path.dirname(__file__)
par_dir = os.path.dirname(curr_dir)
sys.path.insert(0, os.path.abspath(par_dir))

from src.memory_model import interleave_trace, get_memory_perf
from src.kernels import get_max_active_block_per_sm
from ppt import get_gpu_config,get_app_config,get_current_kernel_info, get_kernels_launch_params

from src.sdcm import get_cache_line_access_from_raw_trace, model, process_trace

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
                ):
    grid_size = launch_params['grid_size']
    num_SMs = gpu_config['num_SMs']
    # mapping block trace to SM
    sm_traces = []
    for smi in range(num_SMs):
        smi_blocks = []
        for bidx in range(grid_size):
            if bidx % num_SMs == smi:
                block_trace_path = os.path.join(trace_dir, 'memory_traces', f"kernel_{kernel_id}_block_{bidx}.mem")
                with open(block_trace_path,'r') as f:
                    block_trace = f.readlines()
                # block_trace = get_cache_line_access_from_raw_trace(block_trace_path, gpu_config['l1_cache_line_size'])
                smi_blocks.append(block_trace)
                if len(smi_blocks) >= max_blocks_per_sm:
                    break
        smi_blocks_interleave = interleave_trace(smi_blocks)
        smi_trace = process_trace(smi_blocks_interleave, gpu_config['l1_cache_line_size'])
        sm_traces.append(smi_trace)
    
    # reuse distance model for L1
    l1_hit_rate_list = []
    for smi in range(num_SMs):
        if smi==0 and kernel_id==1:
            with open('sdcm_l1_sm0_trace.txt', 'w') as f:
                for line in sm_traces[smi]:
                    for x in line:
                        f.write(f"{x} ")
                    f.write("\n")
                    # f.write(" ".join(line))
                    # f.write(" ".join(line)+"\n")
        smi_trace = sm_traces[smi]
        hit_rate = model(smi_trace, {'capacity': gpu_config['l1_cache_size'], 'cache_line_size': gpu_config['l1_cache_line_size'], 'associativity': gpu_config['l1_cache_associativity']})
        l1_hit_rate_list.append(hit_rate)
    print(l1_hit_rate_list)
    avg_l1_hit_rate = sum(l1_hit_rate_list) / len(l1_hit_rate_list)
    
    # reuse distance model for L2
    l2_trace = interleave_trace(sm_traces)
    l2_hit_rate = model(l2_trace, {'capacity': gpu_config['l2_cache_size'], 'cache_line_size': gpu_config['l2_cache_line_size'], 'associativity': gpu_config['l2_cache_associativity']})
    
    return [avg_l1_hit_rate, l2_hit_rate]

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
    parser.add_argument("--kernel", dest="kernel_id",
                    type=int,
                    default=-1,
                    help='(1 based index) To choose a specific kernel, add the kernel id')
    parser.add_argument('-M', "--model",
                    choices=['ppt-gpu', 'sdcm'],
                    default='ppt-gpu',
                    help='change memory model')
    args = parser.parse_args()
    
    gpu_config = get_gpu_config(args.config).uarch
    kernels_launch_params = get_kernels_launch_params(args.app_path)
    
    # select model
    if args.model == 'ppt-gpu':
        memory_model = ppt_gpu_model_warpper
    elif args.model == 'sdcm':
        memory_model = sdcm_model_warpper
    else:
        raise ValueError("Invalid model")
    
    kernel_res = []
    
    for i in range(len(kernels_launch_params)):
        occupancy_res = get_max_active_block_per_sm(gpu_config['cc_configs'], kernels_launch_params[i], gpu_config['num_SMs'], gpu_config['shared_mem_size'])
        
        l1_hit_rate, l2_hit_rate = memory_model(i+1, args.app_path, kernels_launch_params[i], occupancy_res['max_active_block_per_sm'], gpu_config)
        
        kernel_res.append([l1_hit_rate, l2_hit_rate])
    
    print(kernel_res)
    print("Done")
    