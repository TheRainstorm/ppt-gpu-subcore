import argparse
import os
import sys

curr_dir = os.path.dirname(__file__)
par_dir = os.path.dirname(curr_dir)
sys.path.insert(0, os.path.abspath(par_dir))

from src.memory_model import get_memory_perf
from src.kernels import get_max_active_block_per_sm
from ppt import get_gpu_config,get_app_config,get_current_kernel_info

def ppt_gpu_memory_model(kernel_id, trace_dir, grid_size, # kernel_info,
                         max_blocks_per_sm, 
                         num_SMs, l1_param, l2_param, # gpu config
                        ):
    mem_traces_dir_path = os.path.join(trace_dir, 'memory_traces')

    gmem_reqs = 1  # e
    avg_block_per_sm = (grid_size + num_SMs - 1) // num_SMs
    memory_stats = get_memory_perf(kernel_id, mem_traces_dir_path, grid_size, num_SMs,\
                                l1_param['cache_size'], l1_param['cache_line_size'], l1_param['cache_associativity'],\
                                l2_param['cache_size'], l2_param['cache_line_size'], l2_param['cache_associativity'],\
                                gmem_reqs, avg_block_per_sm, max_blocks_per_sm)
    print(memory_stats)
    return [memory_stats['umem_hit_rate'], memory_stats['hit_rate_l2']]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ppt-gpu memory model'
    )
    parser.add_argument('-a', "--app", dest="app_path",
                    required=True,
                    help='your application trace path')
    parser.add_argument("--sass",
                    action="store_true",
                    help='Use SASS instruction trace, otherwise PTX')
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
    args = parser.parse_args()
    
    gpu_config = get_gpu_config(args.config)
    app_config = get_app_config(args.app_path)
    
    
    kernel_info_list = []
    instructions_type = "SASS" if args.sass else "PTX"
    if args.kernel_id == -1: # sim all kernel
        for kernel_id in app_config.app_kernels_id:
            kernel_info_list.append(get_current_kernel_info(kernel_id, args.app_path, args.app_path, app_config, instructions_type, args.granularity))
    else:
        kernel_info_list.append(get_current_kernel_info(str(args.kernel_id), args.app_path, args.app_path, app_config, instructions_type, args.granularity))

    
    kernel_res = []
    
    for i in range(len(kernel_info_list)):
        occupancy_res = get_max_active_block_per_sm(gpu_config.uarch['cc_configs'], kernel_info_list[i], gpu_config.uarch['num_SMs'], gpu_config.uarch['shared_mem_size'])
        
        l1_param = {
            'cache_size': gpu_config.uarch['l1_cache_size'],
            'cache_line_size': gpu_config.uarch['l1_cache_line_size'],
            'cache_associativity': gpu_config.uarch['l1_cache_associativity']
        }
        l2_param = {
            'cache_size': gpu_config.uarch['l2_cache_size'],
            'cache_line_size': gpu_config.uarch['l2_cache_line_size'],
            'cache_associativity': gpu_config.uarch['l2_cache_associativity']
        }
        
        l1_hit_rate, l2_hit_rate = ppt_gpu_memory_model(kernel_info_list[i]['kernel_id'], args.app_path, kernel_info_list[i]['grid_size'],
                             occupancy_res['max_active_block_per_sm'],
                             gpu_config.uarch['num_SMs'], l1_param, l2_param)
        
        kernel_res.append([l1_hit_rate, l2_hit_rate])
    
    print(kernel_res)
    print("Done")
    