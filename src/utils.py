class Node():
    __slots__ = 'prev', 'next', 'data'
    def __init__(self, data):
        self.data = data

class LinkedList():
    def __init__(self):
        self.dummy = Node(None)
        self.dummy.prev = self.dummy
        self.dummy.next = self.dummy
        self.size = 0
    
    def remove(self, node):
        if self.size == 0:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1
        
    def append(self, data):
        x = Node(data)
        x.prev = self.dummy.prev
        x.next = self.dummy
        self.dummy.prev.next = x
        self.dummy.prev = x
        self.size += 1
        
    def __len__(self):
        return self.size
# import math
# def ceil(x, s=1):
# 	return s * math.ceil(float(x)/s)
from .helper_methods import ceil
import sys
import importlib
import os

def get_max_active_block_per_sm(cc, launch_data, num_SMs, shared_memory_per_sm, shared_mem_carveout=None, adaptive=False):
    # cc = {
    #     'warp_size': 32,
    #     'max_active_threads_per_SM': 2048,
    #     'max_active_blocks_per_SM': 32,
        
    #     'max_registers_per_SM': 65536,
    #     'register_allocation_size': 256,
        
    #     'smem_allocation_size': 256,
    # }
    
    # launch_data = {
    #     'grid_size': 1024,
    #     'block_size': 256,
    #     'num_regs': 22,  # reg_per_thread
    #     'smem_size': 0, # shared_memory_per_block
    # }
    
    warps_per_block = ceil(launch_data['block_size'] / cc['warp_size'])
    
    max_warps_per_sm = cc['max_active_threads_per_SM'] // cc['warp_size']
    block_per_sm_limit_warps = max_warps_per_sm // warps_per_block
    block_per_sm_limit_blocks = cc['max_active_blocks_per_SM']
    block_limit_warp_or_block = min(block_per_sm_limit_warps, block_per_sm_limit_blocks)
    
    if launch_data['num_regs'] == 0:
        block_per_sm_limit_regs = cc['max_active_blocks_per_SM']
    else:
        regs_per_warp = ceil(launch_data['num_regs'] * cc['warp_size'], cc['register_allocation_size'])
        regs_per_block = regs_per_warp * warps_per_block
        block_per_sm_limit_regs = cc['max_registers_per_SM'] // regs_per_block
    
    def get_smem_limit(shared_memory_per_sm):
        if launch_data['smem_size'] == 0:
            block_per_sm_limit_smem = cc['max_active_blocks_per_SM']
        else:
            smem_size = ceil(launch_data['smem_size'], cc['smem_allocation_size'])
            block_per_sm_limit_smem = shared_memory_per_sm // smem_size
        return block_per_sm_limit_smem
    
    th_max_active_block_per_sm = min(block_limit_warp_or_block, block_per_sm_limit_regs)
    
    if adaptive:
        # minimum smem size that not be a bottleneck
        # for smem_size in range(0, shared_memory_per_sm - 32*1024, 8*1024):  # l1 at least 32KB
        for smem_size_KB in shared_mem_carveout[:-1]:
            smem_size = smem_size_KB * 1024
            block_per_sm_limit_smem = get_smem_limit(smem_size)
            if block_per_sm_limit_smem >= th_max_active_block_per_sm:
                break
    else:
        smem_size = shared_memory_per_sm
        block_per_sm_limit_smem = get_smem_limit(smem_size)
    th_max_active_block_per_sm = min(th_max_active_block_per_sm, block_per_sm_limit_smem)
    
    th_active_warps = th_max_active_block_per_sm * warps_per_block
    th_occupancy = (th_active_warps / max_warps_per_sm) * 100
    
    allocted_block_per_sm = ceil(launch_data['grid_size'] / num_SMs)
    max_active_block_per_sm = min(th_max_active_block_per_sm, allocted_block_per_sm)
    
    occupancy_res = {
        'block_per_sm_limit_warps': block_per_sm_limit_warps,
        'block_per_sm_limit_blocks': block_per_sm_limit_blocks,
        'block_per_sm_limit_regs': block_per_sm_limit_regs,
        'block_per_sm_limit_smem': block_per_sm_limit_smem,
        
        'th_max_active_block_per_sm': th_max_active_block_per_sm,
        'th_active_warps': th_active_warps,
        'th_occupancy': th_occupancy,
        'allocted_block_per_sm': allocted_block_per_sm,
        'max_active_block_per_sm': max_active_block_per_sm,
        
        'adaptive_smem_size': smem_size,
    }
    
    return occupancy_res

def get_launch_params(kernel_id, app_config):
    kernel_id_name = f"kernel_{kernel_id}"
    launch_param_ = app_config[kernel_id_name]
    launch_param = {}
    launch_param['kernel_id'] = kernel_id
    launch_param['kernel_name'] = launch_param_["kernel_name"]
    launch_param['smem_size'] = launch_param_["shared_mem_bytes"]
    launch_param['grid_size'] = launch_param_["grid_size"]
    launch_param['block_size'] = launch_param_["block_size"]
    launch_param['num_regs'] = launch_param_["num_registers"]
    return launch_param

def get_kernels_launch_params(app_path):
    app_config = get_app_config(app_path)
    
    launch_param_list = []
    for kernel_id in app_config['app_kernels_id']:
        launch_param_list.append(get_launch_params(kernel_id, app_config))
    return launch_param_list
    
def get_current_kernel_info(kernel_id, app_name, app_path, app_config, instructions_type, granularity, app_res_ref=None, app_report_dir=None):

    current_kernel_info = {}

    current_kernel_info["app_report_dir"] = app_report_dir if app_report_dir else app_path
    current_kernel_info["kernel_id"] = str(kernel_id)
    current_kernel_info["granularity"] = granularity

    kernel_id_name = f"kernel_{kernel_id}"
    ###########################
    ## kernel configurations ##
    ###########################
    launch_param = get_launch_params(kernel_id, app_config)
    current_kernel_info.update(launch_param)
    
    ##################
    ## memory trace ##
    ##################
    # mem_trace_file = kernel_id_name+".mem"
    mem_trace_file = "memory_traces"
    mem_trace_file_path = os.path.join(app_path, mem_trace_file)

    if not os.path.exists(mem_trace_file_path):
        print(str("\n[Error]\n")+str("<<memory_traces>> directory doesn't exists in ")+app_name+str(" application directory"))
        sys.exit(1)
    current_kernel_info["mem_traces_dir_path"] = mem_trace_file_path
    current_kernel_info["trace_dir"] = app_path

    ################
    ## ISA Parser ##
    ################
    current_kernel_info["ptx_file_path"] = ""
    current_kernel_info["sass_file_path"] = ""

    if instructions_type == "PTX":
        ptx_file = "ptx_traces/"+kernel_id_name+".ptx"
        # if "/" in app_name:
        #     sass_file = app_name.split("/")[-1]+"ptx_traces/"+kernel_id_name+".ptx"
        ptx_file_path = os.path.join(app_path, ptx_file)

        if not os.path.isfile(ptx_file_path):
            print(str("\n[Error]\n")+str("ptx instructions trace file: <<")+str(sass_file)+str(">> doesn't exists in ")+app_name +\
                    str(" application directory"))
            sys.exit(1)
        
        current_kernel_info["ISA"] = 1
        current_kernel_info["ptx_file_path"] = ptx_file_path

    elif instructions_type == "SASS": 
        sass_file = "sass_traces/"+kernel_id_name+".sass"
        # if "/" in app_name:
        #     sass_file = app_name.split("/")[-1]+"sass_traces/"+kernel_id_name+".sass"
        sass_file_path = os.path.join(app_path, sass_file)

        if not os.path.isfile(sass_file_path):
            print(str("\n[Error]\n")+str("sass instructions trace file: <<")+str(sass_file)+str(">> doesn't exists in ")+app_name +\
                    str(" application directory"))
            sys.exit(1)

        current_kernel_info["ISA"] = 2
        current_kernel_info["sass_file_path"] = sass_file_path
    
    current_kernel_info['cache_ref_data'] = None
    if app_res_ref:
        kernel_res_ref = app_res_ref[int(kernel_id)-1]
        cache_ref_data = {}
        cache_ref_data["l1_hit_rate"] = min(1, kernel_res_ref["global_hit_rate"]/100)  # tex_cache_hit_rate
        cache_ref_data["l2_hit_rate"] = min(1, kernel_res_ref["l2_tex_hit_rate"]/100)
        current_kernel_info['cache_ref_data'] = cache_ref_data
    
    return current_kernel_info

def get_gpu_config(gpu_config):
    # sys.path.append(repo_path)
    # get hw configuaration
    try:
        gpu_configs = importlib.import_module("hardware."+gpu_config)
    except:
        print(str("\n[Error]\nGPU hardware config file provided doesn't exist\n", file=sys.stderr))
        sys.exit(1)
    
    # get Target ISA Latencies
    try:
        ISA = importlib.import_module("hardware.ISA."+gpu_configs.uarch["gpu_arch"])
    except:
        print("\n[Error]\nISA for <<"+gpu_configs.uarch["gpu_arch"]+">> doesn't exists in hardware/ISA directory", file=sys.stderr)
        sys.exit(1)
    
    c = gpu_configs.uarch
    initial_interval = {
        # Initiation interval (II) = threadsPerWarp // #FULanes
        "iALU"              :   32*4 // c['num_INT_units_per_SM'],
        "fALU"              :   32*4 // c['num_SP_units_per_SM'],
        "hALU"              :   32*4 // c['num_SP_units_per_SM'],
        "dALU"              :   32*4 // c['num_DP_units_per_SM'],

        "SFU"               :   32*4 // c['num_SF_units_per_SM'],
        "dSFU"              :   32*4 // c['num_SF_units_per_SM'],

        "LDST"              :   32*4 // c['num_LDS_units_per_SM'],
        
        # "bTCU"              :   64,
        "iTCU"              :   32*4 // c['num_TC_units_per_SM'],
        "hTCU"              :   32*4 // c['num_TC_units_per_SM'],
        "fTCU"              :   32*4 // c['num_TC_units_per_SM'],
        "dTCU"              :   32*4 // c['num_TC_units_per_SM'],
        
        "BRA"               :   1,
        "EXIT"              :   1,
    }
    
    # add ISA Latencies to gpu_configs
    gpu_configs.uarch["ptx_isa"] = ISA.ptx_isa
    gpu_configs.uarch["sass_isa"] = ISA.sass_isa
    gpu_configs.uarch["units_latency"] = ISA.units_latency
    gpu_configs.uarch["initial_interval"] = initial_interval
    
    # get cc
    try:
        compute_capability = importlib.import_module("hardware.compute_capability."+str(gpu_configs.uarch["compute_capabilty"]))
    except:
        print("\n[Error]\ncompute capabilty for <<"+gpu_configs.uarch["compute_capabilty"]+">> doesn't exists in hardware/compute_capabilty directory")
        sys.exit(1)
    
    gpu_configs.uarch["cc_configs"] = compute_capability.cc_configs
    
    return gpu_configs

def get_app_config(app_path):
    app_config_path = os.path.join(app_path, "app_config.py")
    app_config = {}
    with open(app_config_path, "r") as file:
        exec(file.read(), app_config)
    return app_config
    