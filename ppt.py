##############################################################################################################################################################################################################################################################
# &copy 2017. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

# All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, 
# irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

# Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, 
# the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works,
# such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
##############################################################################################################################################################################################################################################################

# Author: Yehia Arafa
# Last Update Date: April, 2021
# Copyright: Open source, must acknowledge original author

##########################################################

import json
import sys, os, importlib
# from simian import Simian, Entity
from src.kernels import Kernel
import argparse

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

def get_gpu_config(gpu_config, repo_path=None):
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
        
    # add ISA Latencies to gpu_configs
    gpu_configs.uarch["ptx_isa"] = ISA.ptx_isa
    gpu_configs.uarch["sass_isa"] = ISA.sass_isa
    gpu_configs.uarch["units_latency"] = ISA.units_latency
    gpu_configs.uarch["initial_interval"] = ISA.initial_interval
    
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
    
def main():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    parser = argparse.ArgumentParser(
        description='ppt-gpu. [MPI] For scalabilty, add mpirun call before program command:\nmpirun -np <number of processes>'
    )
    parser.add_argument('-a', "--app", dest="app_path",
                    required=True,
                    help='your application path')
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
    parser.add_argument("--mpi",
                    action="store_true",
                    help='use MPI')
    parser.add_argument("--libmpich-path",
                    default="/usr/lib/x86_64-linux-gnu/libmpich.so",
                    help="path to libmpich.so")
    parser.add_argument("-R", "--report-output-dir",
                        default="output",
                        help="output to a seprate dir, not in the trace dir")
    parser.add_argument("--hw-res",
                        help="hw res json file. use to fix l2 cache miss rate")
    args = parser.parse_args()
    granularity = args.granularity

    # check
    if not os.path.exists(args.app_path):
        print(f"app path {args.app_path} doesn't exist")
        sys.exit(1)
    
    gpu_configs = get_gpu_config(args.config)
    app_config = get_app_config(args.app_path)

    # read cache reference data
    app_res_ref = None
    app_name_ = args.app_path.split("/")[-2]
    app_arg_ = args.app_path.split("/")[-1]
    app_and_arg = f"{app_name_}/{app_arg_}"
    if args.hw_res:
        with open(args.hw_res) as fp:
            res_ref = json.load(fp)
        app_res_ref = res_ref[app_and_arg]

    # make report dir
    app_report_dir = args.app_path
    if args.report_output_dir:
        app_report_dir = os.path.join(args.report_output_dir, app_name_, app_arg_)
        if rank==0:
            if not os.path.exists(app_report_dir):
                os.makedirs(app_report_dir)
    
    kernels_info = []
    instructions_type = "SASS" if args.sass else "PTX"
    if args.kernel_id == -1: # sim all kernel
        for kernel_id in app_config['app_kernels_id']:
            kernels_info.append(get_current_kernel_info(int(kernel_id), args.app_path, args.app_path, app_config, instructions_type, args.granularity, app_res_ref=app_res_ref, app_report_dir=app_report_dir))
    else:
        kernels_info.append(get_current_kernel_info(kernel_id, args.app_path, args.app_path, app_config, instructions_type, args.granularity, app_res_ref=app_res_ref, app_report_dir=app_report_dir))

    # ############################
    # # Simian Engine parameters #
    # ############################
    # simianEngine = Simian("PPT-GPU", useMPI=True, opt=False, appPath = app_report_dir, ISA=instructions_type, granularity=granularity, mpiLibName=args.libmpich_path)
    gpuNode = GPUNode(gpu_configs.uarch, gpu_configs.uarch['cc_configs'], len(kernels_info))
        
    # for i in range (len(kernels_info)):
    #     k_id = i 
	# 	# Add Entity and sched Event only if Hash(Entity_name_i) % MPI.size == MPI.rank
    #     simianEngine.addEntity("Kernel", Kernel, k_id, len(kernels_info), gpuNode, kernels_info[i])
    #     simianEngine.schedService(1, "kernel_call", None, "Kernel", k_id)

    # simianEngine.run()
    # simianEngine.exit()
    if rank==0:
        print(f"Runing: {app_and_arg}")
    for i in range(len(kernels_info)):
        if rank == i%size:
            print(f"Kernel {i} is running on rank {rank}")
            kernel = Kernel(gpuNode, kernels_info[i])
            kernel.kernel_call()


class GPUNode(object):
	"""
	Class that represents a node that has a GPU
	"""
	def __init__(self, gpu_configs, gpu_configs_cc, num_kernels):
		self.num_accelerators = 1 # modeling a node that has 1 GPU for now
		self.accelerators = []
		self.gpu_configs = gpu_configs
		self.gpu_configs_cc = gpu_configs_cc
		#print("GPU node generated")
		self.generate_target_accelerators(num_kernels)

	#generate GPU accelerators inside the node
	def generate_target_accelerators(self, num_kernels):
		accelerators = importlib.import_module("src.accelerators")
		for i in range(self.num_accelerators):
			self.accelerators.append(accelerators.Accelerator(self, i, self.gpu_configs, self.gpu_configs_cc, num_kernels))

if __name__ == "__main__":
	main()

