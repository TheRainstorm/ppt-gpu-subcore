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
import sys, os, getopt, importlib
# from simian import Simian, Entity
from src.kernels import Kernel
import argparse

def usage():
    print("\n[USAGE]\n\
    [option 1] To simulate all kernels of the application:\n\
 python ppt.py --app <your application path> --sass (or --ptx for PTX instruction trace)\
 --config <target GPU hardware configuration> --granularity (1=One Thread Block per SM or 2=Active Thread Blocks per SM or 3=All Thread Blocks per SM)\n\n\
    [option 2] To choose a specific kernel, add the kernel id:\n --kernel <target kernel id>\n\n\
   [MPI] For scalabilty, add mpirun call before program command:\nmpirun -np <number of processes>" )


def get_current_kernel_info(kernel_id, app_name, app_path, app_config, instructions_type, granularity, app_res_ref=None, app_report_dir=None):

    current_kernel_info = {}

    current_kernel_info["app_report_dir"] = app_report_dir if app_report_dir else app_path
    current_kernel_info["kernel_id"] = kernel_id
    current_kernel_info["granularity"] = granularity

    ###########################
    ## kernel configurations ##
    ###########################
    kernel_id_ = int(kernel_id)-1  # 0-index
    kernel_id = "kernel_"+kernel_id

    try:
        kernel_config = getattr(app_config, kernel_id)
    except:
        print(str("\n[Error]\n<<")+str(kernel_id)+str(">> doesn't exists in app_config file"))
        sys.exit(1)

    try:
        kernel_name = kernel_config["kernel_name"]
    except:
        print(str("\n[Error]\n")+str("\"kernel_name\" configuration is missing"))
        sys.exit(1)
    current_kernel_info["kernel_name"] = kernel_name

    try:
        kernel_smem_size = kernel_config["shared_mem_bytes"]
    except:
        print(str("\n[Error]\n")+str("\"shared_mem_bytes\" configuration is missing"))
        sys.exit(1)
    current_kernel_info["smem_size"] = kernel_smem_size

    try:
        kernel_grid_size = kernel_config["grid_size"]
    except:
        print(str("\n[Error]\n")+str("\"grid_size\" configuration is missing"))
        sys.exit(1)
    current_kernel_info["grid_size"] = kernel_grid_size

    try:
        kernel_block_size = kernel_config["block_size"]
    except:
        print(str("\n[Error]\n")+str("\"block_size\" configuration is missing"))
        sys.exit(1)
    current_kernel_info["block_size"] = kernel_block_size
    
    try:
        kernel_num_regs = kernel_config["num_registers"]
    except:
        print(str("\n[Error]\n")+str("\"num_registers\" configuration is missing"))
        sys.exit(1)
    current_kernel_info["num_regs"] = kernel_num_regs

    ##################
    ## memory trace ##
    ##################
    # mem_trace_file = kernel_id+".mem"
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
        ptx_file = "ptx_traces/"+kernel_id+".ptx"
        # if "/" in app_name:
        #     sass_file = app_name.split("/")[-1]+"ptx_traces/"+kernel_id+".ptx"
        ptx_file_path = os.path.join(app_path, ptx_file)

        if not os.path.isfile(ptx_file_path):
            print(str("\n[Error]\n")+str("ptx instructions trace file: <<")+str(sass_file)+str(">> doesn't exists in ")+app_name +\
                    str(" application directory"))
            sys.exit(1)
        
        current_kernel_info["ISA"] = 1
        current_kernel_info["ptx_file_path"] = ptx_file_path

    elif instructions_type == "SASS": 
        sass_file = "sass_traces/"+kernel_id+".sass"
        # if "/" in app_name:
        #     sass_file = app_name.split("/")[-1]+"sass_traces/"+kernel_id+".sass"
        sass_file_path = os.path.join(app_path, sass_file)

        if not os.path.isfile(sass_file_path):
            print(str("\n[Error]\n")+str("sass instructions trace file: <<")+str(sass_file)+str(">> doesn't exists in ")+app_name +\
                    str(" application directory"))
            sys.exit(1)

        current_kernel_info["ISA"] = 2
        current_kernel_info["sass_file_path"] = sass_file_path
    
    current_kernel_info['cache_ref_data'] = None
    if app_res_ref:
        kernel_res_ref = app_res_ref[kernel_id_]
        cache_ref_data = {}
        cache_ref_data["l1_hit_rate"] = min(1, kernel_res_ref["global_hit_rate"]/100)  # tex_cache_hit_rate
        cache_ref_data["l2_hit_rate"] = min(1, kernel_res_ref["l2_tex_hit_rate"]/100)
        current_kernel_info['cache_ref_data'] = cache_ref_data
    
    return current_kernel_info

def main():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    parser = argparse.ArgumentParser(
        description='ppt-gpu. [MPI] For scalabilty, add mpirun call before program command:\nmpirun -np <number of processes>'
    )
    parser.add_argument('-a', "--app", dest="app_name",
                    required=True,
                    help='your application path')
    parser.add_argument("--sass",
                    action="store_true",
                    help='SASS instruction trace')
    parser.add_argument("--ptx",
                    action="store_true",
                    help='PTX instruction trace')
    parser.add_argument('-c', "--config",
                    required=True,
                    help='target GPU hardware configuration')
    parser.add_argument("--granularity",
                    default="2",
                    choices=["1", "2", "3"],
                    help='1=One Thread Block per SM or 2=Active Thread Blocks per SM or 3=All Thread Blocks per SM')
    parser.add_argument("--kernel",
                    type=int,
                    default=-1,
                    help='To choose a specific kernel, add the kernel id')
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
    SASS = args.sass
    PTX = args.ptx
    kernel_id = args.kernel
    granularity = args.granularity

    ######################
    ## specific kernel? ##
    ######################
    if kernel_id == -1:
        all_kernels = True

    ##################
    ## PTX or SASS? ##
    ##################
    if PTX == True and SASS == True:
        print("\n[Error]\nchoose either PTX or SASS")
        usage()
        sys.exit(1)

    ###############
    ## app name ##
    ###############
    app_name = args.app_name
    
    # app_path = str('apps/')+app_name+str('/')
    app_path = app_name
    sys.path.append(app_path)
    
    if not os.path.exists(app_path):
        print(str("\n[Error]\n<<")+str(app_name)+str(">> doesn't exists in apps directory"))
        sys.exit(1)


    #####################################
    ## target hardware configiguration ##
    #####################################
    gpu_config_file = args.config

    try:
        gpu_configs = importlib.import_module("hardware."+gpu_config_file)
    except:
        print(str("\n[Error]\n")+str("GPU hardware config file provided doesn't exist\n"))
        sys.exit(1)

    
    ##############################
    ## Target ISA Latencies ##
    ##############################
    try:
        ISA = importlib.import_module("hardware.ISA."+gpu_configs.uarch["gpu_arch"])
    except:
        print("\n[Error]\nISA for <<"+gpu_configs.uarch["gpu_arch"]+">> doesn't exists in hardware/ISA directory")
        sys.exit(1)
    ptx_isa = ISA.ptx_isa
    units_latency = ISA.units_latency
    initial_interval = ISA.initial_interval
    sass_isa = ISA.sass_isa

    # 将 ISA 下的添加到 gpu_configs 字典中
    gpu_configs.uarch["ptx_isa"] = ptx_isa
    gpu_configs.uarch["sass_isa"] = sass_isa
    gpu_configs.uarch["units_latency"] = units_latency
    gpu_configs.uarch["initial_interval"] = initial_interval

    try:
        compute_capability = importlib.import_module("hardware.compute_capability."+str(gpu_configs.uarch["compute_capabilty"]))
    except:
        print("\n[Error]\ncompute capabilty for <<"+gpu_configs.uarch["compute_capabilty"]+">> doesn't exists in hardware/compute_capabilty directory")
        sys.exit(1)


    ############################
    ## simulation granularity ##
    ############################
    granularity = args.granularity

    # read cache reference data
    app_res_ref = None
    app_name_ = app_path.split("/")[-2]
    app_arg_ = app_path.split("/")[-1]
    app_and_arg = f"{app_name_}/{app_arg_}"
    if args.hw_res:
        with open(args.hw_res) as fp:
            res_ref = json.load(fp)
        app_res_ref = res_ref[app_and_arg]

    app_report_dir = app_path
    if args.report_output_dir:
        app_report_dir = os.path.join(args.report_output_dir, app_name_, app_arg_)
        if rank==0:
            if not os.path.exists(app_report_dir):
                os.makedirs(app_report_dir)
    
    ##############################
    ## app configiguration file ##
    ##############################
    try:
        import app_config
    except:
        print(str("\n[Error]\n")+str("<app_config.py>> file doesn't exist in \"")+app_name+str("\" directory"))
        sys.exit(1)

    app_kernels_id = app_config.app_kernels_id

    kernels_info = []
    instructions_type = "SASS" if SASS else "PTX"
    if all_kernels == True:
        for kernel_id in app_kernels_id:
            kernels_info.append(get_current_kernel_info(str(kernel_id), app_name, app_path, app_config, instructions_type, granularity, app_res_ref=app_res_ref, app_report_dir=app_report_dir))
    else:
        try:
            kernel_id
        except NameError:
            print("\n[Error]\nmissing target kernel id")
            usage()
            sys.exit(1)
        kernels_info.append(get_current_kernel_info(kernel_id, app_name, app_path, app_config, instructions_type, granularity, app_res_ref=app_res_ref, app_report_dir=app_report_dir))
    

    # ############################
    # # Simian Engine parameters #
    # ############################
    # simianEngine = Simian("PPT-GPU", useMPI=True, opt=False, appPath = app_report_dir, ISA=instructions_type, granularity=granularity, mpiLibName=args.libmpich_path)
    gpuNode = GPUNode(gpu_configs.uarch, compute_capability.cc_configs, len(kernels_info))
        
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
            kernel = Kernel(
                {
                    "name": "Kernel",
                    "out": "/dev/null",
                    "engine": None,
                    "num": i,  # used
                }, gpuNode, kernels_info[i])
            kernel.kernel_call(None, None, None)


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

