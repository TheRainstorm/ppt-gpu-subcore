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
from src.utils import get_current_kernel_info, get_gpu_config, get_app_config
import argparse

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
    parser.add_argument("--kernel", dest="kernel_ids",
                    type=int,
                    nargs='*',
                    default=[],
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
    parser.add_argument('-C', '--overwrite-cache-params',
                        default='',
                        help='l1:capacity:cache_line_size:associativity:sector_size,l2:capacity:cache_line_size:associativity:sector_size')
    parser.add_argument('--memory-model',
                        default='simulator',
                        help='memory model to use')
    parser.add_argument('--AMAT_select', default='')
    parser.add_argument('--scale-opt', default='')
    parser.add_argument('--ipc_select', default='')
    parser.add_argument('--act_cycle_select', default='')
    parser.add_argument("--no-overwrite",
                 action="store_true",
                 help="not overwrite already simulated kernels")
    parser.add_argument('--set-gpu-params', default='',
                        help='key:value,key:value')
    args = parser.parse_args()
    granularity = args.granularity

    # check
    if not os.path.exists(args.app_path):
        print(f"app path {args.app_path} doesn't exist")
        sys.exit(1)
    
    gpu_configs = get_gpu_config(args.config, set_gpu_params=args.set_gpu_params)
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
    comm.Barrier()
    
    kernels_info = []
    instructions_type = "SASS" if args.sass else "PTX"
    if not args.kernel_ids: # sim all kernel
        for kernel_id in app_config['app_kernels_id']:
            kernels_info.append(get_current_kernel_info(int(kernel_id), args.app_path, args.app_path, app_config, instructions_type, args.granularity, app_res_ref=app_res_ref, app_report_dir=app_report_dir))
    else:
        for kernel_id in args.kernel_ids:
            kernels_info.append(get_current_kernel_info(kernel_id, args.app_path, args.app_path, app_config, instructions_type, args.granularity, app_res_ref=app_res_ref, app_report_dir=app_report_dir))

    # filter already simulated kernels
    if args.no_overwrite:
        kernels_info_new = []
        for k in kernels_info:
            kernel_prefix = str(k["kernel_id"])+"_"+instructions_type +"_g"+args.granularity
            k_report_file = os.path.join(app_report_dir, "kernel_"+kernel_prefix+".out")
            if os.path.exists(k_report_file):
                print(f"Skip kernel {k['kernel_id']}, already simulated")
            else:
                kernels_info_new.append(k)
        kernels_info = kernels_info_new
    
    # ############################
    # # Simian Engine parameters #
    # ############################
    # simianEngine = Simian("PPT-GPU", useMPI=True, opt=False, appPath = app_report_dir, ISA=instructions_type, granularity=granularity, mpiLibName=args.libmpich_path)
    kernels = [k['kernel_id'] for k in kernels_info]
    gpuNode = GPUNode(gpu_configs.uarch, gpu_configs.uarch['cc_configs'], kernels)
        
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
            print(f"Kernel {kernels_info[i]['kernel_id']} is running on rank {rank}")
            kernel = Kernel(gpuNode, kernels_info[i])
            kernel.kernel_call(memory_model=args.memory_model, overwrite_cache_params=args.overwrite_cache_params,
                                AMAT_select=args.AMAT_select, scale_opt=args.scale_opt, ipc_select=args.ipc_select,
                                act_cycle_select=args.act_cycle_select)


class GPUNode(object):
	"""
	Class that represents a node that has a GPU
	"""
	def __init__(self, gpu_configs, gpu_configs_cc, kernels):
		self.num_accelerators = 1 # modeling a node that has 1 GPU for now
		self.accelerators = []
		self.gpu_configs = gpu_configs
		self.gpu_configs_cc = gpu_configs_cc
		#print("GPU node generated")
		self.generate_target_accelerators(kernels)

	#generate GPU accelerators inside the node
	def generate_target_accelerators(self, kernels):
		accelerators = importlib.import_module("src.accelerators")
		for i in range(self.num_accelerators):
			self.accelerators.append(accelerators.Accelerator(self, i, self.gpu_configs, self.gpu_configs_cc, kernels))

if __name__ == "__main__":
	main()

