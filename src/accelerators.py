##############################################################################
# &copy 2017. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

# All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

# Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
##############################################################################

# Author: Yehia Arafa
# Last Update Date: April, 2021
# Copyright: Open source, must acknowledge original author

##############################################################################

from copy import deepcopy


class Accelerator(object):

    # Accelerator is replicated for each kernel
    def __init__(self, node, id, gpu_configs, gpu_configs_cc, kernels):
        
        self.node =	 node # GPUNode
        self.id = id
        self.cc = gpu_configs_cc

        try:
            self.compute_capabilty = gpu_configs["compute_capabilty"]
        except:
            print_config_error("compute_capabilty")
        
        if self.compute_capabilty >= 70:
            self.new_generation = True
        else:
            self.new_generation = False

        try:
            self.GPU_clockspeed = gpu_configs["clockspeed"]
        except:
            print_config_error("clockspeed")

        try:
            self.num_SMs = gpu_configs["num_SMs"]
        except:
            print_config_error("num_SMs")
        
        if self.new_generation:
            try:
                self.num_INT_units_per_SM = gpu_configs["num_INT_units_per_SM"]
            except:
                print_config_error("num_INT_units_per_SM")
        else:
            self.num_INT_units_per_SM = 0
        
        try:
            self.num_SP_units_per_SM = gpu_configs["num_SP_units_per_SM"]
        except:
            print_config_error("num_SP_units_per_SM")

        try:
            self.num_DP_units_per_SM = gpu_configs["num_DP_units_per_SM"]
        except:
            print_config_error("num_DP_units_per_SM")

        try:
            self.num_SF_units_per_SM = gpu_configs["num_SF_units_per_SM"] 
        except:
            print_config_error("num_SF_units_per_SM")

        try:
            self.num_TC_units_per_SM = gpu_configs["num_TC_units_per_SM"] 
        except:
            print_config_error("num_TC_units_per_SM")

        try:
            self.num_LDS_units_per_SM = gpu_configs["num_LDS_units_per_SM"]
        except:
            print_config_error("num_LDS_units_per_SM")

        try: 
            self.num_BRA_units_per_SM = gpu_configs["num_BRA_units_per_SM"]
        except:
            print_config_error("num_BRA_units_per_SM")

        try:
            self.num_TEX_units_per_SM = gpu_configs["num_TEX_units_per_SM"]
        except:
            print_config_error("num_TEX_units_per_SM")

        try:
            self.num_warp_schedulers_per_SM = gpu_configs["num_warp_schedulers_per_SM"]
        except:
            print_config_error("num_warp_schedulers_per_SM")

        try:
            self.num_inst_dispatch_units_per_SM = gpu_configs["num_inst_dispatch_units_per_SM"]
        except:
            print_config_error("num_inst_dispatch_units_per_SM")

        try:	
            self.l1_cache_bypassed = gpu_configs["l1_cache_bypassed"]
        except:
            if self.new_generation:
                self.l1_cache_bypassed = False
                print_warning("l1_cache_bypassed","active", flag=True)
            else:
                self.l1_cache_bypassed = True
                print_warning("l1_cache_bypassed","active", flag=True)
        
        if not self.l1_cache_bypassed:
            try:
                self.l1_cache_size = gpu_configs["l1_cache_size"]
            except:
                print_config_error("l1_cache_size")

            try:
                self.l1_cache_line_size = gpu_configs["l1_cache_line_size"]
            except:
                print_config_error("l1_cache_line_size")

            try:
                self.l1_cache_associativity = gpu_configs["l1_cache_associativity"]
            except:
                print_config_error("l1_cache_associativity")

        try:
            self.l2_cache_size = gpu_configs["l2_cache_size"]
        except:
            print_config_error("l2_cache_size")
        
        try:
            self.l2_cache_line_size = gpu_configs["l2_cache_line_size"]
        except:
            print_config_error("l2_cache_line_size")
        
        try:
            self.l2_cache_associativity = gpu_configs["l2_cache_associativity"]
        except:
            print_config_error("l2_cache_associativity")

        try:
            self.shared_mem_size = gpu_configs["shared_mem_size"]
        except:
            print_config_error("shared_mem_size")

        try:
            self.num_l2_partitions = gpu_configs["num_l2_partitions"]
        except:
            print_config_error("num_l2_partitions")

        try:
            self.num_dram_channels = gpu_configs["num_dram_channels"]
        except:
            print_config_error("num_dram_channels")

        try:
            self.dram_bandwidth = gpu_configs["dram_th_bandwidth"]
        except:
            print_config_error("dram_th_bandwidth")
        
        try:
            self.dram_clockspeed = gpu_configs["dram_clockspeed"]
        except:
            print_config_error("dram_clockspeed")

        try:
            self.noc_bandwidth = gpu_configs["noc_th_bandwidth"]
        except:
            print_config_error("noc_th_bandwidth")

        try:
            self.warp_scheduling_policy =  gpu_configs["warp_scheduling"]
        except:
            print_config_error("warp_scheduling")

        try:
            self.warp_size = gpu_configs_cc["warp_size"]
        except:
            print_config_error("warp_size", flag=2)
        
        try:
            self.max_block_size = gpu_configs_cc["max_block_size"]
        except:
            print_config_error("max_block_size", flag=2)
        
        try:
            self.smem_allocation_size = gpu_configs_cc["smem_allocation_size"]
        except:
            print_config_error("smem_allocation_size", flag=2)
        
        try:
            self.max_registers_per_SM = gpu_configs_cc["max_registers_per_SM"]
        except:
            print_config_error("max_registers_per_SM", flag=2)

        try:
            self.max_registers_per_block = gpu_configs_cc["max_registers_per_block"]
        except:
            print_config_error("max_registers_per_block", flag=2)
        
        try:
            self.max_registers_per_thread = gpu_configs_cc["max_registers_per_thread"]
        except:
            print_config_error("max_registers_per_thread", flag=2)

        try:
            self.register_allocation_size = gpu_configs_cc["register_allocation_size"]
        except:
            print_config_error("register_allocation_size", flag=2)
        
        try:
            self.max_active_blocks_per_SM = gpu_configs_cc["max_active_blocks_per_SM"]
        except:
            print_config_error("max_active_blocks_per_SM", flag=2)
        
        try:
            self.max_active_threads_per_SM = gpu_configs_cc["max_active_threads_per_SM"]
        except:
            print_config_error("max_active_threads_per_SM", flag=2)

        self.max_active_warps_per_SM = self.max_active_threads_per_SM / self.warp_size

        self.ptx_isa = gpu_configs["ptx_isa"]
        self.units_latency = gpu_configs["units_latency"]
        self.sass_isa = gpu_configs["sass_isa"]
        self.initial_interval = gpu_configs["initial_interval"]
        self.unit_number = gpu_configs["unit_number"]

        self.l1_cache_access_latency = self.units_latency["l1_cache_access"]
        self.l2_cache_access_latency = self.units_latency["l2_cache_access"]
        self.l2_cache_from_l1_access_latency = self.l2_cache_access_latency - self.l1_cache_access_latency
        self.dram_mem_access_latency = self.units_latency["dram_mem_access"]
        self.dram_mem_from_l2_access_latency = self.dram_mem_access_latency - self.l2_cache_from_l1_access_latency
        self.local_mem_access_latency = self.units_latency["local_mem_access"]
        self.const_mem_access_latency = self.units_latency["const_mem_access"]
        self.tex_mem_access_latency = self.units_latency["tex_mem_access"]
        self.tex_cache_access_latency = self.units_latency["tex_cache_access"]
        self.shared_mem_access_latency = self.units_latency["shared_mem_access"]
        self.atomic_op_access_latency = self.units_latency["atomic_operation"]
        self.TB_launch_overhead = self.units_latency["TB_launch_ovhd"]

        self.tasklist = {}
        
        subcore_hw_units = {}
        for unit in self.initial_interval:
            subcore_hw_units[unit] = [0 for i in range(self.unit_number[unit])]     # 每种 unit 包含一个列表，记录一组（32 个）单元的 当前 stall cycle，初始化为 0
        
        sm_hw_units = [deepcopy(subcore_hw_units) for i in range(self.num_warp_schedulers_per_SM)]

        # copy for each kernel
        # self.hw_units = [deepcopy(sm_hw_units) for i in range(kernels)]
        self.hw_units = {}
        for kernel_id in kernels:
            self.hw_units[kernel_id-1] = deepcopy(sm_hw_units)  # 使用时，使用了 0-based index，因此这里减1
        
    def request_unit(self, kernel_id, subcore_id, cycles, unit):
        '''
        return True if the request hw unit is fulfilled
        '''
        units = self.hw_units[kernel_id][subcore_id][unit]
        for i in range(self.unit_number[unit]):
            if units[i] <= cycles:
                units[i] = cycles + self.initial_interval[unit]
                return True
        return False
