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


class Warp(object):
    '''
    class that represents a warp inside a block being executed on an SM
    '''
    def __init__(self, block, gpu, tasklist, kernel_id, avg_mem_lat, avg_atom_lat):
        
        self.gpu = gpu
        self.active = True # True if warp still has computations to run
        self.stalled_cycles = 0 # number of cycles before the warp can issue a new instruction
        self.block = block
        self.current_inst = 0 # current instruction in the tasklist
        self.tasklist = tasklist
        self.completions = []
        self.syncing = False
        self.max_dep = 0
        self.average_memory_latency = avg_mem_lat
        self.average_atom_latency = avg_atom_lat
        self.kernel_id = kernel_id
        self.divergeStalls = 0
        self.stall_type_keeped = 'NoStall'

    def is_active(self):
        return self.active


    def step(self, cycles):
        '''
        advance computations on the current warp by one clock cycle
        '''	 
        inst_issued = 0 #insts that can be issued at the same time
        
        for i in range(self.gpu.num_inst_dispatch_units_per_SM):
            is_issued, stall_type = self.process_inst(cycles)
            if is_issued:
                inst_issued += 1

        stall_type_first = stall_type if inst_issued == 0 else 'NoStall'
        return inst_issued, stall_type_first  # return first inst stall type

    def process_inst(self, cycles):
        '''
        process each instruction in the tasklist
        '''
        is_issued = False
        latency = 0
        
        inst_stall_type_list = [
            'Sync',   # control / synchronization
            'MemData',
            'MemStruct',
            'CompData',
            'CompStruct',
            'NoStall',
        ]
        
        if self.stalled_cycles > cycles:  # still stalled
            pass
        else:
            if self.syncing:
                pass
            elif self.current_inst > len(self.tasklist):  # should not happend?
                self.active = False
            else:
                inst = self.tasklist[self.current_inst]
                max_dep_idx = -1
                max_dep = float("-inf")
                for i in inst[2:]:
                    if i >= len(self.completions):
                        print("[ERROR]\n with instruction: ", inst, " dependency", i)
                    if self.completions[i] > max_dep:
                        max_dep_idx = i
                        max_dep = self.completions[i]

                # current instruction depends on a is_issued not yet computed
                if cycles < max_dep:
                    dep_inst = self.tasklist[max_dep_idx]
                    if 'MEM_ACCESS' in dep_inst[0]:
                        # self.stall_type_keeped = f'MemData.{dep_inst[0].split("_")[0]}'
                        self.stall_type_keeped = f'MemData'
                    else:
                        self.stall_type_keeped = 'CompData'
                    self.stalled_cycles = max_dep   # stall until the dependency is resolved
                # current instruction is safe to execute
                else:
                    self.stall_type_keeped = 'NoStall'
                    if inst[0] == 'GLOB_MEM_ACCESS':
                        # if inst[1] == 'LD':
                        #     latency = self.average_memory_latency
                        # elif inst[1] == 'ST':
                        #     latency = self.average_memory_latency
                        
                        latency = self.average_memory_latency
                        hw_unit = self.gpu.hw_units[self.kernel_id]['LDS_units'] 
                        is_issued = self.gpu.request_unit(cycles, 4, hw_unit)
                        if not is_issued:
                            self.stall_type_keeped = 'MemStruct'

                    elif inst[0] == 'SHARED_MEM_ACCESS':
                        latency = self.gpu.shared_mem_access_latency
                        is_issued = True

                    elif inst[0] == 'LOCAL_MEM_ACCESS':
                        latency = self.gpu.local_mem_access_latency
                        is_issued = True

                    elif inst[0] == 'CONST_MEM_ACCESS':
                        latency = self.gpu.const_mem_access_latency
                        is_issued = True	

                    elif inst[0] == 'TEX_MEM_ACCESS':
                        latency = self.gpu.tex_mem_access_latency
                        is_issued = True
                    
                    elif inst[0] == 'ATOMIC_OP':
                        latency = self.average_atom_latency
                        self.stalled_cycles = cycles + latency
                        self.stall_type_keeped = 'Sync'
                        is_issued = True

                    elif inst[0] == 'iALU':
                        latency = inst[1]
                        if self.gpu.new_generation:
                            hw_unit = self.gpu.hw_units[self.kernel_id]['INT_units']
                        else:
                            hw_unit = self.gpu.hw_units[self.kernel_id]['SP_units']
                        is_issued =  self.gpu.request_unit(cycles, latency, hw_unit)
                   
                    elif inst[0] == 'fALU':
                        latency = inst[1]
                        hw_unit = self.gpu.hw_units[self.kernel_id]['SP_units']
                        is_issued =  self.gpu.request_unit(cycles, latency, hw_unit)

                    elif inst[0] == 'hALU':
                        latency = inst[1]
                        hw_unit = self.gpu.hw_units[self.kernel_id]['SP_units']
                        is_issued =  self.gpu.request_unit(cycles, latency, hw_unit)

                    elif inst[0] == 'dALU':
                        latency = inst[1]
                        hw_unit = self.gpu.hw_units[self.kernel_id]['DP_units']
                        is_issued =  self.gpu.request_unit(cycles, latency, hw_unit)
                    
                    elif inst[0] == 'SFU':
                        latency = inst[1]
                        hw_unit = self.gpu.hw_units[self.kernel_id]['SF_units']
                        is_issued =  self.gpu.request_unit(cycles, latency, hw_unit)

                    elif inst[0] == 'iTCU' or inst[0] == 'hTCU' or inst[0] == "bTCU":
                        latency = inst[1]
                        hw_unit = self.gpu.hw_units[self.kernel_id]['TC_units']
                        is_issued =  self.gpu.request_unit(cycles, 2, hw_unit)

                    elif inst[0] == 'BRA':
                        latency = inst[1]
                        if self.gpu.new_generation:
                            hw_unit = self.gpu.hw_units[self.kernel_id]['BRA_units']
                        else:
                            hw_unit = self.gpu.hw_units[self.kernel_id]['SP_units']
                        is_issued =  self.gpu.request_unit(cycles, latency, hw_unit)
                    
                    elif inst[0] == 'MEMBAR':
                        latency = 0
                        self.stalled_cycles = cycles + latency 
                        self.stall_type_keeped = 'Sync'
                        is_issued = True

                    # Synchronize all warps in the same block
                    elif inst[0] == 'BarrierSYNC':
                        # self.block.sync_warps += 1
                        # self.syncing = True
                        if self.gpu.new_generation:
                            hw_unit = self.gpu.hw_units[self.kernel_id]['INT_units']
                        else:
                            hw_unit = self.gpu.hw_units[self.kernel_id]['SP_units']
                        is_issued =  self.gpu.request_unit(cycles, 4, hw_unit)
                        if not is_issued:
                            self.stall_type_keeped = 'Sync'

                    if not is_issued:
                        if self.stall_type_keeped != 'NoStall':
                            self.stall_type_keeped = 'CompStruct'
                    else:
                        self.divergeStalls += 1
                        self.completions.append(cycles + latency)
                        if self.max_dep < self.completions[-1]:  # No means???
                            self.max_dep = self.completions[-1] 
                        self.current_inst += 1
                        if self.current_inst == len(self.tasklist):
                            self.active = False

        return is_issued, self.stall_type_keeped