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

import random
import sys
from .warps import Warp

class Scheduler(object):
    '''
    class that represents a warp scheduler inside an SM
    '''
    def __init__(self, num_warp_schedulers, policy):
       self.num_warp_schedulers = num_warp_schedulers
       self.policy = policy

    def step(self, warp_list, cycles, subcore_id):
        '''
        advance computation for the active warps by one cycle,
        choose which step function to execute depending on the scheduling policy
        '''
        if self.policy == "LRR":
            return self.step_LRR(warp_list, cycles, subcore_id)
        elif self.policy == "GTO":
            return self.step_GTO(warp_list, cycles)
        elif self.policy == "TL":
            return self.step_TL(warp_list, cycles)
        

    def step_LRR(self, warp_list, cycles, subcore_id):
        '''
        loop over every available warp and issue warp if ready, 
        if warp is not ready, skip and issue next ready warp
        '''
        warps_executed = 0
        insts_executed = 0
        if len(warp_list) == 0:
            return 0, 0, 'Idle', 'Idle'
        
        scheduler_stall_type_list = ['NoStall','CompData','CompStruct','MemData','MemStruct','Sync','Idle']
        warp_stall_type_list = ['NoStall', 'NotSelect', 'CompData','CompStruct','MemData','MemStruct','Sync','Misc', 'Idle']
        
        # sample active warp
        warp_sampled_idx = random.randint(0, len(warp_list)-1)
        warp_sampled = warp_list[warp_sampled_idx]
        
        warp_stall_type_list = []
        warp_executed_idx = -1
        # head = warp_list.dummy
        # curr_node = head.next
        # while curr_node != head:
        #     warp = curr_node.data
        for i,warp in enumerate(warp_list):
            # see if we can execute warp
            if warp.is_active():
                current_inst_executed, stall_type = warp.step(cycles, subcore_id)

                warp_stall_type_list.append(stall_type)
                # warp executed current instruction successfully
                if current_inst_executed:
                    insts_executed += current_inst_executed
                    warps_executed += 1
                    warp_executed_idx = i
                    break  # only one warp can be executed per cycle
            else:
                print("Error: schedule warp that is not active", file=sys.stderr)
                print(i, len(warp_list), warp.current_inst, len(warp.tasklist), "subcore_id", subcore_id, file=sys.stderr)
                exit(-1)
        if warps_executed == 1:
            # pop warp not active
            if not warp_list[warp_executed_idx].is_active():
                warp_list.pop(warp_executed_idx)   # use linked list to improve performance
        
        # scheduler stall type
        if warps_executed == 0:
            if len(warp_list)==0:
                scheduler_stall_type = 'Idle'
            elif 'MemStruct' in warp_stall_type_list:
                scheduler_stall_type = 'MemStruct'
            elif 'MemData' in warp_stall_type_list:
                scheduler_stall_type = 'MemData'
            elif 'Sync' in warp_stall_type_list:
                scheduler_stall_type = 'Sync'
            elif 'CompStruct' in warp_stall_type_list:
                scheduler_stall_type = 'CompStruct'
            elif 'CompData' in warp_stall_type_list:
                scheduler_stall_type = 'CompData'
            else:
                scheduler_stall_type = 'Idle-Else'
        else:
            scheduler_stall_type = 'NoStall'
        
        # sample warp state
        if warp_sampled_idx == warp_executed_idx:
            warp_state = 'NoStall'
        else:
            if warp_sampled.stall_type_keeped == 'NoStall':
                # or warp_sampled.stalled_cycles <= cycles:  # 简化考虑了，实际上还需要考虑是否真的能够发射，比如 ALU 单元是否空闲
                warp_state = 'NotSelect'
            else:
                warp_state = warp_sampled.stall_type_keeped
        
        return insts_executed, warps_executed, scheduler_stall_type, warp_state