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
import sys, time, importlib
# from simian import Entity, Simian
from .helper_methods import *
from .memory_model import *
from .blocks import Block
from .warp_scheduler import Scheduler
from collections import deque
from .utils import LinkedList,get_max_active_block_per_sm
curr_dir = os.path.dirname(__file__)
par_dir = os.path.dirname(curr_dir)
sys.path.insert(0, os.path.abspath(par_dir))
from memory_model.memory_model_warper import memory_model_warpper_single_kernel, ppt_gpu_model_warpper,sdcm_model_warpper_parallel

class Kernel():

    def __init__(self, gpuNode, kernel_info):
        # super(Kernel, self).__init__(base_info)
        # print("kernel %s, %s inits on Entity %d, Rank %d" % (kernel_info['kernel_name'], self.name, self.num, self.engine.rank))
        # sys.stdout.flush()

        ## There is 1 Acc in Node replicated for each Kernel
        self.gpuNode = gpuNode
        self.acc = self.gpuNode.accelerators[0] 

        # get kernel_info
        self.kernel_info = kernel_info
        self.kernel_name = kernel_info['kernel_name']
        self.kernel_id = int(kernel_info["kernel_id"])
        self.kernel_id_0_base = int(kernel_info["kernel_id"]) - 1
        self.mem_traces_dir_path = kernel_info['mem_traces_dir_path']
        self.kernel_grid_size = kernel_info['grid_size']
        self.kernel_block_size = kernel_info['block_size']
        self.kernel_num_regs = kernel_info['num_regs']
        self.kernel_smem_size = kernel_info['smem_size']
        self.ISA = "PTX" if kernel_info['ISA'] == 1 else "SASS"
        self.ptx_file_path = kernel_info["ptx_file_path"]
        self.sass_file_path = kernel_info["sass_file_path"]
        self.cache_ref_data = kernel_info["cache_ref_data"]
        if kernel_info["granularity"] == "1":
            self.simulation_granularity = "OTB" #One Thread Block
        elif kernel_info["granularity"] == "2":
            self.simulation_granularity = "AcTB" #Active Thread Block
        elif kernel_info["granularity"] == "3":
            self.simulation_granularity = "AlTB" #All Thread Block

        # self.simulation_granularity = "TBSW" if kernel_info["granularity"] == "1" else "TBS"

        ## kernel local predictions outputs
        self.pred_out = {}
        pred_out = self.pred_out

        ## init predictions outputs
        pred_out["app_report_dir"] = kernel_info['app_report_dir']
        pred_out["kernel_id"] = self.kernel_id 
        pred_out["kernel_name"] = self.kernel_name
        pred_out["ISA"] = self.ISA
        pred_out["granularity"] = kernel_info["granularity"]
        pred_out["grid_size"] = 0
        pred_out["active_SMs"] = 0
        pred_out["max_active_blocks_per_SM"] = self.acc.max_active_blocks_per_SM
        pred_out["block_per_sm_limit_warps"] = 0.0
        pred_out["block_per_sm_limit_regs"] = 0.0
        pred_out["block_per_sm_limit_smem"] = 0.0
        pred_out["th_max_active_block_per_sm"] = 0
        pred_out["th_active_warps"] = 0
        pred_out["th_active_threads"] = 0
        pred_out["th_occupancy"] = 0.0
        pred_out["allocated_active_warps_per_block"] = 0
        pred_out["achieved_active_warps"] = 0.0
        pred_out["achieved_occupancy"] = 0.0
        pred_out["allocted_block_per_sm"] = 0   # block 数除以 SM 数
        pred_out["max_active_block_per_sm"] = 0 # 首先计算了理论 SM 最大 block 数目（和 reg、shared、block size）。然后该变量还和 allocted_block_per_sm 取了个最小值
        pred_out["block_per_sm_simulate"] = 0    # 根据模拟粒度，决定最终模拟的 SM 中 block 数目。AcTB 时赋值为 max_active_block_per_sm
        pred_out["active_cycles"] = 0
        pred_out["warps_instructions_executed"] = 0
        pred_out["threads_instructions_executed"] = 0
        pred_out["ipc"] = 0
        pred_out["l1_cache_bypassed"] = self.acc.l1_cache_bypassed
        pred_out["comp_cycles"] = 0
        pred_out["gpu_act_cycles"] = 0
        pred_out["gpu_elp_cycles"] = 0
        pred_out["sm_act_cycles.sum"] = 0
        pred_out["sm_elp_cycles.sum"] = 0
        pred_out["last_inst_delay"] = 0
        pred_out["tot_warps_instructions_executed"] = 0
        pred_out["tot_ipc"] = 0
        pred_out["tot_cpi"] = 0.0
        pred_out["tot_throughput_ips"] = 0.0
        pred_out["execution_time_sec"] = 0.0
        pred_out["AMAT"] = 0
        pred_out["ACPAO"] = 0
        pred_out["memory_stats"] = {}
        pred_out["simulation_time"] = {}
        
        # launch statistic
        pred_out["grid_size"] = kernel_info["grid_size"]
        pred_out["block_size"] = kernel_info["block_size"]
        pred_out["num_regs"] = kernel_info["num_regs"]
        pred_out["smem_size"] = kernel_info["smem_size"]

        if self.kernel_block_size > self.acc.max_block_size:
            print_warning("block_size",str(self.acc.max_block_size))
            self.kernel_block_size = self.acc.max_block_size

        if self.kernel_num_regs > self.acc.max_registers_per_thread:
            print_warning("num_registers",str(self.acc.max_registers_per_thread))
            self.kernel_num_regs = self.acc.max_registers_per_thread

        if self.kernel_smem_size > self.acc.shared_mem_size:
            print_warning("shared_mem_bytes",str(self.acc.shared_mem_size))
            self.kernel_smem_size = self.acc.shared_mem_size

        occupancy_res = get_max_active_block_per_sm(self.acc.cc, kernel_info, self.acc.num_SMs, self.acc.shared_mem_size)
        
        pred_out["active_SMs"] = min(self.acc.num_SMs, pred_out["grid_size"])
        pred_out.update(occupancy_res)
        
        ## allocate the block_per_sm_simulate according to the simulation granularity
        if self.simulation_granularity == "OTB":
            pred_out["max_active_block_per_sm"] = 1
            pred_out["block_per_sm_simulate"] = 1
        elif self.simulation_granularity == "AcTB":
            pred_out["block_per_sm_simulate"] = pred_out["max_active_block_per_sm"] 
        elif self.simulation_granularity == "AlTB":
            pred_out["block_per_sm_simulate"] = pred_out["allocted_block_per_sm"]

        ## initilaizing kernel's warp scehdulers 
        self.warp_scheduler = Scheduler(self.acc.num_warp_schedulers_per_SM, self.acc.warp_scheduling_policy)



    def kernel_call(self, overwrite_cache_params=None, memory_model='simulator', 
                    AMAT_select='', scale_opt='', act_cycle_select='', ipc_select='', no_adaptive_cache=False, no_KLL=False):
        '''
        AMAT_select: AMAT_ori, AMAT_sum, AMAT_foumula, const_1000
        scale_opt: ori, float, ceil
        ipc_select: tot_ipc, sm_ipc, smsp_ipc(用不了)
        act_cycle_select: 
        '''
        print(f"AMAT_select={AMAT_select}, scale_opt={scale_opt}, act_cycle_select={act_cycle_select}, ipc_select={ipc_select}")
        # 设置 memory model 需要的变量
        gpu_config = self.gpuNode.gpu_configs
        kernel_param = self.kernel_info  # kernel info 信息更多
        if overwrite_cache_params:
            L = ['', 'cache_size', 'cache_line_size', 'cache_associativity', 'sector_size']
            for cache_params in overwrite_cache_params.split(','):
                for i,p in enumerate(cache_params.split(':')):
                    if i==0:
                        cur = p  # current cache level
                        continue
                    # print(cur, L[i], p)
                    if p:
                        gpu_config[f'{cur}_{L[i]}'] = int(p)
                        print(f"Info: overwrite {cur} {L[i]} to {p}")
        l1_cache_size_old = gpu_config['l1_cache_size']
        if gpu_config['adaptive_cache'] and not no_adaptive_cache:
            occupancy_res = get_max_active_block_per_sm(gpu_config['cc_configs'], kernel_param, gpu_config['num_SMs'], gpu_config['shared_mem_size'],
                                                        shared_mem_carveout=gpu_config['shared_mem_carveout'], adaptive=gpu_config['adaptive_cache'])
            gpu_config['l1_cache_size'] = gpu_config['shared_mem_size'] - occupancy_res['adaptive_smem_size']
            if gpu_config['l1_cache_size'] != l1_cache_size_old:
                print(f"Info: set adaptive L1 cache size from {l1_cache_size_old} to {gpu_config['l1_cache_size']}")
        else:
            print("Disable adaptive cache")
            occupancy_res = get_max_active_block_per_sm(gpu_config['cc_configs'], kernel_param, gpu_config['num_SMs'], gpu_config['shared_mem_size'],
                                                        adaptive=False)
        
                   
        pred_out = self.pred_out

        if self.ISA == "PTX":
            ptx_parser = importlib.import_module("ISA_parser.ptx_parser")
            self.kernel_tasklist, gmem_reqs = ptx_parser.parse(units_latency = self.acc.units_latency, ptx_instructions = self.acc.ptx_isa,\
                                                                ptx_path = self.ptx_file_path, num_warps = pred_out["allocated_active_warps_per_block"])

        elif self.ISA == "SASS":
            sass_parser = importlib.import_module("ISA_parser.sass_parser")
            self.kernel_tasklist, gmem_reqs = sass_parser.parse(units_latency = self.acc.units_latency, sass_instructions = self.acc.sass_isa,\
                                                                sass_path = self.sass_file_path, num_warps = pred_out["allocated_active_warps_per_block"])
                                                    
        ###### ---- memory performance predictions ---- ######
        tic = time.time()
        # pred_out["memory_stats"] = ppt_gpu_model_warpper(kernel_param['kernel_id'], self.kernel_info['trace_dir'], kernel_param, occupancy_res['max_active_block_per_sm'], gpu_config,
        #                                 granularity=int(self.kernel_info['granularity']))
        # pred_out["memory_stats"] = get_memory_perf(pred_out["kernel_id"], self.mem_traces_dir_path, pred_out["grid_size"], self.acc.num_SMs,\
        #                                             self.acc.l1_cache_size, self.acc.l1_cache_line_size, self.acc.l1_cache_associativity,\
        #                                             self.acc.l2_cache_size, self.acc.l2_cache_line_size, self.acc.l2_cache_associativity,\
        #                                             gmem_reqs, int(pred_out["allocted_block_per_sm"]), int(pred_out["block_per_sm_simulate"]), cache_ref_data=self.cache_ref_data)
        pred_out["memory_stats"] = memory_model_warpper_single_kernel(gpu_config, kernel_param, occupancy_res, self.kernel_info['trace_dir'],
                                  model=memory_model, granularity=int(self.kernel_info['granularity']))
        # with open('obj.pkl', 'rb') as f:
        #     pred_out["memory_stats"] = pickle.load(f)
        toc = time.time()
        pred_out["simulation_time"]["memory"] = (toc - tic)

        pred_out["others"] = {}
        # AMAT: Average Memory Access Time (Cycles)
        if pred_out["memory_stats"]["gmem_tot_reqs"] != 0:
            highly_divergent_degree = 17
            l2_parallelism = 1
            dram_parallelism = 1
            pred_out["others"]["diverge_flag"] = 0
            if pred_out["memory_stats"]["gmem_ld_diverg"] >= self.acc.num_dram_channels\
            or pred_out["memory_stats"]["gmem_st_diverg"] >= self.acc.num_dram_channels\
            or pred_out["memory_stats"]["gmem_tot_diverg"] >= highly_divergent_degree:
                pred_out["others"]["diverge_flag"] = 1
                l2_parallelism = min(pred_out["memory_stats"]["gmem_tot_diverg"], self.acc.num_l2_partitions)
                dram_parallelism = min(pred_out["memory_stats"]["gmem_tot_diverg"], self.acc.num_dram_channels)
                # l2_parallelism = self.num_dram_channels
                # dram_parallelism = self.num_dram_channels
                # l2_parallelism = self.num_l2_partitions
                # l2_parallelism = pred_out["memory_stats"]["gmem_tot_diverg"]
                # dram_parallelism = pred_out["memory_stats"]["gmem_tot_diverg"]
            pred_out["others"]["l2_parallelism"] = l2_parallelism
            pred_out["others"]["dram_parallelism"] = dram_parallelism

            l1_cycles_no_contention = (pred_out["memory_stats"]["gmem_tot_sectors_per_sm"]) * self.acc.l1_cache_access_latency
            l2_cycles_no_contention = pred_out["memory_stats"]["l2_tot_trans"] * self.acc.l2_cache_from_l1_access_latency * (1/l2_parallelism)
            dram_cycles_no_contention = pred_out["memory_stats"]["dram_tot_trans"] * self.acc.dram_mem_from_l2_access_latency * (1/dram_parallelism)
            pred_out["others"]["l1_cycles_no_contention"] = l1_cycles_no_contention
            pred_out["others"]["l2_cycles_no_contention"] = l2_cycles_no_contention
            pred_out["others"]["dram_cycles_no_contention"] = dram_cycles_no_contention
            
            mem_cycles_no_contention_sum = l1_cycles_no_contention + l2_cycles_no_contention + dram_cycles_no_contention
            mem_cycles_no_contention = max(l1_cycles_no_contention, l2_cycles_no_contention) 
            mem_cycles_no_contention = max(mem_cycles_no_contention, dram_cycles_no_contention)
            mem_cycles_no_contention = ceil(mem_cycles_no_contention, 1)
            
            dram_service_latency = self.acc.dram_clockspeed * (self.acc.l2_cache_line_size / self.acc.dram_bandwidth)
            dram_queuing_delay_cycles = pred_out["memory_stats"]["dram_tot_trans"] * dram_service_latency * (1/dram_parallelism)
            pred_out["others"]["dram_service_latency"] = dram_service_latency
            pred_out["others"]["dram_queuing_delay_cycles"] = dram_queuing_delay_cycles

            mem_cycles_ovhds = dram_queuing_delay_cycles
        
            noc_service_latency = self.acc.dram_clockspeed * (self.acc.l1_cache_line_size / self.acc.noc_bandwidth)
            noc_queueing_delay_cycles = pred_out["memory_stats"]["l2_tot_trans"] * noc_service_latency * (1/l2_parallelism)
            pred_out["others"]["noc_service_latency"] = noc_service_latency
            pred_out["others"]["noc_queueing_delay_cycles"] = noc_queueing_delay_cycles

            if noc_queueing_delay_cycles > (self.acc.l2_cache_from_l1_access_latency + self.acc.dram_mem_from_l2_access_latency):
                mem_cycles_ovhds += noc_queueing_delay_cycles

            mem_cycles_ovhds = ceil(mem_cycles_ovhds, 1)
            pred_out["others"]["mem_cycles_no_contention"] = mem_cycles_no_contention
            pred_out["others"]["mem_cycles_no_contention_sum"] = mem_cycles_no_contention_sum
            pred_out["others"]["mem_cycles_ovhds"] = mem_cycles_ovhds

            tot_mem_cycles = ceil((mem_cycles_no_contention + mem_cycles_ovhds), 1)
            tot_mem_cycles_sum = ceil((mem_cycles_no_contention_sum + mem_cycles_ovhds), 1)
            
            pred_out["AMAT_ori"] = ceil(tot_mem_cycles/pred_out["memory_stats"]["gmem_tot_reqs"], 1)
            pred_out["AMAT_sum"] = ceil(tot_mem_cycles_sum/pred_out["memory_stats"]["gmem_tot_reqs"], 1)

            m1 = 1 - pred_out["memory_stats"]['l1_hit_rate']
            m2 = 1 - pred_out["memory_stats"]['l2_hit_rate']
            # m2 = 1 - 0.8
            pred_out["AMAT_foumula"] = ceil(self.acc.l1_cache_access_latency + m1*(self.acc.l2_cache_from_l1_access_latency + m2*self.acc.dram_mem_from_l2_access_latency), 1)
            
        # AMAT 选择
        if AMAT_select.startswith('const'): # const_1000
            amat = int(AMAT_select.split('_')[1])
        elif AMAT_select=='':
            amat = pred_out["AMAT_ori"]
        else:
            amat = pred_out[AMAT_select]
        pred_out["AMAT"] = amat
        
        # ACPAO: Average Cycles Per Atomic Operation
        # ACPAO = atomic operations latency / total atomic requests
        # atomic operations latency= (atomic & redcutions transactions * access latency of atomic & red requests)
        if pred_out["memory_stats"]["atom_red_tot_trans"] != 0:
            pred_out["ACPAO"] = (self.acc.atomic_op_access_latency * pred_out["memory_stats"]["atom_red_tot_trans"])\
                            /(pred_out["memory_stats"]["atom_tot_reqs"] + pred_out["memory_stats"]["red_tot_reqs"])



        ###### ---- compute performance predictions ---- ######
        tic = time.time()
        block_list = self.spawn_blocks(self.acc, pred_out["block_per_sm_simulate"], pred_out["allocated_active_warps_per_block"],\
                                        self.kernel_tasklist, self.kernel_id_0_base, self.ISA, pred_out["AMAT"], pred_out["ACPAO"])

        
        ## before we do anything we need to activate Blocks up to active blocks
        for i in range(pred_out["max_active_block_per_sm"]):
            block_list[i].active = True
            block_list[i].waiting_to_execute = False  # default: active: false, wait: true

        pred_out["comp_cycles"] = self.acc.TB_launch_overhead

        num_subcore = self.acc.num_warp_schedulers_per_SM
        pred_out['num_subcore'] = num_subcore
        max_warp_per_subcore = self.acc.max_active_threads_per_SM // self.acc.warp_size // num_subcore
        sm_stats = {}
        sm_stats['active_blocks'] = {}
        scheduler_stats = {}
        scheduler_stats['active_warps'] = [{} for i in range(num_subcore)]
        scheduler_stats['eligible_warps'] = [{} for i in range(num_subcore)]
        scheduler_stats['issued_warps'] = [{} for i in range(num_subcore)]
        scheduler_stats['stall_types'] = [{} for i in range(num_subcore)]
        warp_stats = {}
        warp_stats['stall_types'] = [{} for i in range(num_subcore)]
        
        pred_out['scheduler_stats'] = scheduler_stats
        pred_out['warp_stats'] = warp_stats
        pred_out['sm_stats'] = sm_stats
        
        def counter_inc(counter, key):
            if key in counter:
                counter[key] += 1
            else:
                counter[key] = 1
        new_active_warp_list = deque()  # when schedule new block to SM, add its warp to new list
        block_is_visited = set()
        subcore_warp_list = [[] for i in range(num_subcore)]  # schedule warps in new list to subcore
                                                              # each subcore warp scheduler maintains a pool of warps
        subcore_completed = [0 for i in range(num_subcore)]  # record the cycle when subcore idle
        subcore_warp_executed = [0 for i in range(num_subcore)]
        subcore_instr_executed = [0 for i in range(num_subcore)]
        
        # subcore_warp_list = [LinkedList() for i in range(num_subcore)]
        ## process instructions of the tasklist by the active blocks every cycle
        while self.blockList_has_active_warps(block_list):
            if pred_out['active_cycles']%10000 == 0:
                print(f"kernel {self.kernel_id}: active cycles {pred_out['active_cycles']}")
            
            ## compute the list warps in active blocks
            current_active_block_list = []
            current_active_blocks = 0

            for block in block_list:
                if current_active_blocks >= pred_out["max_active_block_per_sm"]:
                    break
                
                ## all warps inside this block finished execution
                if not block.is_active() and not block.is_waiting_to_execute():
                    continue
                
                ## block is ready to execute
                if not block.is_active() and block.is_waiting_to_execute():
                    block.active = True
                    block.waiting_to_execute = False
                    ## add latency of scheduling a new TB 
                    pred_out["comp_cycles"] += self.acc.TB_launch_overhead / pred_out["max_active_block_per_sm"]
                
                ## this block still has warps executing; add its warps to the warp list
                if block.is_active() and not block.is_waiting_to_execute():
                    current_active_block_list.append(block)
                    # new block is visited, add to the new warp list
                    if block not in block_is_visited:
                        block_is_visited.add(block)
                        for warp in block.warp_list:
                            if warp.is_active():
                                new_active_warp_list.append(warp)
                    current_active_blocks += 1
            '''
            fetch warp to subcore warp list
            '''
            # allocate chunk (maybe imbalance?)
            # for i in range(num_subcore):
            #     while len(new_active_warp_list) and len(subcore_warp_list[i]) < max_warp_per_subcore:
            #         warp = new_active_warp_list.popleft()
            #         subcore_warp_list[i].append(warp)
            # # balance
            # reamain_slot = [num_subcore - len(subcore_warp_list[i]) for i in range(num_subcore)]
            # all_remain = sum(reamain_slot)
            # allocate_weight = [len(reamain_slot[i]) / all_remain for i in range(num_subcore)]
            # At present, new warp is added only when a block is swap in, so we can simply distribute
            # new added warp to subcore evenly.
            for i, warp in enumerate(new_active_warp_list):
                subcore_warp_list[i % num_subcore].append(warp)
            # check not exceed max_warp_per_subcore
            if any([len(subcore_warp_list[i]) > max_warp_per_subcore for i in range(num_subcore)]):
                print("Error: exceed max warp per subcore", file=sys.stderr)
                exit(1)
            # empty new warp list
            new_active_warp_list = deque()
            
            '''
            subcore scheduler issue warps
            '''
            counter_inc(sm_stats['active_blocks'], len(current_active_block_list))
            # print(debug_i)
            for i in range(num_subcore):
                is_empty = len(subcore_warp_list[i]) == 0
                counter_inc(scheduler_stats['active_warps'][i], len(subcore_warp_list[i]))
                ## pass warps belonging to the active blocks to the warp scheduler to step the computations
                instructions_executed, warp_executed, scheduler_stall_type, warp_state_sampled = \
                    self.warp_scheduler.step(subcore_warp_list[i], pred_out["active_cycles"], i)
                
                counter_inc(warp_stats['stall_types'][i], warp_state_sampled)
                
                counter_inc(scheduler_stats['issued_warps'][i], warp_executed)
                counter_inc(scheduler_stats['stall_types'][i], scheduler_stall_type)
                pred_out["warps_instructions_executed"] += instructions_executed
                subcore_warp_executed[i] += warp_executed
                subcore_instr_executed[i] += instructions_executed
                
                # subcore becomes idle
                if not is_empty and len(subcore_warp_list[i])==0:
                    subcore_completed[i] = pred_out["active_cycles"]

            for block in current_active_block_list:
                pred_out["achieved_active_warps"] += block.count_active_warps()

            ## next cycles
            pred_out["active_cycles"] += 1

        pred_out["achieved_active_warps"] = pred_out["achieved_active_warps"] / pred_out["active_cycles"]
        pred_out["achieved_occupancy"]= (float(pred_out["achieved_active_warps"]) / float(self.acc.max_active_warps_per_SM)) * 100
        
        # 几种不同计算 scale 的方式
        scale1 = pred_out["allocted_block_per_sm"] / pred_out["block_per_sm_simulate"]
        scale2 = ceil(pred_out["allocted_block_per_sm"] / pred_out["block_per_sm_simulate"], 1)
        num_workloads_left = pred_out["allocted_block_per_sm"] - pred_out["block_per_sm_simulate"]
        remaining_cycles = ceil((num_workloads_left/pred_out["block_per_sm_simulate"]),1)
        scale_ori = max(1, remaining_cycles)
        if scale_opt=='ori' or scale_opt=='':
            scale = scale_ori
        elif scale_opt=='float':
            scale = scale1
        elif scale_opt=='ceil':
            scale = scale2
        else:
            print('Error: not support scale_opt', file=sys.stderr)
            exit(1)
        
        # 计算 active_cycle_scale
        kernel_detail = {}
        # PPT-GPU 原本计算方法
        #TODO: has to be done in a more logical way per TB
        last_inst_delay = 0
        for block in block_list:
            last_inst_delay_act_min = max(last_inst_delay, block.actual_end - pred_out["active_cycles"])
            last_inst_delay_act_max = max(last_inst_delay, block.actual_end)
        kernel_detail["last_inst_delay_act_min"], kernel_detail["last_inst_delay_act_max"] = last_inst_delay_act_min, last_inst_delay_act_max
        
        act_cycles_min = pred_out["active_cycles"] + pred_out["comp_cycles"] + last_inst_delay_act_min
        act_cycles_max = pred_out["active_cycles"] + pred_out["comp_cycles"] + last_inst_delay_act_max
        kernel_detail["act_cycles_min"], kernel_detail["act_cycles_max"] = act_cycles_min, act_cycles_max

        pred_out["PPT-GPU_min"] = pred_out["gpu_act_cycles_min"] = act_cycles_min * scale_ori
        pred_out["PPT-GPU_max"] = pred_out["gpu_act_cycles_max"] = act_cycles_max * scale_ori

        block_actual_end = max([block.actual_end for block in block_list])
        
        # 修正后的计算方法，按照 subcore 的最大完成时间来计算
        subcore_completed_nzero = [e for e in subcore_completed if e != 0]
        smsp_act_cycles_min = min(subcore_completed_nzero)
        smsp_act_cycles_max = max(subcore_completed_nzero)
        smsp_act_cycles_avg = sum(subcore_completed_nzero) / len(subcore_completed_nzero)
        last_inst = block_actual_end - smsp_act_cycles_max
        tail = smsp_act_cycles_max - smsp_act_cycles_avg
        
        # 考虑负载不均衡（avg，tail） + 最后一条指令
        result = {}
        result["ours_base"] = pred_out["active_cycles"] * scale
        result["ours_BL"] = (pred_out["active_cycles"] + pred_out['comp_cycles']) * scale
        result["ours_smsp_min"] = smsp_act_cycles_min * scale
        result["ours_smsp_max"] = smsp_act_cycles_max * scale
        result["ours_smsp_avg"] = smsp_act_cycles_avg * scale
        result["ours_smsp_avg_LI"] = smsp_act_cycles_avg * scale + last_inst
        result["ours_smsp_avg_tail"] = smsp_act_cycles_avg * scale + tail
        result["ours_smsp_avg_tail_LI"] = smsp_act_cycles_avg * scale + tail + last_inst
        result["ours_smsp_avg_scale2"] = smsp_act_cycles_avg * scale2
        result["ours_smsp_avg_tail_scale2"] = smsp_act_cycles_avg * scale2 + tail
        result["ours_smsp_avg_tail_scale2_LI"] = smsp_act_cycles_avg * scale2 + tail + last_inst
        pred_out["active_cycle_scale"] = result["ours_smsp_avg_LI" if act_cycle_select=='' else act_cycle_select]
        pred_out['result'] = result
        
        # 计算 Kernel launch latency
        # kernel_lat = 2.16*(pred_out['grid_size'] if pred_out['grid_size'] >= 128 else 128) + 1656
        gs = pred_out['grid_size']
        bs = pred_out['block_size']
        c0 = gpu_config['cycle_gs_coef_1']
        c1 = gpu_config['slop_bs_coef']
        slop = c1[0]*bs**2 + c1[1]*bs + c1[2]
        if no_KLL:
            kernel_lat = 0
        else:
            kernel_lat = slop*gs + c0
        kernel_detail['kernel_lat'] = kernel_lat

        # 记录有用数据 kernel_detail
        kernel_detail['active_cycles'] = pred_out["active_cycles"]
        # kernel_detail['subcore_completed'] = subcore_completed
        kernel_detail['subcore_cycles'] = ','.join([str(e) for e in subcore_completed])
        kernel_detail['comp_cycles'] = pred_out["comp_cycles"]
        kernel_detail['comp_cycles_scale'] = pred_out["comp_cycles"] * scale
        kernel_detail['scale'] = scale
        kernel_detail['scale1'] = scale1
        kernel_detail['scale2'] = scale2
        kernel_detail['scale_ori'] = scale_ori
        kernel_detail['block_actual_end'] = block_actual_end
        kernel_detail['smsp_act_cycles_avg'] = smsp_act_cycles_avg
        kernel_detail['smsp_act_cycles_avg'] = smsp_act_cycles_avg
        kernel_detail['last_inst'] = last_inst
        kernel_detail['tail'] = tail
        pred_out['kernel_detail'] = kernel_detail
        
        # 计算最终 SM cycle
        pred_out["my_gpu_act_cycles_min"] = result["ours_smsp_min"]
        pred_out["my_gpu_act_cycles_max"] = pred_out["active_cycle_scale"] + kernel_lat
        result['my_gpu_act_cycles_max'] = pred_out["my_gpu_act_cycles_max"]

        pred_out["sm_act_cycles.sum"] = pred_out["gpu_act_cycles_max"] * pred_out["active_SMs"] # 所有 SM(active) 总周期
        pred_out["sm_elp_cycles.sum"] = pred_out["gpu_act_cycles_max"] * self.acc.num_SMs
        pred_out["my_sm_act_cycles.sum"] = pred_out["my_gpu_act_cycles_max"] * pred_out["active_SMs"]
        pred_out["my_sm_elp_cycles.sum"] = pred_out["my_gpu_act_cycles_max"] * self.acc.num_SMs
        
        # 指令数相关
        # warps_instructions_executed 是单个 SM 模拟部分块的指令数
        avg_instructions_executed_per_block = pred_out["warps_instructions_executed"] / len(block_list)
        # SM 正常应该分配的线程块数（向上取整，代表分配最多的一个）
        pred_out['sm_warps_instructions_executed'] = avg_instructions_executed_per_block * pred_out["allocted_block_per_sm"]
        # 整个程序的指令数
        pred_out["tot_warps_instructions_executed"] = avg_instructions_executed_per_block * pred_out["grid_size"]
        # pred_out["tot_threads_instructions_executed"] = (pred_out["tot_warps_instructions_executed"] * self.kernel_block_size) / pred_out["allocated_active_warps_per_block"]  # 这里 allocated_active_warps_per_block 为 0 了，不知道原本逻辑是什么
        pred_out["tot_threads_instructions_executed"] = pred_out["tot_warps_instructions_executed"] * 32
        pred_out['warp_insts'] = len(next(iter(self.kernel_tasklist.items())))
        # ipc 计算
        # 方法1：程序总指令 / sm 周期之和。（仍然表示 sm 的 ipc）
        # pred_out["tot_ipc"] = pred_out["tot_warps_instructions_executed"] * (1.0/pred_out["sm_act_cycles.sum"])
        pred_out["my_tot_ipc"] = pred_out["tot_warps_instructions_executed"] * (1.0/pred_out["my_sm_act_cycles.sum"])
        pred_out['tot_ipc'] = pred_out["my_tot_ipc"]
        pred_out["tot_cpi"] = 1 * (1.0/pred_out["tot_ipc"])
        pred_out["tot_throughput_ips"] = pred_out["tot_ipc"] * self.acc.GPU_clockspeed
        pred_out["execution_time_sec"] = pred_out["sm_elp_cycles.sum"] * (1.0/self.acc.GPU_clockspeed)
        
        # 方法2：sm 总指令 / sm 周期
        pred_out['sm_ipc'] = pred_out['sm_warps_instructions_executed'] / pred_out["my_gpu_act_cycles_max"]
        pred_out['sm_cpi'] = 1/ pred_out['sm_ipc']
        
        # 方法3：在 helper_methods.py。使用 scheduler 数据。结果乘上子核数目
        # smsp_ipc
        
        # 选择 ipc
        pred_out["tot_ipc"] = pred_out["sm_ipc" if ipc_select=='' else ipc_select]
        
        # KL 和剩余部分比值
        pred_out['KL_ratio'] = kernel_lat / (pred_out["my_gpu_act_cycles_max"] - kernel_lat)

        pred_out['subcore_warp_executed'] = subcore_warp_executed
        pred_out['subcore_instr_executed'] = subcore_instr_executed
        pred_out['subcore_warp_executed_scaled'] = [e * scale for e in subcore_warp_executed]
        pred_out['subcore_instr_executed_scaled'] = [e * scale for e in subcore_instr_executed]
        
        # 不再区分原本错误的相关 cycle，输出自己的 cycle 作为最终结果
        # 对应 get_stat_sim 中使用到的结果
        pred_out["gpu_act_cycles_min"] = pred_out["my_gpu_act_cycles_min"]
        pred_out["gpu_act_cycles_max"] = pred_out["my_gpu_act_cycles_max"]
        pred_out["sm_act_cycles.sum"] = pred_out["my_sm_act_cycles.sum"]
        pred_out["sm_elp_cycles.sum"] = pred_out["my_sm_elp_cycles.sum"]
        
        toc = time.time()
        pred_out["simulation_time"]["compute"] = (toc - tic)

        pred_out["kernel_tasklist"] = self.kernel_tasklist
        warp_inst_len = [len(tasklist) for tasklist in self.kernel_tasklist.values()]
        pred_out['warp_inst_len'] = warp_inst_len
        
        ## commit results
        dump_output(pred_out)    

    def spawn_blocks(self, gpu, blocks_per_SM, warps_per_block, tasklist, kernel_id, isa, avg_mem_lat, avg_atom_lat):
        '''
        return a list of Blocks to run on one SM each with allocated number of warps
        '''
        block_list = []
        for i in range(blocks_per_SM):
            block_list.append(Block(gpu, i, warps_per_block, self.acc.num_warp_schedulers_per_SM, tasklist, kernel_id, isa, avg_mem_lat, avg_atom_lat))
        return block_list


    def blockList_has_active_warps(self, block_list):
        '''
        return true if any block in the block list is active
        '''
        for block in block_list:
            if block.is_active(): ## block is active if it has any active warp
                return True
        return False

