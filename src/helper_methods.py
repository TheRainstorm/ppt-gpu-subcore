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


import os, sys, math, time
from scipy import special as sp
import json

def dump_output(pred_out):
    '''
    模型有三个输出文件
    1. 输出报告，包含格式化的重要信息
    2. tasklist，包含每个warp的指令列表
    3. json文件，包含所有的预测信息，用于之后脚本分析
    '''
    kernel_prefix = str(pred_out["kernel_id"])+"_"+pred_out["ISA"] +"_g"+pred_out["granularity"]
    outF = open(os.path.join(pred_out["app_report_dir"], "kernel_"+kernel_prefix+".out"), "w+")
    outF2 = open(os.path.join(pred_out["app_report_dir"], f"kernel_{pred_out['kernel_id']}_tasklist.txt"), "w")
    outF3 = open(os.path.join(pred_out["app_report_dir"], f"kernel_{pred_out['kernel_id']}_pred_out.json"), "w")

    print("kernel name:", pred_out["kernel_name"], file=outF)
    
    print("\n- Total GPU computations is divided into " + str(pred_out["grid_size"])+\
                " thread block(s) running on " + str(pred_out["active_SMs"]) + " SM(s)", file=outF)
    print(f"\n- Launch Statistic:", file=outF)
    print(f"\t* Grid Size: {pred_out['grid_size']}", file=outF)
    print(f"\t* Block Size: {pred_out['block_size']}", file=outF)
    print(f"\t* Registers Per Thread: {pred_out['num_regs']}", file=outF)
    print(f"\t* Shared Memory Per Thread: {pred_out['smem_size']}", file=outF)
    
    print("\n- Modeled SM-0 running", pred_out["block_per_sm_simulate"], "thread block(s):", file=outF)
    print("\t* allocated max active thread block(s):", pred_out["max_active_block_per_sm"], file=outF)
    print("\t* allocated max active warps per thread block:", pred_out["allocated_active_warps_per_block"], file=outF)

    print("\n- Occupancy of SM-0:", file=outF)
    print("\t* Thread block limit registers:", pred_out["block_per_sm_limit_regs"], file=outF)
    print("\t* Thread block limit shared memory:", pred_out["block_per_sm_limit_smem"], file=outF)
    print("\t* Thread block limit warps:", pred_out["block_per_sm_limit_warps"], file=outF)
    print("\t* Thread block Limit SM:", pred_out["max_active_blocks_per_SM"], file=outF)
    print("\t* theoretical max active thread block(s):", pred_out["th_max_active_block_per_sm"], file=outF)
    print("\t* theoretical max active warps per SM:", pred_out["th_active_warps"], file=outF)
    print("\t* theoretical occupancy:", pred_out["th_occupancy"],"%", file=outF)
    print("\t* achieved active warps per SM:", round(pred_out["achieved_active_warps"], 2), file=outF)
    print("\t* achieved occupancy:", round(pred_out["achieved_occupancy"], 2),"%", file=outF)

    print("\n- Memory Performance:", file=outF)
    print("\t* AMAT:", pred_out["AMAT"], file=outF)
    print("\t* ACPAO:", pred_out["ACPAO"], file=outF)
    print("\t* unified L1 cache hit rate:", round((pred_out["memory_stats"]["l1_hit_rate"]*100),2),"%", file=outF)
    print("\t* unified L1 cache hit rate for read transactions (global memory accesses):", round((pred_out["memory_stats"]["l1_hit_rate_ldg"]*100),2),"%", file=outF)
    if pred_out["memory_stats"]["lmem_used"]:
        print("\t* unified L1 cache hit rate (global memory accesses):", round((pred_out["memory_stats"]["l1_hit_rate_g"]*100),2),"%", file=outF)
        
    print("\t* L2 cache hit rate:", round((pred_out["memory_stats"]["l2_hit_rate"]*100),2),"%", file=outF)
    
    print("\n\t* Global Memory Requests:", file=outF)
    print("\t\t** GMEM read requests:", pred_out["memory_stats"]["gmem_ld_reqs"], file=outF)
    print("\t\t** GMEM write requests:", pred_out["memory_stats"]["gmem_st_reqs"], file=outF)
    print("\t\t** GMEM total requests:", pred_out["memory_stats"]["gmem_tot_reqs"], file=outF)

    print("\n\t* Global Memory Transactions:", file=outF)
    print("\t\t** GMEM read transactions:", pred_out["memory_stats"]["gmem_ld_sectors"], file=outF)
    print("\t\t** GMEM write transactions:", pred_out["memory_stats"]["gmem_st_sectors"], file=outF)
    print("\t\t** GMEM total transactions:", pred_out["memory_stats"]["gmem_tot_sectors"], file=outF)

    print("\n\t* Global Memory Divergence:", file=outF)
    print("\t\t** number of read transactions per read requests: "+ str(pred_out["memory_stats"]["gmem_ld_diverg"])+\
        " ("+str(round(((pred_out["memory_stats"]["gmem_ld_diverg"]/32)*100),2))+"%)", file=outF)
    print("\t\t** number of write transactions per write requests: "+ str(pred_out["memory_stats"]["gmem_st_diverg"])+\
        " ("+str(round(((pred_out["memory_stats"]["gmem_st_diverg"]/32)*100),2))+"%)", file=outF)

    print("\n\t* L2 Cache Transactions (for global memory accesses):", file=outF)
    print("\t\t** L2 read transactions:", pred_out["memory_stats"]["l2_ld_trans"], file=outF)
    print("\t\t** L2 write transactions:", pred_out["memory_stats"]["l2_st_trans"], file=outF)
    print("\t\t** L2 total transactions:", pred_out["memory_stats"]["l2_tot_trans"], file=outF)

    print("\n\t* DRAM Transactions (for global memory accesses):", file=outF)
    print("\t\t** DRAM total transactions:", pred_out["memory_stats"]["dram_tot_trans"], file=outF)

    print("\n\t* Total number of global atomic requests:", pred_out["memory_stats"]["atom_tot_reqs"], file=outF)
    print("\t* Total number of global reduction requests:", pred_out["memory_stats"]["red_tot_reqs"], file=outF)
    print("\t* Global memory atomic and reduction transactions:", pred_out["memory_stats"]["atom_red_tot_trans"], file=outF)
    
    print("\n- Kernel cycles:", file=outF)
    # print("\t* GPU active cycles (min):", place_value(int(pred_out["gpu_act_cycles_min"])), file=outF)
    # print("\t* GPU active cycles (max):", place_value(int(pred_out["gpu_act_cycles_max"])), file=outF)
    # print("\t* SM active cycles (sum):", place_value(int(pred_out["sm_act_cycles.sum"])), file=outF)
    # print("\t* SM elapsed cycles (sum):", place_value(int(pred_out["sm_elp_cycles.sum"])), file=outF)
    print("\t* My GPU active cycles (min):", place_value(int(pred_out["my_gpu_act_cycles_min"])), file=outF)
    print("\t* My GPU active cycles (max):", place_value(int(pred_out["my_gpu_act_cycles_max"])), file=outF)
    print("\t* My SM active cycles (sum):", place_value(int(pred_out["my_sm_act_cycles.sum"])), file=outF)
    print("\t* My SM elapsed cycles (sum):", place_value(int(pred_out["my_sm_elp_cycles.sum"])), file=outF)
    print(f"result: {json.dumps(pred_out['result'], indent=4)}", file=outF)
    print(f"kernel detail: {json.dumps(pred_out['kernel_detail'], indent=4)}", file=outF)
    
    print("\n- Warp instructions executed:", place_value(int(pred_out["tot_warps_instructions_executed"])), file=outF)
    print("- Thread instructions executed:", place_value(int(pred_out["tot_threads_instructions_executed"])), file=outF)
    # print("- Instructions executed per clock cycle (IPC):", round(pred_out["tot_ipc"], 3), file=outF)
    print("- My Instructions executed per clock cycle (IPC):", round(pred_out["my_tot_ipc"], 3), file=outF)
    print("- Clock cycles per instruction (CPI): ", round(pred_out["tot_cpi"], 3), file=outF)
    print("- Total instructions executed per seconds (MIPS):", int(round((pred_out["tot_throughput_ips"]/1000000), 3)), file=outF)
    print("- Kernel execution time:", round((pred_out["execution_time_sec"]*1000000),4), "us", file=outF)

    print("\n- Scheduler Stat:", file=outF)
    active_block_per_cycle = counter_avg(pred_out["sm_stats"]["active_blocks"])
    active_warp_per_cycle_list, avg1 = counter_avg_list(pred_out["scheduler_stats"]["active_warps"])
    issued_warp_per_cycle_list, avg3 = counter_avg_list(pred_out["scheduler_stats"]["issued_warps"])
    print(f"debug: active_cycle: {pred_out['active_cycles']}", file=outF)
    print(f"debug: active_block_per_cycle: {active_block_per_cycle}", file=outF)
    print(f"Active Warps Per Scheduler: {avg1} {active_warp_per_cycle_list}", file=outF)
    print(f"Eligible Warps Per Scheduler: TODO", file=outF)
    print(f"Issued Warp Per Scheduler: {avg3} {issued_warp_per_cycle_list}", file=outF)
    no_eligible_pct = count_eq_zero_pct(pred_out["scheduler_stats"]["issued_warps"][0])
    print(f"No Eligible(subcore 0, no issued): {no_eligible_pct}", file=outF)
    
    warp_cpi_list = get_warp_cpi(pred_out['warp_stats']['stall_types'])
    sched_cpi_list = get_scheduler_cpi(pred_out['scheduler_stats']['stall_types'])
    cpi_list = [cpi['debug']['average_warp_cycle_per_inst'] for cpi in warp_cpi_list]
    avg_cpi = sum(cpi_list)/len(cpi_list)
    
    pred_out['active_block_per_cycle'] = active_block_per_cycle
    pred_out['active_warp_per_cycle'] = avg1
    pred_out['issued_warp_per_cycle'] = avg3
    pred_out['active_warp_per_cycle_smsp'] = active_warp_per_cycle_list
    pred_out['issued_warp_per_cycle_smsp'] = issued_warp_per_cycle_list
    pred_out['smsp_ipc'] = avg3 * pred_out['num_subcore']
    pred_out['smsp_cpi'] = 1 / pred_out['smsp_ipc']
    
    print("\nDebug:", file=outF)
    print(f"Grid Size: {pred_out['grid_size']}", file=outF)
    print(f"Block Size: {pred_out['block_size']}", file=outF)
    print(f"block_per_sm_simulate: {pred_out['block_per_sm_simulate']}", file=outF)
    print(f"max_active_block_per_sm: {pred_out['max_active_block_per_sm']}", file=outF)
    print(f"allocted_block_per_sm: {pred_out['allocted_block_per_sm']}", file=outF)
    
    print(f"AMAT_ori: {pred_out['AMAT_ori']}", file=outF)
    print(f"AMAT_sum: {pred_out['AMAT_sum']}", file=outF)
    print(f"AMAT_foumula: {pred_out['AMAT_foumula']}", file=outF)
    
    print(f"achieved_occupancy: {pred_out['achieved_occupancy']}", file=outF)
    print(f"active_cycle_scale: {pred_out['active_cycle_scale']}", file=outF)
    print(f"kernel_lat: {pred_out['kernel_detail']['kernel_lat']}", file=outF)
    print(f"my_gpu_act_cycles_max: {pred_out['my_gpu_act_cycles_max']}", file=outF)
    
    print(f"tot_ipc: {pred_out['tot_ipc']}", file=outF)
    print(f"tot_cpi: {pred_out['tot_cpi']}", file=outF)
    print(f"sm_ipc: {pred_out['sm_ipc']}", file=outF)
    print(f"sm_cpi: {pred_out['sm_cpi']}", file=outF)
    
    pred_out['active_warp_per_cycle_smsp'] = avg1
    pred_out['issue_warp_per_cycle_smsp'] = avg3
    pred_out['active_block_per_cycle_sm'] = active_block_per_cycle
    print(f"Active Warps Per Scheduler: {avg1} {active_warp_per_cycle_list}", file=outF)
    print(f"Issued Warp Per Scheduler: {avg3} {issued_warp_per_cycle_list}", file=outF)
    print(f"Scheduler (CPI): {1/avg3}", file=outF)
    print(f"Warp (CPI): {cpi_list} {avg_cpi}", file=outF)
    
    print(f"tot_warps_instructions_executed: {pred_out['tot_warps_instructions_executed']}", file=outF)
    print(f"sm_warps_instructions_executed: {pred_out['sm_warps_instructions_executed']}", file=outF)
    
    if 'debug_memory_print' in pred_out['memory_stats']:
        print("\n- Memory Performance:", file=outF)
        print(pred_out['memory_stats']['debug_memory_print'], file=outF)

    print("\n- Instruction Statistics:", file=outF)
    inst_count_dict = get_inst_count_dict(pred_out["kernel_tasklist"])
    print(f"{json.dumps(inst_count_dict, indent=4)}", file=outF)
    
    print("\n- Simulation Time:", file=outF)
    print("\t* Memory model:", round(pred_out["simulation_time"]["memory"], 3), "sec,", convert_sec(pred_out["simulation_time"]["memory"]), file=outF)
    print("\t* Compute model:", round(pred_out["simulation_time"]["compute"], 3), "sec,", convert_sec(pred_out["simulation_time"]["compute"]), file=outF)

    print("\n- Warp Stat:", file=outF)
    pred_out['warp_cpi'] = avg_cpi
    print(f"Warp Cycle Per Issued Instruction: {cpi_list} {avg_cpi}", file=outF)
    print(f"CPI stack Warp: {json.dumps(warp_cpi_list, indent=4)}", file=outF)
    print(f"CPI stack Scheduler: {json.dumps(sched_cpi_list, indent=4)}", file=outF)
    
    def dump_tasklist(tasklists):
        # summary
        insts_cnt = [(warp_id, len(tasklist)) for warp_id, tasklist in tasklists.items()]
        insts_cnt.sort(key=lambda x: x[0])
        print(insts_cnt, file=outF2)
        # inst
        for warp_id, warp_tasklist in tasklists.items():
            print(warp_id, file=outF2)
            for task in warp_tasklist:
                print(task, file=outF2)
    dump_tasklist(pred_out["kernel_tasklist"])
    del pred_out["kernel_tasklist"]
    
    if 'debug_memory_print' in pred_out['memory_stats']:
        del pred_out['memory_stats']['debug_memory_print']
    
    json.dump(pred_out, outF3, indent=4)
    outF2.close()
    outF3.close()
    outF.close()

def get_inst_count_dict(tasklist, scale_factor=1):
    inst_count_dict = {}
    for warp_id, warp_tasklist in tasklist.items():
        for task in warp_tasklist:
            inst_count_dict[task[0]] = inst_count_dict.get(task[0], 0) + 1
    # scale
    for inst in inst_count_dict:
        inst_count_dict[inst] *= scale_factor
    return inst_count_dict

def state_to_cpi(state_dict):
    cpi_stack = {}
    total_cycle = sum(state_dict.values())
    total_inst = state_dict.get('NoStall', 0)  # avoid zero
    if total_inst == 0:
        return {}
    
    average_warp_cycle_per_inst = total_cycle/total_inst
    
    for state in state_dict:
        cpi_stack[state] = state_dict[state]/total_inst  
    cpi_stack['debug'] = {}
    cpi_stack['debug']['total_cycle'] = total_cycle
    cpi_stack['debug']['total_inst'] = total_inst
    cpi_stack['debug']['average_warp_cycle_per_inst'] = average_warp_cycle_per_inst
    return cpi_stack

def get_scheduler_cpi(scheduler_stat_list):
    cpi_list = [state_to_cpi(stat) for stat in scheduler_stat_list]
    cpi_non_empty = [cpi for cpi in cpi_list if cpi]
    return cpi_non_empty
    
def get_warp_cpi(sample_list):
    cpi_list = [state_to_cpi(stat) for stat in sample_list]
    cpi_non_empty = [cpi for cpi in cpi_list if cpi]
    return cpi_non_empty

def counter_avg(counter):
    total_cycle = 0
    prod_sum  =0
    
    for key in counter:
        prod = key * counter[key]
        prod_sum += prod
        total_cycle += counter[key]
    return prod_sum/total_cycle

def count_eq_zero_pct(counter):
    total_cycle = 0
    eq_zero = 0
    for key in counter:
        if key == 0:
            eq_zero += counter[key]
        total_cycle += counter[key]
    return eq_zero/total_cycle
        
def counter_avg_list(counter_list):
    res = [counter_avg(counter) for counter in counter_list]
    res_no_zero = [x for x in res if x != 0]
    avg = sum(res_no_zero)/len(res_no_zero)
    return res_no_zero, avg

def place_value(number): 
    return ("{:,}".format(number))


def convert_sec(seconds): 
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def print_config_error(config_name, flag=0):
	if flag == 1:
		print("\n[Error]\nGPU Compute Capabilty \"" +config_name+"\" is not supported")
		sys.exit(1)
	elif flag == 2:
		print("\n[Error]\n\""+config_name+"\" is not defined in the hardware compute capability file")
		sys.exit(1)
	else:
		print("\n[Error]\n\""+config_name+"\" config is not defined in the hardware configuration file")
		sys.exit(1)


def print_warning(arg1, arg2, flag=False):
	if flag:
		print("\n[Warning]\n\"" + arg1 + "\" is not defined in the config file "+\
		"assuming L1 cache is "+ arg2 + "\n", file=sys.stderr)
	else:
		print("\n[Warning]\n\"" + arg1 + "\" can't be more than " + arg2\
		 	+" registers\n assuming \"" + arg1 + "\" = " + arg2 + "\n", file=sys.stderr)


def ceil(x, s=1):
	return s * math.ceil(float(x)/s)


def floor(x, s=1):
    return s * math.floor(float(x)/s)


def qfunc(arg):
    return 0.5-0.5*sp.erf(arg/1.41421)


def ncr(n, m):
    '''
    n choose m
    '''
    if(m>n): return 0
    r = 1
    for j in range(1,m+1):
        try:
            r *= (n-m+j)/float(j)
        except FloatingPointError:
            continue
    return r


class stack_el(object):
    def __init__ (self,data):
        self.address = data[0]
        self.access_time = data[1]
        self.next_el = None
        self.prev_el = None
    def __str__(self):
        return "(%s %d)" %(self.address,self.access_time)


class Stack(object):
    def __init__ (self):
        self.elements = []
        self.sp = {} #dictionary of stack pointers

    def push(self,address,t,el=None):
        if not el is None:
            se = el 
        else:
            se = stack_el((address,t))

        if not self.sp == {}:  #stack is not empty
            se.next_el = self.sp["top"]
            se.prev_el = None
            self.sp["top"].prev_el = se

        else: #stack empty
            se.next_el = None
            se.prev_el = None

        self.sp["top"] = se
        self.sp[se.access_time] = se

    def update(self,last_access,now,address):
        try:
            se = self.sp.pop(last_access) #pop deletes key, and returns value
            assert(se.address == address)
            assert(se.access_time == last_access)
            d = 0 #calculate distance from se to top of stack
            tmp = se
            while (not tmp.prev_el is None):
                d += 1
                tmp = tmp.prev_el
            if not se.prev_el is None: #remove from linked list
                se.prev_el.next_el = se.next_el
            else: 
                #the element is already at the top of the stack. 
                #just update access time and return depth
                se.access_time = now
                self.sp[se.access_time] = se
                return d
            if not se.next_el is None:
                se.next_el.prev_el = se.prev_el
            #update access time
            se.access_time = now
            #create an entry for popped element at the top of the stack
            self.push(None,None,el=se)
            #return the depth  of this element before it was brought to the top
            return d

        except KeyError:
            print ("internal error. (%s %d) not found in dictionary" %(address,last_access))
            quit()