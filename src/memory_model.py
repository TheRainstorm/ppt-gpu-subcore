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


import multiprocessing, os, math
import pickle
from joblib import Parallel, delayed
from .helper_methods import *

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def interleave_trace(smi_trace):
    '''
    return warp level interleaved trace
    '''
    max_block_len = len(max(smi_trace, key=len))
    interleaved_trace = []
    for i in range(max_block_len):
        for j in range(len(smi_trace)):
            if i < len(smi_trace[j]):
                interleaved_trace.append(smi_trace[j][i])

    return interleaved_trace


def get_line_adresses(addresses, l1_cache_line_size):
    '''
    coalescing the addresses of the warp
    '''
    line_idx = int(math.log(l1_cache_line_size,2))
    sector_size = 32
    sector_idx = int(math.log(sector_size,2))
    line_mask = ~(2**line_idx - 1)
    sector_mask = ~(2**sector_idx - 1)
    
    cache_line_set = set()
    sector_set = set()  # count sector number
    
    for addr in addresses:
        # 排除 0 ？
        if addr:
            addr = int(addr, base=16)
            cache_line = addr >> line_idx
            cache_line_set.add(cache_line)
            sector = addr >> sector_idx
            sector_set.add(sector)
    
    return list(cache_line_set), list(sector_set)


def preprocess_private_trace(interleaved_trace, SMi_trans_file, l1_cache_line_size):
    '''
    preprocess the SM addresses
    '''
    tracefile = open(SMi_trans_file, "w+")
    S = {}
    shared_trace = []
    S["gmem_ld_reqs"] = 0
    S["gmem_st_reqs"] = 0
    S["gmem_ld_trans"] = 0
    S["gmem_st_trans"] = 0
    S["lmem_ld_reqs"] = 0
    S["lmem_st_reqs"] = 0
    S["lmem_ld_trans"] = 0
    S["lmem_st_trans"] = 0
    S["lmem_used"] = False
    S["atom_reqs"] = 0
    S["red_reqs"] = 0
    S["atom_red_trans"] = 0

    warp_id = 0 ## warp counter for l2 inclusion
    inst_id = 0 ## LD=0 - ST=1
    mem_id = 2  ## Global=0 - Local=1

    for items in interleaved_trace:
        addrs = items.strip().split(" ")
        access_type = addrs[0]
        addrs.pop(0)
        line_addrs, sector_addrs = get_line_adresses(addrs, l1_cache_line_size)

        ## global reduction operations
        if "RED" in access_type:
            S["red_reqs"] += 1
            S["atom_red_trans"] += len(line_addrs)
            continue

        ## global atomic operations
        if "ATOM" in access_type:
            S["atom_reqs"] += 1
            S["atom_red_trans"] += len(line_addrs)
            continue

        warp_id += 1
        ## global memory access
        if "LDG" in access_type or "STG" in access_type: 
            mem_id = 0
            if "LDG" in access_type:
                inst_id = 0
                S["gmem_ld_reqs"] += 1
                S["gmem_ld_trans"] += len(line_addrs)
            elif "STG" in access_type:
                inst_id = 1
                S["gmem_st_reqs"] += 1
                S["gmem_st_trans"] += len(line_addrs)
        
        ## local memory access
        elif "LDL" in access_type or "STL" in access_type:
            mem_id = 1
            S["lmem_used"] = True
            if "LDL" in access_type:
                inst_id = 0
                S["lmem_ld_reqs"] += 1
                S["lmem_ld_trans"] += len(line_addrs)
            elif "STL" in access_type:
                inst_id = 1
                S["lmem_st_reqs"] += 1
                S["lmem_st_trans"] += len(line_addrs)

        for individual_addrs in line_addrs:
            ####
            print(inst_id, mem_id, warp_id, individual_addrs, file=tracefile)
            ###
        
        shared_trace.append(line_addrs)

    return S, shared_trace


def preprocess_shared_trace(interleaved_trace, kernel_id, mem_trace_dir_path):
    '''
    preprocess the shared addresses
    '''
    shared_trace = mem_trace_dir_path+"/K"+str(kernel_id)+"_shared.trace"
    shared_trace_file = open(shared_trace, "w+")
    num_lines = 0

    for line_addrs in interleaved_trace:
        for individual_addrs in line_addrs: 
            num_lines += 1
            ####
            print(individual_addrs, file=shared_trace_file)
            ####

    return num_lines, shared_trace


def get_hit_rate_analytical(reuse_profile, cache_size, line_size, associativity):
    '''
    return the hit rate given RP, cache size, line size, and associativity
    '''
    phit = 0.0 ## Sum (probability of stack distance * probability of hit given D)
    num_blocks = (1.0 * cache_size)/line_size  ## B = Num of Blocks (cache_size/line_size)
    for items in reuse_profile:
        if not items:
            continue
        items = items.split(",")
        stack_distance = int(items[0])
        probability = float(items[1])
        # try:
        #     probability = float(items[1])
        # except:
        #     probability = 0
        ## compute probability of a hit given D
        if stack_distance == -1:   phit += 0
        elif stack_distance == 0:  phit += probability
        else:
            mean = stack_distance * (associativity/num_blocks)
            variance = mean * ((num_blocks-associativity)/num_blocks)
            phit += probability * ( 1 - qfunc( abs(associativity-1-mean) / math.sqrt(variance) ) )

    return phit


def private_SM_computation(SM_id, kernel_id, grid_size, num_SMs, mem_trace_dir_path, max_blocks_per_SM,\
                          l1_cache_size, l1_cache_line_size, l1_cache_associativity, use_sm_trace=False):

    SM_stats = {}
    smi_trace = []
    shared_trace = []
    
    if use_sm_trace:
        trace_file = mem_trace_dir_path+"/kernel_"+str(kernel_id)+"_sm_"+str(SM_id)+".mem"
        interleaved_trace = open(trace_file,'r').readlines()
        smi_trace = interleaved_trace  # smi_trace just for check, not used
    else:
        count_blocks = 0
        for block_id in range(grid_size):
            if count_blocks > max_blocks_per_SM:
                break
            current_block_id = block_id % num_SMs
            if current_block_id == SM_id:
                try:
                    count_blocks += 1
                    trace_file = mem_trace_dir_path+"/kernel_"+str(kernel_id)+"_block_"+str(block_id)+".mem"
                    # block_trace = open(trace_file,'r').read().strip().split("\n=====\n")
                    block_trace = open(trace_file,'r').readlines()
                    smi_trace.append(block_trace)
                except:
                    # print("\n[Warning]\n"+trace_file+" not found\n")
                    continue

    if smi_trace:
        SMi_trans_file = mem_trace_dir_path+"/K"+str(kernel_id)+"_SM"+str(SM_id)+".trace"
        umem_rpi_file = mem_trace_dir_path+"/K"+str(kernel_id)+"_UMEM_SM"+str(SM_id)+".rp"
        gmem_rpi_lds_file = mem_trace_dir_path+"/K"+str(kernel_id)+"_GMEM_SM"+str(SM_id)+"_lds.rp"
        gmem_rpi_file = mem_trace_dir_path+"/K"+str(kernel_id)+"_GMEM_SM"+str(SM_id)+".rp"
        lmem_rpi_file = mem_trace_dir_path+"/K"+str(kernel_id)+"_LMEM_SM"+str(SM_id)+".rp"

        # interleaved_trace = []
        # assert len(smi_trace)==20
        # concurrent_blocks = 2
        # for i in range(0, len(smi_trace), concurrent_blocks):
        #     current_blocks = smi_trace[i:i+concurrent_blocks]
        #     current_interleave_trace = interleave_trace(current_blocks)
        #     interleaved_trace += current_interleave_trace
        if not use_sm_trace:
            interleaved_trace = interleave_trace(smi_trace)

        memory_stats, shared_trace = preprocess_private_trace(interleaved_trace, SMi_trans_file, l1_cache_line_size)
        SM_stats.update(memory_stats)
        lmem_used = SM_stats["lmem_used"]

        gmem_num_lines = SM_stats["gmem_ld_trans"] + SM_stats["gmem_st_trans"]
        lmem_num_lines = SM_stats["lmem_ld_trans"] + SM_stats["lmem_st_trans"]
        umem_num_lines_tot = gmem_num_lines + lmem_num_lines
        umem_num_lines_lds = SM_stats["gmem_ld_trans"] + SM_stats["lmem_ld_trans"] 

        ## call PARDA executable file to compute the RD & RP
        umem_hit_rate = 0.0
        gmem_hit_rate_ld = 0.0
        gmem_hit_rate = 0.0
        lmem_hit_rate = 0.0

        if lmem_used:
            cmd = f"{repo_dir}/reuse_distance_tool/parda.x --input="+SMi_trans_file+" --sm_id="+str(SM_id)+" --lines="+str(umem_num_lines_tot)+\
                  " --assoc="+str(l1_cache_associativity)+" --output_dir="+mem_trace_dir_path+" --kernel="+str(kernel_id)+" --lmem"
            os.system(cmd)
        else: 
            cmd = f"{repo_dir}/reuse_distance_tool/parda.x --input="+SMi_trans_file+" --sm_id="+str(SM_id)+" --lines="+str(umem_num_lines_tot)+\
                " --assoc="+str(l1_cache_associativity)+" --output_dir="+mem_trace_dir_path+" --kernel="+str(kernel_id)
            os.system(cmd)

        ## calculate the hit rates from the RP
        rpi_umem = open(umem_rpi_file,'r').read().strip().split('\n')
        if rpi_umem != [""]:
            umem_hit_rate = get_hit_rate_analytical(rpi_umem, l1_cache_size, l1_cache_line_size, l1_cache_associativity)
            rpi_gmem_lds = open(gmem_rpi_lds_file,'r').read().strip().split('\n')
            if rpi_gmem_lds != [""]:
                gmem_hit_rate_ld = get_hit_rate_analytical(rpi_gmem_lds, l1_cache_size, l1_cache_line_size, l1_cache_associativity)
        
        SM_stats["umem_hit_rate"] = umem_hit_rate
        SM_stats["gmem_hit_rate_ld"] = gmem_hit_rate_ld

        if lmem_used:
            rpi_gmem = open(gmem_rpi_file,'r').read().strip().split('\n')
            gmem_hit_rate = get_hit_rate_analytical(rpi_gmem, l1_cache_size, l1_cache_line_size, l1_cache_associativity)

            rpi_lmem = open(lmem_rpi_file,'r').read().strip().split('\n')
            lmem_hit_rate = get_hit_rate_analytical(rpi_lmem, l1_cache_size, l1_cache_line_size, l1_cache_associativity)

        SM_stats["gmem_hit_rate"] = gmem_hit_rate
        SM_stats["lmem_hit_rate"] = lmem_hit_rate

        SM_stats["shared_trace"] = shared_trace

        cmd = "rm "+ SMi_trans_file+" "+umem_rpi_file+" "+gmem_rpi_lds_file+" "
        if lmem_used:
            cmd += gmem_rpi_file+" "+lmem_rpi_file
        os.system(cmd)

    return SM_stats




def get_memory_perf(kernel_id, mem_trace_dir_path, grid_size, num_SMs, l1_cache_size, l1_cache_line_size, l1_cache_associativity,\
                    l2_cache_size, l2_cache_line_size, l2_cache_associativity, gmem_reqs, allocted_block_per_sm, block_per_sm_simulate, cache_ref_data=None):

    blck_id = -1
    shared_trace = []
    parallel_out_list = []
    umem_hit_rates_list = []
    gmem_hit_rates_lds_list = []
    gmem_hit_rates_list = []
    gmem_ld_reqs_list = []
    gmem_st_reqs_list = []
    gmem_ld_trans_list = []
    gmem_st_trans_list = []
    lmem_hit_rates_list = []
    lmem_ld_reqs_list = []
    lmem_st_reqs_list = []
    lmem_ld_trans_list = []
    lmem_st_trans_list = []
    atom_reqs_list = []
    red_reqs_list = []
    atom_red_trans_list = []
    shared_trace_max_block_len = 0
    
    memory_stats = {}
    memory_stats["hit_rate_l2"] = 0  # nvprof set 100%
    memory_stats["umem_hit_rate"] = 0
    memory_stats["gmem_hit_rate_lds"] = 0
    memory_stats["gmem_hit_rate"] = 0
    memory_stats["gmem_ld_reqs"] = 0
    memory_stats["gmem_st_reqs"] = 0
    memory_stats["gmem_tot_reqs"] = 0
    memory_stats["gmem_ld_trans"] = 0
    memory_stats["gmem_avg_ld_trans"] = 0
    memory_stats["gmem_st_trans"] = 0
    memory_stats["gmem_avg_st_trans"] = 0
    memory_stats["gmem_tot_trans"] = 0
    memory_stats["gmem_avg_trans"] = 0
    memory_stats["gmem_ld_diverg"] = 0
    memory_stats["gmem_st_diverg"] = 0
    memory_stats["gmem_tot_diverg"] = 0
    memory_stats["l1_sm_trans_gmem"] = 0
    memory_stats["l2_ld_trans_gmem"] = 0
    memory_stats["l2_st_trans_gmem"] = 0
    memory_stats["l2_tot_trans_gmem"] = 0
    memory_stats["l2_avg_trans_gmem"] = 0
    memory_stats["dram_tot_trans_gmem"] = 0
    memory_stats["dram_avg_trans_gmem"] = 0
    memory_stats["lmem_used"] = False
    memory_stats["lmem_hit_rate"] = 0
    memory_stats["lmem_tot_reqs"] = 0
    memory_stats["lmem_tot_trans"] = 0
    memory_stats["atom_tot_reqs"] = 0
    memory_stats["red_tot_reqs"] = 0
    memory_stats["atom_red_tot_trans"] = 0

    if gmem_reqs == 0:
        return memory_stats
    
    num_cores = multiprocessing.cpu_count()
    parallel_comp = 0
    if grid_size < num_SMs:
        parallel_comp = grid_size
    else:
        parallel_comp = num_SMs

    num_jobs = min(parallel_comp, num_cores)
    # if os.path.exists('sm_output.pickle'):  # to accelerate debug
    #     print("read from pickle")
    #     with open('sm_output.pickle', 'rb') as f:
    #         SMs_output_list = pickle.load(f)
    # else:
    #     SMs_output_list = Parallel(n_jobs=num_jobs, prefer="processes")(delayed(private_SM_computation)(i, kernel_id, grid_size,\
    #                                                                                                 num_SMs, mem_trace_dir_path, block_per_sm_simulate,\
    #                                                                                                 l1_cache_size, l1_cache_line_size,\
    #                                                                                                 l1_cache_associativity)\
    #                                                                                                 for i in range(parallel_comp))
    # with open('sm_output.pickle', 'wb') as f:
    #     print("write to pickle")
    #     pickle.dump(SMs_output_list, f)
    SMs_output_list = Parallel(n_jobs=num_jobs, prefer="processes")(delayed(private_SM_computation)(i, kernel_id, grid_size,\
                                                                                                num_SMs, mem_trace_dir_path, block_per_sm_simulate,\
                                                                                                l1_cache_size, l1_cache_line_size,\
                                                                                                l1_cache_associativity)\
                                                                                                for i in range(parallel_comp))

    for SMi_stats in SMs_output_list:
        if SMi_stats:
            memory_stats["lmem_used"] = SMi_stats["lmem_used"]
            gmem_ld_reqs_list.append(SMi_stats["gmem_ld_reqs"])
            gmem_st_reqs_list.append(SMi_stats["gmem_st_reqs"])
            gmem_ld_trans_list.append(SMi_stats["gmem_ld_trans"])
            gmem_st_trans_list.append(SMi_stats["gmem_st_trans"])
            lmem_ld_reqs_list.append(SMi_stats["lmem_ld_reqs"])
            lmem_st_reqs_list.append(SMi_stats["lmem_ld_reqs"])
            lmem_ld_trans_list.append(SMi_stats["lmem_ld_trans"])
            lmem_st_trans_list.append(SMi_stats["lmem_ld_trans"])
            atom_reqs_list.append(SMi_stats["atom_reqs"])
            red_reqs_list.append(SMi_stats["red_reqs"])
            atom_red_trans_list.append(SMi_stats["atom_red_trans"])
            umem_hit_rates_list.append(SMi_stats["umem_hit_rate"])
            gmem_hit_rates_lds_list.append(SMi_stats["gmem_hit_rate_ld"])
            gmem_hit_rates_list.append(SMi_stats["gmem_hit_rate"])
            lmem_hit_rates_list.append(SMi_stats["lmem_hit_rate"])
            shared_trace.append(SMi_stats["shared_trace"])
            # shared_trace_max_block_len = max(shared_trace_max_block_len, len(SMi_stats["shared_trace"]))
        

    if cache_ref_data:
        memory_stats["hit_rate_l2"] = cache_ref_data['l2_hit_rate']
    else:
        shared_interleaved_trace = interleave_trace(shared_trace)
        shared_num_lines, shared_trace_file = preprocess_shared_trace(shared_interleaved_trace, kernel_id, mem_trace_dir_path)

        ## call reuse_distance_tool to compute the RD & RP
        cmd = f"{repo_dir}/reuse_distance_tool/parda.x --input="+shared_trace_file+" --lines="+str(shared_num_lines)+\
            " --assoc="+str(l2_cache_associativity)+" --output_dir="+mem_trace_dir_path+" --kernel="+str(kernel_id)+" --l2"
        os.system(cmd)

        ## calculate the hit rates from the RP
        shared_rp = mem_trace_dir_path+"/K"+str(kernel_id)+"_shared.rp"
        rp_l2 = open(shared_rp,'r').read().strip().split('\n')
        memory_stats["hit_rate_l2"] = get_hit_rate_analytical(rp_l2, l2_cache_size, l2_cache_line_size, l2_cache_associativity)

        cmd = "rm "+ shared_trace_file+ " "+shared_rp
        os.system(cmd)

    ## ---- unified (l1/tex/local) memory ---- ##
    memory_stats["umem_hit_rate_list"] = umem_hit_rates_list
    memory_stats["umem_hit_rate"] = sum(umem_hit_rates_list) / len(umem_hit_rates_list)
    memory_stats["gmem_hit_rate_lds"] = sum(gmem_hit_rates_lds_list) / len(gmem_hit_rates_lds_list)

    ## ---- atomic & reduction instructions ---- ##
    memory_stats["atom_tot_reqs"] = int(sum(atom_reqs_list) / len(atom_reqs_list))
    memory_stats["red_tot_reqs"] = int(sum(red_reqs_list) / len(red_reqs_list))
    memory_stats["atom_red_tot_trans"] = int(sum(atom_red_trans_list) / len(atom_red_trans_list))

    ## ---- global memory ---- ##
    if memory_stats["lmem_used"]:
        memory_stats["gmem_hit_rate"] = sum(gmem_hit_rates_list) / len(gmem_hit_rates_list)
    else:
        memory_stats["gmem_hit_rate"] = memory_stats["umem_hit_rate"]
    memory_stats["gmem_ld_reqs"] = int((sum(gmem_ld_reqs_list) * allocted_block_per_sm) / block_per_sm_simulate)
    memory_stats["gmem_st_reqs"] = int((sum(gmem_st_reqs_list) * allocted_block_per_sm) / block_per_sm_simulate)
    memory_stats["gmem_tot_reqs"] = memory_stats["gmem_ld_reqs"] + memory_stats["gmem_st_reqs"]
    memory_stats["gmem_ld_trans"] = int((sum(gmem_ld_trans_list) * allocted_block_per_sm) / block_per_sm_simulate)
    gmem_sm_ld_trans = int(sum(gmem_ld_trans_list) / len(gmem_ld_trans_list))
    memory_stats["gmem_sm_ld_trans"] = int((gmem_sm_ld_trans * allocted_block_per_sm) / block_per_sm_simulate)
    memory_stats["gmem_st_trans"] = int((sum(gmem_st_trans_list) * allocted_block_per_sm) / block_per_sm_simulate)
    gmem_sm_st_trans = int(sum(gmem_st_trans_list) / len(gmem_st_trans_list))
    memory_stats["gmem_sm_st_trans"] = int((gmem_sm_st_trans * allocted_block_per_sm) / block_per_sm_simulate)
    memory_stats["gmem_tot_trans"] = memory_stats["gmem_ld_trans"] + memory_stats["gmem_st_trans"]
    memory_stats["l1_sm_trans_gmem"] =  memory_stats["gmem_sm_ld_trans"] + memory_stats["gmem_sm_st_trans"]

    if memory_stats["gmem_ld_reqs"] != 0:
        memory_stats["gmem_ld_diverg"] = round(memory_stats["gmem_ld_trans"] / memory_stats["gmem_ld_reqs"], 2)
    if memory_stats["gmem_st_reqs"] != 0:
        memory_stats["gmem_st_diverg"] = round(memory_stats["gmem_st_trans"] / memory_stats["gmem_st_reqs"], 2)
    if memory_stats["gmem_tot_reqs"] != 0:
        memory_stats["gmem_tot_diverg"] = round(memory_stats["gmem_tot_trans"] / memory_stats["gmem_tot_reqs"], 2)

    ## ---- l2 transactions from l1 cache for global memory accesses only ---- ##
    memory_stats["l2_ld_trans_gmem"] = int(memory_stats["gmem_ld_trans"] * (1 - memory_stats["gmem_hit_rate_lds"]))
    memory_stats["l2_st_trans_gmem"] = memory_stats["gmem_st_trans"]
    memory_stats["l2_tot_trans_gmem"] = memory_stats["l2_ld_trans_gmem"] + memory_stats["l2_st_trans_gmem"]
    # memory_stats["l2_avg_trans_gmem"] = int(memory_stats["gmem_tot_trans"] * (1 - memory_stats["gmem_hit_rate"]))

    ## ---- DRAM transactions from l2 for global memory accesses only---- ##
    memory_stats["dram_tot_trans_gmem"] = int(memory_stats["l2_tot_trans_gmem"] * (1 - memory_stats["hit_rate_l2"]))
    memory_stats["dram_ld_trans_gmem"] =  memory_stats["l2_ld_trans_gmem"] * (1 - memory_stats["hit_rate_l2"])
    memory_stats["dram_st_trans_gmem"] =  memory_stats["l2_st_trans_gmem"] * (1 - memory_stats["hit_rate_l2"])
    # memory_stats["dram_avg_trans_gmem"] = int(memory_stats["l2_avg_trans_gmem"] * (1 - memory_stats["hit_rate_l2"]))
    

    ## ---- local memory ---- ##
    if memory_stats["lmem_used"]:
        memory_stats["lmem_hit_rate"] = sum(lmem_hit_rates_list) / len(lmem_hit_rates_list)
        memory_stats["lmem_tot_reqs"] = sum(lmem_ld_trans_list) + sum(lmem_st_trans_list)
        memory_stats["lmem_tot_trans"] = sum(lmem_ld_trans_list) + sum(lmem_st_trans_list)


    return  memory_stats
