##############################################################################
# This configuration file models NVIDIA VOLTA TITAN V GPU

# * GPU Microarchitecture adopted from:
# - GP102
# - [1] https://www.techpowerup.com/gpu-specs/geforce-gtx-1080-ti.c2877
# - [2] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-6-x
# - [Whitepaper NVIDIA GeForce GTX 1080](https://www.es.ele.tue.nl/~heco/courses/ECA/GPU-papers/GeForce_GTX_1080_Whitepaper_FINAL.pdf)
# - devicequery

##############################################################################


from src.cache_simulator import W
uarch = {

    "gpu_name"                          :  "GTX 1080Ti",
    "gpu_arch"                          :  "Pascal", #This name has to match one of the files in ISA module
    
    # compute capabilty defines the physical limits of GPUs 
    # options available:
    #   - Kepler: 35, 37
    #   - Maxwell: 50, 52, 53
    #   - Pascal: 60, 61
    #   - Volta: 70 
    #   - Turing: 75
    "compute_capabilty"                 :  61,
    
    # base GPU clock speed in HZ                
    "clockspeed"                        :  1481 * 10**6,

    # streaming multiprocessors (SMs)
    "num_SMs"                           :  28,
    # represents [INT] units; ** THIS UNIT IS IN VOLTA & TURING ONLY ** 
    # responsible for int instructions
    "num_INT_units_per_SM"              :  0,
    # represents [FP32] units 
    # responsible for Single-Precision floating point instructions 
    "num_SP_units_per_SM"               :  128, # [2], 64 for GP100
    # represents [FP64] units in volta & Turing
    # responsible for Double-Precision floating point instructions
    "num_DP_units_per_SM"               :  32,
    # special function unites per SM
    # responsible for transcendental instructions  
    "num_SF_units_per_SM"               :  32,  # [2], 16 for GP100
    # tensor core units per SM               
    "num_TC_units_per_SM"               :  0,
    # load & store units per SM
    "num_LDS_units_per_SM"              :  16,  # ?
    # branch units per SM; ** THIS UNIT IS IN VOLTA & TURING ONLY ** 
    # to handle and execute branch instructions             
    "num_BRA_units_per_SM"              :  0,
    # texture units per SM               
    "num_TEX_units_per_SM"              :  4, # ?
    # warp scheduler units per SM
    "num_warp_schedulers_per_SM"        :  4,   # GP100, 2 per SM, [2]
    # instructions issued per warp
    "num_inst_dispatch_units_per_SM"    :  1,

    # L1 cache configs can be skipped if this option is True
    "l1_cache_bypassed"                 :  False,       # obsolete now, TODO: implement bypass L1 in memory model
    
    # In Volta, L1 cache data storage is unified with SMEM data storage
    # for a total of 128KB size for both
    # SMEM size can be: 96KB, 64KB, 32KB, 16KB, 8KB, 0KB of size
    # default config is 32KB for L1 cache size and 96KB for SMEM
    # ** Sizes are in Byte **
    "l1_cache_size"                     :  48 * 1024, # [2]
    "l1_cache_line_size"                :  128,
    "l1_cache_associativity"            :  4,
    "l1_sector_size"                    :  32,
    "l1_write_allocate"                 :  True,
    "l1_write_strategy"                 :  W.write_through,
    
    "l2_cache_size"                     :  2.75 * 1024*1024, # devicequery, [1]
    "l2_cache_line_size"                :  128,
    "l2_cache_associativity"            :  24,
    "l2_sector_size"                    :  32,
    "l2_write_allocate"                 :  True,
    "l2_write_strategy"                 :  W.write_back,
    
    "adaptive_cache"                    :  False,
    "shared_mem_size"                   :  96 * 1024,   # [2]

    "num_l2_partitions"	                :  48, # ?
    "num_dram_channels"	                :  24, # ?
    # DRAM theoritical BW, measured through microbenchmarking
    "dram_th_bandwidth"                 :  484.4 * 10**9, # [1]
    # base GPU DRAM clock speed in HZ                
    "dram_clockspeed"                   :  1376 * 10**6,  # [1]
    # NOC theoritical BW, measured through microbenchmarking
    "noc_th_bandwidth"                  :  1140 * 10**9, # ?

    # warp scheduling: to select which warp to execute from the active warp pool 
    # options available:
    #   - LRR: Loosely Round Robin
    #   - GTO: Greedy Then Oldest -- currently not available, TO BE IMPLEMEMNTED --
    #   - TL: Two Level -- currently not available, TO BE IMPLEMEMNTED --
    "warp_scheduling"                   :  "LRR",
    
}

