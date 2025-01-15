##############################################################################
# This configuration file models NVIDIA A100 PCIe 40 GB

# * GPU Microarchitecture adopted from:
# - [1] https://www.techpowerup.com/gpu-specs/titan-v.c3051
# - [2] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
# - [3] https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf

##############################################################################


from src.cache_simulator import W
uarch = {

    "gpu_name"                          :  "A100-40G",
    "gpu_arch"                          :  "Ampere", #This name has to match one of the files in ISA module
    
    # compute capabilty defines the physical limits of GPUs 
    # options available:
    #   - Kepler: 35, 37
    #   - Maxwell: 50, 52, 53
    #   - Pascal: 60(P100), 61(1080)
    #   - Volta: 70 
    #   - Turing: 75
    #   - Ampere: 80(A100), 86(3090)
    "compute_capabilty"                 :  80,
    
    # base GPU clock speed in HZ                
    "clockspeed"                        :  1410 * 10**6, # [2], boost, base 765MHz

    # streaming multiprocessors (SMs)
    "num_SMs"                           :  108,
    # represents [INT] units; ** THIS UNIT IS IN VOLTA & TURING ONLY ** 
    # responsible for int instructions
    "num_INT_units_per_SM"              :  64,  # [2]
    # represents [FP32] units 
    # responsible for Single-Precision floating point instructions 
    "num_SP_units_per_SM"               :  64,  # [2]
    # represents [FP64] units in volta & Turing
    # responsible for Double-Precision floating point instructions
    "num_DP_units_per_SM"               :  32,  # [2]
    # special function unites per SM
    # responsible for transcendental instructions  
    "num_SF_units_per_SM"               :  16,  # [2]
    # tensor core units per SM               
    "num_TC_units_per_SM"               :  4,   # [2]
    # load & store units per SM
    "num_LDS_units_per_SM"              :  16,
    # branch units per SM; ** THIS UNIT IS IN VOLTA & TURING ONLY ** 
    # to handle and execute branch instructions             
    "num_BRA_units_per_SM"              :  4,
    # texture units per SM               
    "num_TEX_units_per_SM"              :  4,
    # warp scheduler units per SM
    "num_warp_schedulers_per_SM"        :  4,   # [2]
    # instructions issued per warp
    "num_inst_dispatch_units_per_SM"    :  1,

    # L1 cache configs can be skipped if this option is True
    "l1_cache_bypassed"                 :  False,   # obsolete now, TODO: implement bypass L1 in memory model
    
    # In Ampere, L1 cache data storage is unified with SMEM data storage
    "shared_mem_size"                   :  192 * 1024,  # [2], 192 KB for 8.0 and 8.7, 128 KB for 8.6, 8.9
    # Similar to the Volta architecture, the amount of the unified data cache reserved for
    # shared memory is configurable on a per kernel basis.
    # When using adaptive cache, `shared_mem_size` refers to the total L1 cache size. 
    # Possible shared memory sizes are defined in `shared_mem_carveout`.
    "adaptive_cache"                   :  True,
    "shared_mem_carveout"               : [0, 8, 16, 32, 64, 100, 132, 164],  # [2], 8.0, 8.7. For 8.6, 8.9,  [0, 8, 16, 32, 64, 100]
    
    # when use adaptive cache
    "l1_cache_size"                     :  96 * 1024,   # [2]
    "l1_cache_line_size"                :  128,
    "l1_cache_associativity"            :  4,
    "l1_sector_size"                    :  32,
    "l1_write_allocate"                 :  True,
    "l1_write_strategy"                 :  W.write_through,
    "l2_cache_size"                     :  40 * 1024*1024,   # [1]
    "l2_cache_line_size"                :  128,
    "l2_cache_associativity"            :  32,
    "l2_sector_size"                    :  32,
    "l2_write_allocate"                 :  True,
    "l2_write_strategy"                 :  W.write_back,

    "num_l2_partitions"	                :  48, # ?
    "num_dram_channels"	                :  24, # ?
    # DRAM theoritical BW, measured through microbenchmarking
    "dram_th_bandwidth"                 :  1560 * 10**9, #B/s   # [1]
    # base GPU DRAM clock speed in HZ                
    "dram_clockspeed"                   :  1215 * 10**6, # [1]
    # NOC theoritical BW, measured through microbenchmarking
    "noc_th_bandwidth"                  :  1140 * 10**9, # ?

    # warp scheduling: to select which warp to execute from the active warp pool 
    # options available:
    #   - LRR: Loosely Round Robin
    #   - GTO: Greedy Then Oldest -- currently not available, TO BE IMPLEMEMNTED --
    #   - TL: Two Level -- currently not available, TO BE IMPLEMEMNTED --
    "warp_scheduling"                   :  "LRR",
    
    "cycle_gs_coef_1": 2446.57,
    "slop_bs_coef": [4.57e-07, 0.00031, 0.91],
}

