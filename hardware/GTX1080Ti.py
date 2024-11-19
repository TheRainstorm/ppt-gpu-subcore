##############################################################################
# This configuration file models NVIDIA VOLTA TITAN V GPU

# * GPU Microarchitecture adopted from:
# - GP102
# - https://www.techpowerup.com/gpu-specs/geforce-gtx-1080-ti.c2877
# - [Whitepaper NVIDIA GeForce GTX 1080](https://www.es.ele.tue.nl/~heco/courses/ECA/GPU-papers/GeForce_GTX_1080_Whitepaper_FINAL.pdf)

##############################################################################


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
    "num_INT_units_per_SM"              :  64,
    # represents [FP32] units 
    # responsible for Single-Precision floating point instructions 
    "num_SP_units_per_SM"               :  64,
    # represents [FP64] units in volta & Turing
    # responsible for Double-Precision floating point instructions
    "num_DP_units_per_SM"               :  32,
    # special function unites per SM
    # responsible for transcendental instructions  
    "num_SF_units_per_SM"               :  16,
    # tensor core units per SM               
    "num_TC_units_per_SM"               :  0,
    # load & store units per SM
    "num_LDS_units_per_SM"              :  16,
    # branch units per SM; ** THIS UNIT IS IN VOLTA & TURING ONLY ** 
    # to handle and execute branch instructions             
    "num_BRA_units_per_SM"              :  0,
    # texture units per SM               
    "num_TEX_units_per_SM"              :  4, # ?
    # warp scheduler units per SM
    "num_warp_schedulers_per_SM"        :  2,
    # instructions issued per warp
    "num_inst_dispatch_units_per_SM"    :  1,

    # L1 cache configs can be skipped if this option is True
    "l1_cache_bypassed"                 :  False,
    
    # In Volta, L1 cache data storage is unified with SMEM data storage
    # for a total of 128KB size for both
    # SMEM size can be: 96KB, 64KB, 32KB, 16KB, 8KB, 0KB of size
    # default config is 32KB for L1 cache size and 96KB for SMEM
    # ** Sizes are in Byte **
    "l1_cache_size"                     :  16 * 1024, # ?
    "l1_cache_line_size"                :  32, # ?
    "l1_cache_associativity"            :  64, # ?
    "l2_cache_size"                     :  2.75 * 1024*1024, # devicequery
    "l2_cache_line_size"                :  64, # ?
    "l2_cache_associativity"            :  24, # ?
    "shared_mem_size"                   :  48 * 1024,

    # L2 total size 4.5 MB, each subpartition is 96 KB. This gives ~ 48 memory parition
    "num_l2_partitions"	                :  48, # ?
    # Volta has HBM which has 24 channels each (128 bits) 16 bytes width
    "num_dram_channels"	                :  24, # ?
    # DRAM theoritical BW, measured through microbenchmarking
    "dram_th_bandwidth"                 :  565 * 10**9, # ?
    # base GPU DRAM clock speed in HZ                
    "dram_clockspeed"                   :  5505 * 10**6,  # devicequery
    # NOC theoritical BW, measured through microbenchmarking
    "noc_th_bandwidth"                  :  1140 * 10**9, # ?

    # warp scheduling: to select which warp to execute from the active warp pool 
    # options available:
    #   - LRR: Loosely Round Robin
    #   - GTO: Greedy Then Oldest -- currently not available, TO BE IMPLEMEMNTED --
    #   - TL: Two Level -- currently not available, TO BE IMPLEMEMNTED --
    "warp_scheduling"                   :  "LRR",
    
}

