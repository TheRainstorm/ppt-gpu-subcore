##############################################################################
# This configuration file models NVIDIA VOLTA TITAN V GPU

# * GPU Microarchitecture adopted from:
# - GP100
# - https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf

##############################################################################


uarch = {

    "gpu_name"                          :  "P100",
    "gpu_arch"                          :  "Pascal", #This name has to match one of the files in ISA module
    
    # compute capabilty defines the physical limits of GPUs 
    # options available:
    #   - Kepler: 35, 37
    #   - Maxwell: 50, 52, 53
    #   - Pascal: 60, 61
    #   - Volta: 70 
    #   - Turing: 75
    "compute_capabilty"                 :  60,
    
    # base GPU clock speed in HZ                
    "clockspeed"                        :  1328 * 10**6,

    # streaming multiprocessors (SMs)
    "num_SMs"                           :  56,
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
    "l1_cache_bypassed"                 :  False, # ?
    
    # In Volta, L1 cache data storage is unified with SMEM data storage
    # for a total of 128KB size for both
    # SMEM size can be: 96KB, 64KB, 32KB, 16KB, 8KB, 0KB of size
    # default config is 32KB for L1 cache size and 96KB for SMEM
    # ** Sizes are in Byte **
    "l1_cache_size"                     :  64 * 1024, # ?
    "l1_cache_line_size"                :  ,                
    "l1_cache_associativity"            :  ,  
    "l2_cache_size"                     :  4 * 1024*1024,
    "l2_cache_line_size"                :  ,             
    "l2_cache_associativity"            :  ,          
    "shared_mem_size"                   :  64 * 1024,

    "num_l2_partitions"	                :  48, # ?
    # Volta has HBM which has 24 channels each (128 bits) 16 bytes width
    "num_dram_channels"	                :  24, # ?
    # DRAM theoritical BW, measured through microbenchmarking
    "dram_th_bandwidth"                 :  565 * 10**9, #B/s
    # base GPU DRAM clock speed in HZ                
    "dram_clockspeed"                   :  850 * 10**6,
    # NOC theoritical BW, measured through microbenchmarking
    "noc_th_bandwidth"                  :  1140 * 10**9, #B/s

    # warp scheduling: to select which warp to execute from the active warp pool 
    # options available:
    #   - LRR: Loosely Round Robin
    #   - GTO: Greedy Then Oldest -- currently not available, TO BE IMPLEMEMNTED --
    #   - TL: Two Level -- currently not available, TO BE IMPLEMEMNTED --
    "warp_scheduling"                   :  "LRR",
    
}

