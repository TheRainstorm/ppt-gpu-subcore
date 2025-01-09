##############################################################################
# This configuration file models NVIDIA VOLTA TITAN V GPU

# * GPU Microarchitecture adopted from:
# - https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
# - https://www.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.10-GPU-Gaming-Pub/HC29.21.132-Volta-Choquette-NVIDIA-Final3.pdf
# - https://ieeexplore.ieee.org/document/8344474

# - [1] https://www.techpowerup.com/gpu-specs/titan-v.c3051
# - [2] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x
# - [3] [Dissecting the NVIDIA Volta GPUArchitecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826) 
# - [4] Khairy, Mahmoud, Jain Akshay, Tor Aamodt, and Timothy G. Rogers. “Exploring Modern GPU Memory System Design Challenges through Accurate Modeling.” In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), 473–86, 2020. https://doi.org/10.1109/ISCA45697.2020.00047.

##############################################################################

from src.cache_simulator import W
uarch = {

    "gpu_name"                          :  "TITAN V",
    "gpu_arch"                          :  "Volta", #This name has to match one of the files in ISA module
    
    "compute_capabilty"                 :  70,
    
    # base GPU clock speed in HZ                
    "clockspeed"                        :  1200 * 10**6,    # [1] 1200 base, 1455  boost

    # streaming multiprocessors (SMs)
    "num_SMs"                           :  80,
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
    "num_TC_units_per_SM"               :  8,   # [2]
    # load & store units per SM
    "num_LDS_units_per_SM"              :  32,
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
    
    # In Volta, L1 cache data storage is unified with SMEM data storage
    # for a total of 128KB size for both
    # SMEM size can be: 96KB, 64KB, 32KB, 16KB, 8KB, 0KB of size
    # default config is 32KB for L1 cache size and 96KB for SMEM
    # ** Sizes are in Byte **
    "shared_mem_size"                   :  128 * 1024,
    # When using adaptive cache, `shared_mem_size` refers to the total L1 cache size. 
    # Possible shared memory sizes are defined in `shared_mem_carveout`.
    "adaptive_cache"                   :  True,
    "shared_mem_carveout"               : [0, 8, 16, 32, 64, 96],    # [2]
    
    "l1_cache_size"                     :  32 * 1024,
    "l1_cache_line_size"                :  128,
    "l1_cache_associativity"            :  4,
    "l1_sector_size"                    :  32,
    "l1_write_allocate"                 :  True,                # write non-allocate make worse simulation results
    "l1_write_strategy"                 :  W.write_through,
    "l2_cache_size"                     :  4.5 * 1024*1024,     # [1]
    "l2_cache_line_size"                :  128,
    "l2_cache_associativity"            :  32,
    "l2_sector_size"                    :  32,
    "l2_write_allocate"                 :  True,
    "l2_write_strategy"                 :  W.write_back,        # [4]

    # L2 total size 4.5 MB, each subpartition is 96 KB. This gives ~ 48 memory parition
    "num_l2_partitions"	                :  48,
    # Volta has HBM which has 24 channels each (128 bits) 16 bytes width
    "num_dram_channels"	                :  24,
    # DRAM theoritical BW, measured through microbenchmarking
    "dram_th_bandwidth"                 :  900 * 10**9, #B/s    # [3]
    # base GPU DRAM clock speed in HZ                
    "dram_clockspeed"                   :  877 * 10**6,         # [1]
    # NOC theoritical BW, measured through microbenchmarking
    "noc_th_bandwidth"                  :  1140 * 10**9, #B/s

    # warp scheduling: to select which warp to execute from the active warp pool 
    # options available:
    #   - LRR: Loosely Round Robin
    #   - GTO: Greedy Then Oldest -- currently not available, TO BE IMPLEMEMNTED --
    #   - TL: Two Level -- currently not available, TO BE IMPLEMEMNTED --
    "warp_scheduling"                   :  "LRR",
    
    "cycle_gs_coef_1": 2717.61,
    "slop_bs_coef": [1.23e-07, -8.59e-05, 2.13],
}

