##############################################################################
# SASS instructions adpoted from:
# - https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html

# Instructions Latencies adopted from: 
# Y. Arafa et al., "Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs," HPEC'19
# https://github.com/NMSU-PEARL/GPUs-ISA-Latencies

##############################################################################

units_latency = {

    #ALU Units Latencies
    "iALU"              :   4,
    "fALU"              :   4,
    "hALU"              :   4,
    "dALU"              :   8,
    
    "SFU"               :   4,
    "dSFU"              :   8,
    
    "iTCU"              :   4,
    "hTCU"              :   4,
    
    "BRA"               :   4,
    #Memory Units Latencies
    "dram_mem_access"   :   302,
    "l1_cache_access"   :   33,
    "l2_cache_access"   :   213,
    "local_mem_access"  :   302,
    "const_mem_access"  :   8,
    "shared_mem_access" :   33,
    "tex_mem_access"    :   436,
    "tex_cache_access"  :   86,
    "atomic_operation"  :   245,
    #ThreadBlock
    "TB_launch_ovhd"    :   700

}

# initial_interval = {
#     # Initiation interval (II) = threadsPerWarp / #FULanes
#     "iALU"              :   32 // 16,
#     "fALU"              :   32 // 16,
#     "hALU"              :   32 // 16,
#     "dALU"              :   32 // 8,

#     "SFU"               :   32 // 4,
#     "dSFU"              :   32 // 4,

#     "LDST"              :   32 // 8,   # load store unit
    
#     # "bTCU"              :   64,
#     "iTCU"              :   32 // 2,
#     "hTCU"              :   32 // 2,
#     # "fTCU"              :   32 // 2,
#     # "dTCU"              :   32 // 2,
#     "BRA"               :   32 // 32,
#     "EXIT"              :   32 // 16,
# }

sass_isa = {

    # Integer Instructions
    "BMSK"              : "iALU",
    "BREV"              : "iALU",
    "FLO"               : "iALU",
    "IABS"              : "iALU",
    "IADD"              : "iALU",
    "IADD3"             : "iALU",
    "IADD32I"           : "iALU",
    "IDP"               : "iALU",
    "IDP4A"             : "iALU",
    "IMAD"              : "iALU",
    "IMNMX"             : "iALU",
    "IMUL"              : "iALU",
    "IMUL32I"           : "iALU",
    "ISCADD"            : "iALU",
    "ISCADD32I"         : "iALU",
    "ISETP"             : "iALU",
    "LEA"               : "iALU",
    "LOP"               : "iALU",
    "LOP3"              : "iALU",
    "LOP32I"            : "iALU",
    "POPC"              : "iALU",
    "SHF"               : "iALU",
    "SHL"               : "iALU",
    "SHR"               : "iALU",
    "VABSDIFF"          : "iALU",
    "VABSDIFF4"         : "iALU",
    "CCTL"              : "iALU",
    # Single-Precision Floating Instructions
    "FADD"              : "fALU",
    "FADD32I"           : "fALU",
    "FCHK"              : "fALU",
    "FFMA32I"           : "fALU",
    "FFMA"              : "fALU",
    "FMNMX"             : "fALU",
    "FMUL"              : "fALU",
    "FMUL32I"           : "fALU",
    "FSEL"              : "fALU",
    "FSET"              : "fALU",
    "FSETP"             : "fALU",
    "FSWZADD"           : "fALU",
    # Half-Precision Floating Instructions
    "HADD2"             : "hALU",
    "HADD2_32I"         : "hALU",
    "HFMA2"             : "hALU",
    "HFMA2_32I"         : "hALU",
    "HMUL2"             : "hALU",
    "HMUL2_32I"         : "hALU",
    "HSET2"             : "hALU",
    "HSETP2"            : "hALU",
    # Double-Precision Floating Instructions
    "DADD"              : "dALU",
    "DFMA"              : "dALU",
    "DMUL"              : "dALU",
    "DSETP"             : "dALU",
    # SFU Special Instructions
    "MUFU"              : "SFU",
    #Tensor Core
    "IMMA"              : "iTCU",
    "HMMA"              : "hTCU",
    # Conversion Instructions
    "F2F"               : "iALU",
    "F2I"               : "iALU",
    "I2F"               : "iALU",
    "I2I"               : "iALU",
    "I2IP"              : "iALU",
    "FRND"              : "iALU",
    # Movement Instructions
    "MOV"               : "iALU",
    "MOV32I"            : "iALU",
    "PRMT"              : "iALU",
    "SEL"               : "iALU",
    "SGXT"              : "iALU",
    "SHFL"              : "iALU",
    # Predicate Instructions
    "PLOP3"             : "iALU",
    "PSETP"             : "iALU",
    "P2R"               : "iALU",
    "R2P"               : "iALU",
    # Control Instructions
    "BMOV"              : "BRA",
    "BPT"               : "BRA",
    "BRA"               : "BRA",
    "BREAK"             : "BRA",
    "BRX"               : "BRA",
    "BSSY"              : "BRA",
    "BSYNC"             : "BRA",
    "CALL"              : "BRA",
    "EXIT"              : "BRA",
    "JMP"               : "BRA",
    "JMX"               : "BRA",
    "KILL"              : "BRA",
    "NANOSLEEP"         : "BRA",
    "RET"               : "BRA",
    "RPCMOV"            : "BRA",
    "RTT"               : "BRA",
    "WARPSYNC"          : "BRA",
    "YIELD"             : "BRA",
    # Miscellaneous Instructions
    "B2R"               : "iALU",
    "BAR"               : "iALU",
    "CS2R"              : "iALU",
    "DEPBAR"            : "iALU",
    "GETLMEMBASE"       : "iALU",
    "LEPC"              : "iALU",
    "NOP"               : "iALU",
    "PMTRIG"            : "iALU",
    "R2B"               : "iALU",
    "S2R"               : "iALU",
    "SETCTAID"          : "iALU",
    "SETLMEMBASE"       : "iALU",
    "VOTE"              : "iALU"
}


ptx_isa = { # ---> (ptx v.72)

    # Integer Instructions
    "add"               : "iALU",
    "sub"               : "iALU",
    "mul"               : "iALU",
    "mad"               : "iALU",
    "mul24lo"           : ["iALU", 8],
    "mul24hi"           : ["iALU", 15],
    "mad24lo"           : ["iALU", 8],
    "mad24hi"           : ["iALU", 15],
    "sad"               : "iALU",
    "mad"               : "iALU",
    "div"               : ["iALU", 71],
    "rem"               : ["iALU", 71],
    "abs"               : "iALU",
    "neg"               : "iALU",
    "min"               : "iALU",
    "max"               : "iALU",
    "popc"              : "iALU",
    "clz"               : ["iALU", 8],
    "bfind"             : "iALU",
    "fns"               : "iALU",
    "brev"              : ["iALU", 8],
    "bfe"               : ["iALU", 125],
    "bfi"               : ["iALU", 125],
    "dp4a"              : "iALU",
    "dp2a"              : "iALU",
    "ret"               : "iALU",
    "exit"              : "iALU",
    "bar"               : "iALU",
    # Logic and Shift Instructions
    "and"               : "iALU",
    "or"                : "iALU",
    "not"               : "iALU",
    "xor"               : "iALU",
    "cnot"              : ["iALU", 8],
    "lop3"              : "iALU",
    "shf"               : "iALU",
    "shl"               : "iALU",
    "shr"               : "iALU",
    # Extended-Precision Integer Instructions
    "addc"              : "iALU",
    "sub.cc"            : "iALU",
    "subc"              : "iALU",
    "mad.cc"            : "iALU",
    "madc"              : "iALU",
    # Single-Precision Floating Instructions
    "ftestp"            : "fALU",
    "fcopysign"         : "fALU",
    "fadd"              : "fALU",
    "fsub"              : "fALU",
    "fmul"              : "fALU",
    "ffma"              : "fALU",
    "fmad"              : "fALU",
    "fdiv"              : ["fALU", 335],
    "fabs"              : "fALU",
    "fneg"              : "fALU",
    "fmin"              : "fALU",
    "fmax"              : "fALU",
    "frcp"              : ["SFU", 60],
    "Fastfrcp"          : ["SFU", 23],
    "fsqrt"             : ["SFU", 60],
    "Fastfsqrt"         : ["SFU", 18],
    "frsqrt"            : ["SFU", 60],
    "Fastfrsqrt"        : ["SFU", 18],
    "fsin"              : ["SFU", 8],
    "fcos"              : ["SFU", 8],
    "fex2"              : ["SFU", 16],
    "flg2"              : ["SFU", 16],
    "ftanh"             : ["SFU", 8],
    # Half-Precision Floating Instructions
    "hfadd"             : "hALU",
    "hfsub"             : "hALU", 
    "hfmul"             : "hALU",
    "hfma"              : "hALU",  
    "hneg"              : "hALU",
    "habs"              : "hALU",  
    "hmin"              : "hALU", 
    "hmax"              : "hALU", 
    "htanh"             : "hALU", 
    "hex2"              : "hALU",
    # Double-Precision Floating Instructions
    "dadd"              : "dALU",
    "dsub"              : "dALU",
    "dmul"              : "dALU",
    "dmad"              : "dALU",
    "dfma"              : "dALU",
    "dabs"              : "dALU",
    "dneg"              : "dALU",
    "dmin"              : "dALU",
    "dmax"              : "dALU",
    "dmax"              : "dALU",
    "ddiv"              : ["dALU", 506],
    "Fastddiv"          : ["dALU", 159],
    "drcp"              : ["dALU", 114],
    "dsqrt"             : ["dALU", 106],
    "drsqrt"            : ["dALU", 106],
    # Tensor Core
    "wmma"              : "iTCU",
    "hwmma"             : "hTCU",
    # Conversion & Movement Instructions
    "mov"               : "iALU",
    "shfl"              : "iALU",
    "prmt"              : "iALU",
    "cvta"              : "iALU",
    "cvt"               : "iALU",
    # Comparision & Selection Instructions
    "set"               : "iALU",
    "setp"              : "iALU",
    "selp"              : "iALU",
    # Control Instructions
    "bra"               : "BRA",
    "call"              : "BRA",
    
}