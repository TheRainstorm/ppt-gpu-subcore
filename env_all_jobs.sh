
execute_with_settings() {
    local setting_name=$1
    date=$(date '+%Y-%m-%d-%H-%M-%S')
    if [ -z "$setting_name" ]; then
        echo "$date: Running with setting: $setting_name"
    else
        echo "$date: Running with setting: $setting_name" | tee -a $log_file
    fi

    # 动态调用设置函数
    $setting_name

    # Source 环境
    source env.sh

    # 调用 run_memory
    run_memory

}

default(){
benchmarks="rodinia-3.1-full|polybench-full|GPU_Microbenchmark|deepbench|Tango|pannotia"
filter_app="rodinia-3.1-full|polybench-full|Tango|pannotia"

gpu="titanv"
# gpu="gtx1080ti"
cuda_version="11.0"
GPU=0
trace_dir_base=/staff/fyyuan/hw_trace01
ppt_gpu_version="PPT-GPU-memory"

run_name="paper3"
run_name="paper4"
run_name="paper5"
run_name="paper6"

# change part
GPU_PROFILE="TITANV"
memory_suffix="_default"
granularity=2
memory_model="simulator"
filter_l2="--filter-l2"
use_sm_trace=" "
extra_params=""
}

base(){
default
memory_suffix="_base"
}

ampere_base(){
default
GPU_PROFILE="A100-40G"
memory_suffix="_base"
}

nofilterl2(){
default
memory_suffix="_nofilterl2"
filter_l2=" "
}

no_adaptive(){
default
memory_suffix="_no_adaptive"
extra_params=" --no-adaptive-cache"
}

no_filter_and_adaptive(){
default
memory_suffix="_no_filter_and_adaptive"
filter_l2=" "
extra_params=" --no-adaptive-cache"
}

# sector lab
sector_l1_CL32(){
default
memory_suffix="_sector_l1_CL32"
memory_extra_params="-C l1::32::,l2::::"
extra_params=${memory_extra_params}
}

sector_l2_CL32(){
default
memory_suffix="_sector_l2_CL32"
memory_extra_params="-C l1::::,l2::32::"
extra_params=${memory_extra_params}
}

sector_l1_CL128(){
default
memory_suffix="_sector_l1_CL128"
memory_extra_params="-C l1::128::128,l2::::"
extra_params=${memory_extra_params}
}

sector_l2_CL128(){
default
memory_suffix="_sector_l2_CL128"
memory_extra_params="-C l1::::,l2::128::128"
extra_params=${memory_extra_params}
}

l2_64CL_64S(){
default
memory_suffix="_l2_64CL_64S"
memory_extra_params="-C l1::::,l2::64::64"
extra_params=${memory_extra_params}
}

l2_64CL_32S(){
default
memory_suffix="_l2_64CL_32S"
memory_extra_params="-C l1::::,l2::64::32"
extra_params=${memory_extra_params}
}

l2_16A(){
default
memory_suffix="_l2_16A"
memory_extra_params="-C l1::::,l2:::16:"
extra_params=${memory_extra_params}
}

l2_64A(){
default
memory_suffix="_l2_64A"
memory_extra_params="-C l1::::,l2:::64:"
extra_params=${memory_extra_params}
}

# ppt-gpu
ppt_gpu_base_CL128(){
default
memory_model="ppt-gpu"
memory_suffix="_base_CL128"
memory_extra_params="-C l1::128::,l2::128::"
extra_params="${memory_extra_params} --no-adaptive-cache"
}

ppt_gpu_base_CL32(){
default
memory_model="ppt-gpu"
memory_suffix="_base_CL32"
memory_extra_params="-C l1::32::,l2::32::"
extra_params="${memory_extra_params} --no-adaptive-cache"
}

ppt_gpu_ampere_base_CL32(){
default
GPU_PROFILE="A100-40G"
memory_model="ppt-gpu"
memory_suffix="_base_CL32"
memory_extra_params="-C l1::32::,l2::32::"
extra_params="${memory_extra_params} --no-adaptive-cache"
}

# sdcmL1
sdcmL1_base(){
default
memory_model="sdcmL1"
memory_suffix="_base"
}

sdcmL1_ampere_base(){
default
GPU_PROFILE="A100-40G"
memory_model="sdcmL1"
memory_suffix="_base"
}

# sdcm
sdcm_base(){
default
memory_model="sdcm"
memory_suffix="_base"
}

sdcm_ampere_base(){
default
GPU_PROFILE="A100-40G"
memory_model="sdcm"
memory_suffix="_base"
}

no_flush(){
default
memory_suffix="_no_flush_l2"
extra_params=" --no-flush-l2"
}

fix_l2_write_hit_rate(){
default
memory_suffix="_fix_l2_write_hit_rate"
}

l2_write_miss_read_dram(){
default
memory_suffix="_l2_write_miss_read_dram"
}

# ampere trace
ampere_trace(){
default
gpu="a100-40g"
GPU_PROFILE="A100-40G"
memory_suffix="_base"
}

ppt_gpu_ampere_trace(){
default
gpu="a100-40g"
GPU_PROFILE="A100-40G"
memory_model="ppt-gpu"
memory_suffix="_base_CL32"
memory_extra_params="-C l1::32::,l2::32::"
extra_params="${memory_extra_params} --no-adaptive-cache"
}


# ampere search
ampere_l1_16A(){
default
GPU_PROFILE="A100-40G"
memory_suffix="_ampere_l1_16A"
memory_extra_params="-C l1:::16:,l2::::"
extra_params=${memory_extra_params}
}

ampere_l1_64A(){
default
GPU_PROFILE="A100-40G"
memory_suffix="_ampere_l1_64A"
memory_extra_params="-C l1:::64:,l2::::"
extra_params=${memory_extra_params}
}

ampere_l2_16A(){
default
GPU_PROFILE="A100-40G"
memory_suffix="_ampere_l2_16A"
memory_extra_params="-C l1::::,l2:::16:"
extra_params=${memory_extra_params}
}

ampere_l2_64A(){
default
GPU_PROFILE="A100-40G"
memory_suffix="_ampere_l2_64A"
memory_extra_params="-C l1::::,l2:::64:"
extra_params=${memory_extra_params}
}


# # 遍历并执行每个设置
# for setting in "${settings[@]}"; do
#     execute_with_settings $setting
# done


# 设置任务分类
parallel_settings=(
# simulator
base
ampere_base
nofilterl2
no_adaptive
sector_l1_CL32
sector_l2_CL32

# sdcm
sdcmL1_base
sdcmL1_ampere_base
sdcm_base
sdcm_ampere_base

# # simulator bonus
# sector_l1_CL128
# sector_l2_CL128
# l2_64CL_64S
# l2_64CL_32S
# l2_16A
# l2_64A
)

sequential_settings=(
# ppt-gpu
ppt_gpu_base_CL128
ppt_gpu_base_CL32
ppt_gpu_ampere_base_CL32
)

# # ampere all
# parallel_settings=(
# ampere_base
# sdcmL1_ampere_base
# sdcm_ampere_base
# ppt_gpu_ampere_base_CL32
# )
# sequential_settings=()



# # ampere search
# parallel_settings=(
# ampere_l1_16A
# ampere_l2_16A
# ampere_l1_64A
# ampere_l2_64A
# )
# sequential_settings=()