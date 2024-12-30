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

    # 模拟
    run 3
}

default(){
benchmarks="rodinia-3.1-full|polybench-full|GPU_Microbenchmark|deepbench|Tango|pannotia"
filter_app="rodinia-3.1-full|polybench-full|Tango|pannotia"

gpu="titanv"
# gpu="gtx1080ti"
cuda_version="11.0"
GPU=1
trace_dir_base=/staff/fyyuan/hw_trace01
ppt_gpu_version="PPT-GPU"

run_name="paper"

# change part
GPU_PROFILE="TITANV"
model='ppt2'
time_out=7200
}

# manual
ppt2_single(){
default
filter_app="Tango:AN"
source env.sh
}


ppt_ori_cl32(){
default
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="cl32"
time_out=7200
}

ppt2(){
default
run_name="paper"
}

ppt2_old_memory(){
default
run_name="old_memory"
model_extra_params="-C l1::32::,l2::32:: --memory-model ppt-gpu"
# source env.sh
}

# Ampere
ppt2_ampere(){
default
gpu="ampere"
cuda_version="11.0"
run_name="paper"
}

# Trace/profile manual
trace_all(){
benchmarks="rodinia-3.1-full|polybench-full|GPU_Microbenchmark|deepbench|Tango"
filter_app=$benchmarks
GPU=1
trace_dir_base=/staff/fyyuan/hw_trace02
time_out=7200
source env.sh
}

trace(){
filter_app="rodinia-3.1-full:gaussian-rodinia-3.1:2"
GPU=1
time_out=7200
source env.sh
}
