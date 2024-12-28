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
    run 2
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
}

ppt_ori(){
default
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="paper"
}
