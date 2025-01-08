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
unset_env
benchmarks="rodinia-3.1-full|polybench-full|GPU_Microbenchmark|deepbench|Tango|pannotia"
#filter_app="rodinia-3.1-full|polybench-full|Tango|pannotia"
filter_app=$benchmarks

gpu="titanv"
# gpu="gtx1080ti"
cuda_version="11.0"
GPU=1
trace_dir_base=/staff/fyyuan/hw_trace02
ppt_gpu_version="PPT-GPU"

run_name="paper"

# change part
GPU_PROFILE="TITANV"
model='ppt2'
time_out=7200

ppt_src='/staff/fyyuan/repo/PPT-GPU/ppt.py'
model_extra_params=''
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

ppt_ori_ampere(){
default
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="cl32"
time_out=7200
GPU_PROFILE="A100-40G"
}

ppt_ori_ampere_fix_cycle(){
default
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="fix_cycle"
time_out=7200
GPU_PROFILE="A100-40G"
model_extra_params="<--fix-cycle>"
}

ppt_ori_fix_cycle(){
default
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="fix_cycle"
time_out=7200
model_extra_params="<--fix-cycle>"
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
GPU_PROFILE="A100-40G"
cuda_version="11.0"
run_name="paper"
}

ppt2_ampere2(){
	default
	trace_dir_base=/staff/fyyuan/hw_trace02
	gpu="a100-40g"
	GPU_PROFILE="A100-40G"
	cuda_version="11.0"
	run_name="paper"
	source env.sh
}

# Trace/profile manual
trace_all(){
benchmarks="rodinia-3.1-full|polybench-full|GPU_Microbenchmark|deepbench|Tango|pannotia"
filter_app=pannotia
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

parallel_settings=(
ppt2
ppt2_old_memory
ppt2_ampere
ppt2_ampere2
)
sequential_settings=(
ppt_ori_cl32
ppt_ori_ampere
ppt_ori_fix_cycle
# ppt_ori_ampere_fix_cycle
)

# 定义最大并行任务数
MAX_PARALLEL=4

# 当前运行任务数
current_jobs=0

log_file=run_memory_paper_all_jobs.log

# 控制并行任务队列
run_in_parallel() {
    local setting_name=$1

    # 检查当前任务数是否达到最大并行限制
    while [ "$current_jobs" -ge "$MAX_PARALLEL" ]; do
        wait -n  # 等待至少一个后台任务完成
        current_jobs=$((current_jobs - 1))  # 任务完成，减少计数
    done

    # 启动新任务
    execute_with_settings "$setting_name" &
    current_jobs=$((current_jobs + 1))
}

# 顺序执行（非并行任务）
run_in_sequence() {
    local setting_name=$1
    execute_with_settings "$setting_name"
}

# Run
run_parallel(){
# 先运行并行任务
for setting in "${parallel_settings[@]}"; do
    run_in_parallel "$setting"
done
echo "All tasks completed."
}

run_sequence(){
# 再运行非并行任务
for setting in "${sequential_settings[@]}"; do
    run_in_sequence "$setting"
done
echo "All tasks completed."
}