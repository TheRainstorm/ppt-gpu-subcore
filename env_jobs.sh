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
    if [ $2 -eq 0 ]; then
        echo "no run"
    else
        echo "run"
        run 3
    fi
}

default(){
# unset_env
benchmarks="rodinia-3.1-full|polybench-full|pannotia|Tango|GPU_Microbenchmark|deepbench"
#filter_app="rodinia-3.1-full|polybench-full|Tango|pannotia"
filter_app=$benchmarks

gpu="titanv"
# gpu="gtx1080ti"
cuda_version="11.0"
GPU=1
trace_dir_base=/staff/fyyuan/hw_trace02
ppt_gpu_version="PPT-GPU"
use_ncu=1
profile_cpi=0
hw_prof_type=ncu-full
loop=1

run_name="paper"

# change part
GPU_PROFILE="TITANV"
model='ppt2'
time_out=7200

ppt_src='/staff/fyyuan/repo/PPT-GPU/ppt.py'
model_extra_params=''
profling_extra_params=""
}

ppt_ori_cl32(){
default
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="cl32"
time_out=7200
}

ppt_ori_fix_cycle(){
default
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="fix_cycle"
time_out=7200
model_extra_params="<--fix-cycle>"
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
gpu="a100-40g"
GPU_PROFILE="A100-40G"
cuda_version="11.0"
run_name="paper"
}

# expriment
ppt2_KL(){
default
run_name="KL"
}
ppt2_ampere_KL(){
default
GPU_PROFILE="A100-40G"
run_name="KL"
}

# cache associaty
ppt2_KL_deepbench(){
default
filter_app="deepbench"
run_name="KL_db2"
}

ppt_ori_db(){
default
filter_app="deepbench"
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="cl32"
time_out=7200
}

# Trace/profile manual
trace_all(){
default
benchmarks="rodinia-3.1-full|polybench-full|GPU_Microbenchmark|pannotia|Tango|deepbench"
# filter_app="GPU_Microbenchmark:LGThrottle|GPU_Microbenchmark:longScoreboard"
# filter_app="pannotia"
filter_app=$benchmarks
GPU=1
trace_dir_base=/staff/fyyuan/hw_trace02
time_out=7200
hw_prof_type=ncu-full
loop=3
profling_extra_params="--no-overwrite"
source env.sh
}

trace_ampere(){
default
benchmarks="rodinia-3.1-full|polybench-full|GPU_Microbenchmark|deepbench|Tango|pannotia"
# filter_app="GPU_Microbenchmark:LGThrottle|GPU_Microbenchmark:longScoreboard"
filter_app=$benchmarks
gpu=a100-40g
GPU_PROFILE="A100-40G"
GPU=1
trace_dir_base=/staff/fyyuan/hw_trace02
time_out=7200
source env.sh
}

sim(){
benchmarks="rodinia-3.1-full|polybench-full|GPU_Microbenchmark|deepbench|Tango|pannotia"
filter_app="GPU_Microbenchmark:LGThrottle|GPU_Microbenchmark:longScoreboard"
# filter_app=$benchmarks
time_out=900
source env.sh
}

# parallel_settings=(
# ppt2
# ppt2_old_memory
# ppt2_ampere
# ppt2_ampere2
# )
parallel_settings=(
ppt2_KL
ppt2_ampere_KL
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