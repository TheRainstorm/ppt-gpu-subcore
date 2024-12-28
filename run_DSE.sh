#!/bin/bash

# TODO 还未完成

source env_all_jobs.sh

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
    memory_extra_params="-C l1:${L1_capacity}:::,l2::::"

    # Source 环境
    source env.sh

    # 调用 run_memory
    run_memory

}
# SM num


# L1 Capacity
L1_capacity_list=(16 32 48 96 128 256)
L1_cacheline_list=(16 32 64 128)
L1_associativity_list=(1 2 4 8 16 32 64 128)

L2_capacity_list=(1 2 4 8 16 32 64 96 128)  # MB
L2_associativity_list=(1 2 4 8 16 32 64 128)

SM_list=(2 4 8 16 32 64 96 128)



for L1_capacity in "${L1_capacity_list[@]}"; do

    default

    memory_extra_params="-C l1:${L1_capacity}:::,l2::::"
    memory_suffix="_DSE_"
    filter_l2=" "
done

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

# 先运行并行任务
for setting in "${parallel_settings[@]}"; do
    run_in_parallel "$setting"
done
