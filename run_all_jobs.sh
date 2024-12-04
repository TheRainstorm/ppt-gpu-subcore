#!/bin/bash

source env_all_jobs.sh

# 定义最大并行任务数
MAX_PARALLEL=8

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


# 在两台机器上分别跑，覆盖掉原本配置
# parallel_settings=()
# sequential_settings=()

# 先运行并行任务
for setting in "${parallel_settings[@]}"; do
    run_in_parallel "$setting"
done
# 等待所有并行任务完成
wait

# 再运行非并行任务
for setting in "${sequential_settings[@]}"; do
    run_in_sequence "$setting"
done

echo "All tasks completed."