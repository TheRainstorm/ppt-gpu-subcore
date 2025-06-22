# `BASH_SOURCE` works for bash, `-${(%):-%N}}` works for zsh
ppt_gpu_dir="$(cd "$(dirname "${BASH_SOURCE:-${(%):-%N}}")"/../ && pwd)"

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

    derived_vars
    . $ppt_gpu_dir/quick_start/run_helper.sh

    print_summary
    # 模拟
    if [ $2 -eq 0 ]; then
        echo "no run"
    else
        echo "run"
        run 3  # run simulation and draw
    fi
}

derived_vars(){
gpu_sim=$(echo $GPU_PROFILE | tr 'A-Z' 'a-z')  # 转换成小写：因为 tracing/profiling 使用的是小写的 GPU 名称，模拟使用的是大写的 GPU 名称，要使二者涉及的文件名兼容

# Trace/Profiling
res_hw_ncu_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_ncu.json  # ncu 原始性能数据
res_hw_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}.json          # 需要的硬件数据（以 nvprof 变量名为基准）
res_hw_cpi_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_cpi.json  # 硬件 CPI 硬件数据
# Simulation
report_dir=${ppt_gpu_dir}/tmp_output/${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}  # 模拟结果存放目录
res_sim_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}.json  # 模拟结果
res_sim_cpi_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}_cpi.json # 模拟 CPI 结果
res_hw_sim_json=${ppt_gpu_dir}/tmp/res_hw_${gpu_sim}_${cuda_version}.json  # 模拟对应的硬件数据
res_hw_cpi_sim_json=${ppt_gpu_dir}/tmp/res_hw_${gpu_sim}_${cuda_version}_cpi.json # 模拟对应的硬件 CPI 数据

draw_output=${ppt_gpu_dir}/tmp_draw/draw_${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}
}

default(){
trace_dir=${ppt_gpu_dir}/hw_trace_collect/ppt-gpu-${gpu}/${cuda_version}
# export apps_yaml=${ppt_gpu_dir}/scripts/apps/define-all-apps.yml
export apps_yaml=${ppt_gpu_dir}/quick_start/apps_quick_start.yaml
export GPGPU_WORKLOADS_ROOT=/path/to/GPGPUs-Workloads           # ！！！请根据实际情况修改！！！

benchmarks="rodinia-3.1-full"   # benchmark 列表，|分隔
filter_app=$benchmarks          # 进一步选择应用程序列表，仅对选中应用Tracing/Profiling/Simulating。支持层次化名字、正则、列表三种格式，多个匹配条件使用|分隔，详见 scripts/common.py:filter_app_list
                                # 1) suite[:exec[:count]] 2) regex:.*-rodinia-2.0-ft 3）app1,app2,app3

gpu="titanv"            # Tracing/profiling 的 GPU
GPU=0                   # Tracing/profiling 时使用的 GPU 编号
cuda_version="11.0"     # Tracing/profiling 时选择不同 CUDA 版本编译出的应用程序
GPU_PROFILE="TITANV"    # 模拟的 GPU

model='ppt2'            # 模拟模型名称（用于区分原始模型和修改后的模型），本 quick start 无需修改
run_name="paper"        # 区分模拟名称，控制模拟结果的文件名
model_extra_params=""   # 见 ppt.py 支持的额外参数
}

ppt2(){
default
run_name="paper"
}

ppt2_no_subcore(){
default
run_name="paper_no_subcore"
model_extra_params="--set-gpu-params num_LDS_units_per_SM:16,num_inst_dispatch_units_per_SM:4,num_warp_schedulers_per_SM:1"
}

ppt2_FU_no_limit(){
default
run_name="paper_FU_no_limit"
model_extra_params="--set-gpu-params num_LDS_units_per_SM:128,num_INT_units_per_SM:128,num_SP_units_per_SM:128,num_DP_units_per_SM:128,num_SF_units_per_SM:128,num_TC_units_per_SM:128"
}

parallel_settings=(
ppt2
ppt2_no_subcore
ppt2_FU_no_limit
)

sequential_settings=(

)

# 定义最大并行任务数
MAX_PARALLEL=4
log_file=run_jobs.log

# 控制并行任务队列
current_jobs=0 # 当前运行任务数
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