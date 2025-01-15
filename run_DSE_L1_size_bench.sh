#!/bin/bash
trace_dir=/staff/fyyuan/hw_trace02/ppt-gpu-titanv/11.0
GPU_PROFILE=TITANV
MAX_PARALLEL=4
granularity=2

benchmarks="rodinia-3.1-full|polybench-full|pannotia|Tango"
filter_app=$benchmarks
# filter_app="rodinia-3.1-full:polybench-2mm"

# DSE param
DSE_param=l1_cache_size
DSE_prefix=/staff/fyyuan/repo/PPT-GPU/tmp_output/DSE_SM/L1_size_
extra_params='--no-overwrite --no_adaptive_cache'

# SM num
param_list=(16384 32768 49152 98304 131072 196608 262144)
# param_list=(131072 196608 262144)

app_and_arg=b+tree-rodinia-3.1/file___data_mil_txt_command___data_command_txt
kernel_id=1

export PYTHONPATH=/staff/fyyuan/repo/PPT-GPU:$PYTHONPATH


app=${trace_dir}/${app_and_arg}

run(){
current_jobs=0
for x in "${param_list[@]}"; do

# 检查当前任务数是否达到最大并行限制
while [ "$current_jobs" -ge "$MAX_PARALLEL" ]; do
    wait -n  # 等待至少一个后台任务完成
    current_jobs=$((current_jobs - 1))  # 任务完成，减少计数
done

echo $(date +%Y%m%d-%H:%M:%S) ${DSE_param}:${x}

# extra_params_ppt="<${extra_params}|--kernel|${kernel_id}|--set-gpu-params|num_LDS_units_per_SM:16,${DSE_param}:${x}>"
extra_params_ppt="${extra_params} --kernel ${kernel_id} --set-gpu-params num_LDS_units_per_SM:16,${DSE_param}:${x}"

echo python /staff/fyyuan/repo/PPT-GPU/scripts/run_simulation2.py -M mpiexec -F $filter_app -B $benchmarks -T /staff/fyyuan/hw_trace02/ppt-gpu-titanv/11.0 -H ${GPU_PROFILE} --granularity ${granularity} -R ${DSE_prefix}${x} --time-out 2000 --extra-params $extra_params_ppt

python /staff/fyyuan/repo/PPT-GPU/scripts/run_simulation2.py -M mpiexec -F $filter_app -B $benchmarks -T /staff/fyyuan/hw_trace02/ppt-gpu-titanv/11.0 -H ${GPU_PROFILE} --granularity ${granularity} -R ${DSE_prefix}${x} --time-out 2000 --extra-params $extra_params_ppt &

current_jobs=$((current_jobs + 1))
done
}

get_data(){
echo python paper_scripts/get_dse_res2.py -F $filter_app -B $benchmarks -N ${DSE_param} --param_list ${param_list[@]} -P ${DSE_prefix} --app_and_arg ${app_and_arg} --kernel_id ${kernel_id}
python paper_scripts/get_dse_res2.py -F $filter_app -B $benchmarks -N ${DSE_param} --param_list ${param_list[@]} -P ${DSE_prefix} --app_and_arg ${app_and_arg} --kernel_id ${kernel_id}
}

# run
wait
get_data
echo "All job done"
