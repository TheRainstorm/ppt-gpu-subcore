#!/bin/bash
trace_dir=/staff/fyyuan/hw_trace02/ppt-gpu-titanv/11.0
GPU_PROFILE=TITANV
MAX_PARALLEL=4
granularity=2
extra_params='--no-overwrite --no_adaptive_cache'

# DSE param
DSE_param=l1_cache_size
DSE_prefix=/staff/fyyuan/repo/PPT-GPU/tmp_output/DSE_SM/L1_size_

pre_set(){
# SM num
param_list=(16384 32768 49152 98304 131072 196608 262144)
# param_list=(16384 32768)

app_and_arg=b+tree-rodinia-3.1/file___data_mil_txt_command___data_command_txt
kernel_id=1
}

post_set(){
app=${trace_dir}/${app_and_arg}
}

app_btree(){
pre_set
app_and_arg=b+tree-rodinia-3.1/file___data_mil_txt_command___data_command_txt
post_set
}

app_btree2(){
pre_set
app_and_arg=b+tree-rodinia-3.1/file___data_mil_txt_command___data_command_txt
kernel_id=2
post_set
}

app_2mm(){
pre_set
app_and_arg=polybench-2mm/NO_ARGS
post_set
}

run(){
current_jobs=0
for x in "${param_list[@]}"; do

# 检查当前任务数是否达到最大并行限制
while [ "$current_jobs" -ge "$MAX_PARALLEL" ]; do
    wait -n  # 等待至少一个后台任务完成
    current_jobs=$((current_jobs - 1))  # 任务完成，减少计数
done

echo $(date +%Y%m%d-%H:%M:%S) ${DSE_param}:${x}

echo mpiexec -n 2 python /staff/fyyuan/repo/PPT-GPU/ppt.py --set-gpu-params num_LDS_units_per_SM:16,${DSE_param}:${x} --granularity ${granularity} ${extra_params} --mpi --app ${app} --sass --config ${GPU_PROFILE} --report-output-dir ${DSE_prefix}${x} --kernel ${kernel_id}
mpiexec -n 2 python /staff/fyyuan/repo/PPT-GPU/ppt.py --set-gpu-params num_LDS_units_per_SM:16,${DSE_param}:${x} --granularity ${granularity} ${extra_params} --mpi --app ${app} --sass --config ${GPU_PROFILE} --report-output-dir ${DSE_prefix}${x} --kernel ${kernel_id} &

current_jobs=$((current_jobs + 1))
done
}

get_data(){

# python paper_scripts/get_dse_res.py -N ${DSE_param} --param_list ${param_list[@]} -P ${DSE_prefix} --app_and_arg ${app_and_arg} --kernel_id ${kernel_id}
python paper_scripts/get_dse_res.py -N ${DSE_param} --param_list ${param_list[@]} -P ${DSE_prefix} --app_and_arg ${app_and_arg} --kernel_id ${kernel_id}
}

one_dse(){
run
wait
get_data
echo "All job done"
}

app_2mm
one_dse

app_btree
one_dse
