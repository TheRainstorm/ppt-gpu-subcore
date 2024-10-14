my_home=/staff/fyyuan/
ppt_gpu_dir=$my_home/repo/PPT-GPU
cuda_version=11.0
benchmarks=rodinia-2.0-ft,rodinia-3.1,ubench
filter_app="rodinia-2.0-ft|rodinia-3.1|ubench"
# filter_app="rodinia-2.0-ft|rodinia-3.1"
# filter_app="ubench"

model=ppt-gpu
gpu=titanv
run_name=dev

export CUDA_VERSION=$cuda_version
export GPUAPPS_ROOT=$my_home/repo/accel-sim-framework/gpu-app-collection
export UBENCH_ROOT=$my_home/repo/GPU_Microbenchmark
export apps_yaml=${ppt_gpu_dir}/scripts/apps/define-all-apps.yml

# trace_dir=$my_home/repo/accel-sim-framework/hw_run/traces/${model}-${gpu}/${cuda_version}
trace_dir=/extra/hw_trace/${model}-${gpu}/${cuda_version}

# run hw
res_hw_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}.json
res_hw_nvprof_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_nvprof.json
res_hw_ncu_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_ncu.json
res_hw_ncu_cpi_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_ncu_cpi.json
res_hw_cpi_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_cpi.json
# run sim
report_dir=${ppt_gpu_dir}/tmp_output/${model}_${gpu}_${run_name}
res_sim_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${run_name}.json
res_sim_cpi_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${run_name}_cpi.json
res_sim_detail_cpi_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${run_name}_detail_cpi.json
res_sim_sched_cpi_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${run_name}_sched_cpi.json
# draw
draw_output=${ppt_gpu_dir}/tmp_draw/draw_${model}_${gpu}_${run_name}


## run single app
# single_app=rodinia-2.0-ft
# single_app=rodinia-2.0-ft:lud-rodinia-2.0-ft:0
# single_app=rodinia-3.1:hotspot-rodinia-3.1:1
# single_app=rodinia-3.1:bfs-rodinia-3.1:2
single_app=rodinia-3.1:particlefilter_naive-rodinia-3.1
single_app="rodinia-3.1:b+tree-rodinia-3.1|rodinia-3.1:hotspot-rodinia-3.1:0"
# single_app=rodinia-3.1:hotspot-rodinia-3.1:0
single_app=rodinia-3.1:gaussian-rodinia-3.1:0
single_app=rodinia-3.1:particlefilter_naive-rodinia-3.1

# keep model report and draw output in the same dir
single_report_dir=${report_dir}
single_draw_output=${single_report_dir}

. $ppt_gpu_dir/run_helper.sh

