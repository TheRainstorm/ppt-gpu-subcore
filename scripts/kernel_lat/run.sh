ppt_gpu_dir=/staff/fyyuan/repo/PPT-GPU
cuda_version=11.0
benchmarks=ubench
single_app=rodinia-2.0-ft
model=ppt-gpu
gpu=titanv
run_name=kernel_lat

export CUDA_VERSION=$cuda_version
export GPUAPPS_ROOT=/staff/fyyuan/repo/accel-sim-framework/gpu-app-collection
export UBENCH_ROOT=/staff/fyyuan/repo/GPU_Microbenchmark
# export apps_yaml=${ppt_gpu_dir}/scripts/apps/define-all-apps.yml
export apps_yaml=${ppt_gpu_dir}/scripts/apps/kernel_lat.yml


trace_dir=/staff/fyyuan/workspace/hw_trace/kernel_lat_${model}-${gpu}/${cuda_version}

# run hw
res_hw_ncu_json=${ppt_gpu_dir}/scripts/kernel_lat/tmp/res_hw_${gpu}_ncu.json

run(){
python ${ppt_gpu_dir}/scripts/run_hw_profling.py -B ${benchmarks} -T ${trace_dir} --select ncu --loop_cnt 1
python ${ppt_gpu_dir}/scripts/get_stat_hw.py -B ${benchmarks} -T ${trace_dir} --select ncu -o ${res_hw_ncu_json} --loop 1
}
