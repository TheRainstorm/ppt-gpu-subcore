default(){
ppt_gpu_dir=/staff/fyyuan/repo/PPT-GPU
cuda_version=11.0

benchmarks=ubench
filter_app=ubench
loop=3
model=ppt-gpu

# 修改部分
gpu=titanv
GPU=1
run_name=kernel_lat
profile_type=ncu-cycle

extra_params=" --no-overwrite"
}

titanv(){
default
gpu="titanv"
GPU=1
loop=10
}

ampere(){
default
gpu="a100-40g"
GPU=0
loop=1
extra_params=""
}

titanv_ncu_full(){
default
gpu="titanv"
GPU=1
loop=10
profile_type=ncu-full
}

titanv_ncu_rep(){
default
gpu="titanv"
GPU=2
loop=3
profile_type=ncu-rep
extra_params=" --no-overwrite --ncu-rep-dir /staff/fyyuan/ncu-rep/titanv-kernel-lat"
}

# ampere
titanv
# titanv_ncu_full
# titanv_ncu_rep

export CUDA_VERSION=$cuda_version
export GPUAPPS_ROOT=/staff/fyyuan/repo/accel-sim-framework/gpu-app-collection
export UBENCH_ROOT=/staff/fyyuan/repo/GPU_Microbenchmark
# export apps_yaml=${ppt_gpu_dir}/scripts/apps/define-all-apps.yml
export apps_yaml=${ppt_gpu_dir}/scripts/apps/kernel_lat.yml

trace_dir=/staff/fyyuan/hw_trace02/kernel_lat_${gpu}/${cuda_version}
res_hw_ncu_json=${ppt_gpu_dir}/scripts/kernel_lat/tmp/res_hw_${gpu}_${profile_type}_${loop}.json

run(){
# python ${ppt_gpu_dir}/scripts/run_hw_profling.py -B ${benchmarks} -T ${trace_dir} --select ncu --loop_cnt 3
# rm ${res_hw_ncu_json} # clean old apps
# python ${ppt_gpu_dir}/scripts/get_stat_hw.py -B ${benchmarks} -T ${trace_dir} --select ncu -o ${res_hw_ncu_json} --loop 3

# python ${ppt_gpu_dir}/scripts/run_hw_profling.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} -D ${GPU} --select ncu-full --loop-cnt ${loop}
python ${ppt_gpu_dir}/scripts/run_hw_profling.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} -D ${GPU} --select ${profile_type} --loop-cnt ${loop} $(echo ${extra_params})
# python ${ppt_gpu_dir}/scripts/run_hw_profling.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} -D ${GPU} --select ncu-rep --loop-cnt ${loop} --ncu-rep-dir /staff/fyyuan/ncu-rep/a100-40g-kernel-lat

rm ${res_hw_ncu_json} # clean old apps
python ${ppt_gpu_dir}/scripts/get_stat_hw.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} --select ${profile_type} -o ${res_hw_ncu_json} --loop-cnt ${loop}
# python ${ppt_gpu_dir}/scripts/convert_hw_metrics.py -i ${res_hw_ncu_json} -o ${res_hw_json}
python ${ppt_gpu_dir}/scripts/kernel_lat/tocsv.py -i ${res_hw_ncu_json} -o kernel_lat_${gpu}_${profile_type}_${loop}.xlsx
}

echo "app cuda_version: $CUDA_VERSION"
echo "HW GPU: $gpu [${GPU}]"
echo "trace_dir: $trace_dir"

echo "benchmarks: $benchmarks"
echo "filter_app: $filter_app"
echo "profile_type: $profile_type"