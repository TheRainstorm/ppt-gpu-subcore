### Benchmarks Select
benchmarks=${benchmarks:-"rodinia-3.1-full|polybench-full|GPU_Microbenchmark|deepbench|Tango"}
filter_app=${filter_app:-$benchmarks}

### Run model select
model=ppt-gpu
cuda_version=${cuda_version:-"11.0"}
run_name=${run_name:-"dev"}

GPU="${GPU:-0}"
nvbit_version=${nvbit_version:-"1.5.5"}
loop=1  # hw profiling loop count

# detect gpu
gpu_detect(){
    gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    case "$gpu_model" in
        "NVIDIA GeForce GTX 1080 Ti")
            gpu="gtx1080ti"
            ;;
        "NVIDIA TITAN V")
            gpu="titanv"
            ;;
        "NVIDIA A100 80GB PCIe")
            gpu="a100-80g"
            ;;
        "NVIDIA A100-PCIE-40GB")
            gpu="a100-40g"
            ;;
        *)
            gpu="unknown"
            ;;
    esac
    echo $gpu
}
gpu=${gpu:-$(gpu_detect)}

use_ncu=1
profile_cpi=1
if [ "$gpu" = "gtx1080ti" ]; then
    echo "Warning: gtx1080ti does not support ncu, can't profile cpi for now"
    use_ncu=0
fi

# detect cuda version
export CUDA_VERSION=$cuda_version  # used in app yaml
curr_cuda_version=`nvcc --version | grep release | sed -re 's/.*release ([0-9]+\.[0-9]+).*/\1/'`;
if [ -z "$curr_cuda_version" ]; then
    echo "Error: CUDA version not found, nvibit trace need CUDA in PATH"
fi

hw_identifier="${gpu}-$GPU-${CUDA_VERSION}"
sim_identifier="${gpu}-$GPU-${CUDA_VERSION}-${run_name}"

### Set File Path
my_home=/staff/fyyuan
ppt_gpu_dir=$my_home/repo/PPT-GPU
trace_dir_base=${trace_dir_base:-$my_home/hw_trace3}

# cuda_version_major=`nvcc --version | grep release | sed -re 's/.*release ([0-9]+)\..*/\1/'`;
export GPUAPPS_ROOT=$my_home/repo/accel-sim-framework/gpu-app-collection
export UBENCH_ROOT=$my_home/repo/GPU_Microbenchmark
export GPGPU_WORKLOADS_ROOT=$my_home/repo/GPGPUs-Workloads
export apps_yaml=${ppt_gpu_dir}/scripts/apps/define-all-apps.yml

trace_dir=${trace_dir_base}/${model}-${gpu}/${cuda_version}

# run hw
res_hw_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}.json
res_hw_cpi_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_cpi.json
# raw profile data
res_hw_nvprof_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_nvprof.json
res_hw_ncu_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_ncu.json
# raw stall related data
res_hw_nvprof_cpi_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_nvprof_cpi.json
res_hw_ncu_cpi_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_ncu_cpi.json

# run sim
report_dir=${ppt_gpu_dir}/tmp_output/${model}_${gpu}_${cuda_version}_${run_name}
res_sim_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${run_name}.json
res_sim_lite_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${run_name}_lite.json
# from full sim res, get cpi, detail cpi, sched cpi result
res_sim_cpi_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${run_name}_cpi.json
res_sim_detail_cpi_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${run_name}_detail_cpi.json
res_sim_sched_cpi_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${run_name}_sched_cpi.json
# draw
draw_output=${ppt_gpu_dir}/tmp_draw/draw_${model}_${gpu}_${cuda_version}_${run_name}

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
single_app=GPU_Microbenchmark:l2_bw_32f:0
single_app=GPU_Microbenchmark:l1_bw_32f:0

# keep model report and draw output in the same dir
single_report_dir=${report_dir}
single_draw_output=${single_report_dir}

. $ppt_gpu_dir/run_helper.sh

log_file=run_helper.log
print_summary(){
    date '+%Y-%m-%d %H:%M:%S'
    echo "Summary:"
    echo "app cuda_version: $CUDA_VERSION"
    echo "nvcc cuda_version: $curr_cuda_version"
    echo "gpu: $gpu [${GPU}]"
    echo "run_name: $run_name"
    echo "apps_yaml: $apps_yaml"
    echo "benchmarks: $benchmarks"
    echo "filter_app: $filter_app"
    echo "trace_dir: $trace_dir"
    echo "res_hw_json: $res_hw_json"
    echo "res_sim_json: $res_sim_json"
    echo "report_dir: $report_dir"
    echo "draw_output: $draw_output"
}

unset_env(){
    unset benchmarks filter_app cuda_version
    unset run_name nvbit_version 
    unset GPU gpu
    unset trace_dir_base
}

print_summary