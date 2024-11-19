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
# time_out=10800 # 3h
# time_out=7200 # 2h
time_out=3600 # 1h
# detect gpu
gpu_detect(){
    gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    case "$gpu_model" in
        "NVIDIA GeForce GTX 1080 Ti")
            gpu="gtx1080ti"
            ;;
        "Tesla P100-PCIE-16GB")
            gpu="p100"
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
GPU_PROFILE=${GPU_PROFILE:-TITANV}
gpu_sim=$(echo $GPU_PROFILE | tr 'A-Z' 'a-z')
granularity=${granularity:-2}
filter_l2=${filter_l2:-" "}
use_approx=${use_approx:-" "}
use_sm_trace=${use_sm_trace:-" "}

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
trace_dir_base=${trace_dir_base:-$my_home/hw_trace01}
ppt_gpu_version=${ppt_gpu_version:-"PPT-GPU"}
ppt_gpu_dir=$my_home/repo/${ppt_gpu_version}

# cuda_version_major=`nvcc --version | grep release | sed -re 's/.*release ([0-9]+)\..*/\1/'`;
export GPUAPPS_ROOT=$my_home/repo/accel-sim-framework/gpu-app-collection
export UBENCH_ROOT=$my_home/repo/GPU_Microbenchmark
export GPGPU_WORKLOADS_ROOT=$my_home/repo/GPGPUs-Workloads
export apps_yaml=${ppt_gpu_dir}/scripts/apps/define-all-apps.yml

trace_dir=${trace_dir_base}/${model}-${gpu}/${cuda_version}

# run hw
res_hw_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}.json
res_hw_cpi_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_cpi.json
# e.g volta trace(and profile) to simulate Ampere GPU Profile
res_hw_sim_json=${ppt_gpu_dir}/tmp/res_hw_${gpu_sim}_${cuda_version}.json
res_hw_cpi_sim_json=${ppt_gpu_dir}/tmp/res_hw_${gpu_sim}_${cuda_version}_cpi.json
# raw profile data
res_hw_nvprof_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_nvprof.json
res_hw_ncu_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_ncu.json
# raw stall related data
res_hw_nvprof_cpi_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_nvprof_cpi.json
res_hw_ncu_cpi_json=${ppt_gpu_dir}/tmp/res_hw_${gpu}_${cuda_version}_ncu_cpi.json

# run sim
report_dir=${ppt_gpu_dir}/tmp_output/${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}
res_sim_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}.json
res_sim_lite_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}_lite.json
# from full sim res, get cpi, detail cpi, sched cpi result
res_sim_cpi_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}_cpi.json
res_sim_detail_cpi_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}_detail_cpi.json
res_sim_sched_cpi_json=${ppt_gpu_dir}/tmp/res_${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}_sched_cpi.json

# run memory
memory_model=${memory_model:-"ppt-gpu"} # sdcm, simulator
memory_suffix=${memory_suffix:-""}  # distinguish different run with the same memory model
res_memory_json=${ppt_gpu_dir}/tmp/res_memory_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}_${memory_model}${memory_suffix}.json

# draw
draw_output=${ppt_gpu_dir}/tmp_draw/draw_${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}

## run single app
single_app=${single_app:-"rodinia-3.1:backprop-rodinia-3.1"}
# keep model report and draw output in the same dir
single_report_dir=${report_dir}
single_draw_output=${single_report_dir}

. $ppt_gpu_dir/run_helper.sh

log_file=run_helper.log
print_summary(){
    date '+%Y-%m-%d %H:%M:%S'
    echo "Summary:\n"
    echo "[Tracing]:"
    echo "app cuda_version: $CUDA_VERSION"
    echo "nvcc cuda_version: $curr_cuda_version"
    echo "HW GPU: $gpu [${GPU}]"
    echo "trace_dir: $trace_dir"
    echo ""

    echo "[Simulation]:"
    echo "GPU Profile: $GPU_PROFILE"
    echo "run_name: $run_name"
    echo "apps_yaml: $apps_yaml"
    echo "benchmarks: $benchmarks"
    echo "filter_app: $filter_app"
    echo ""

    echo "[Files]:"
    echo "res_hw_json: $res_hw_json"
    echo "res_hw_sim_json: $res_hw_sim_json"
    echo "res_sim_json: $res_sim_json"
    echo "report_dir: $report_dir"
    echo "draw_output: $draw_output"
    echo "res_memory_json: $res_memory_json"
    echo ""

    echo "[Memory]:"
    echo "run_name: $run_name"
    echo "memory_model: $memory_model"
    echo "memory_suffix: $memory_suffix"
    echo "granularity: $granularity, filter_l2: $filter_l2, use_approx: $use_approx, use_sm_trace: $use_sm_trace"
    echo "memory_extra_params: $memory_extra_params"
}

unset_env(){
    unset benchmarks filter_app cuda_version
    unset run_name nvbit_version 
    unset GPU gpu
    unset trace_dir_base
    unset ppt_gpu_version
    unset memory_model
    unset GPU_PROFILE
    unset granularity filter_l2 use_approx use_sm_trace memory_suffix
}

print_summary