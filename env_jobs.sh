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

    # Source 环境
    source env.sh

    # 模拟
    if [ $2 -eq 0 ]; then
        echo "no run"
    else
        echo "run"
        run 3
    fi
}

default(){
# unset_env
benchmarks="rodinia-3.1-full|polybench-full|pannotia|Tango|micro|GPU_Microbenchmark|deepbench"
#filter_app="rodinia-3.1-full|polybench-full|Tango|pannotia"
benchmarks="rodinia-3.1-full|polybench-full|pannotia|Tango|micro"
filter_app=$benchmarks

gpu="titanv"
# gpu="gtx1080ti"
cuda_version="11.0"
GPU=1
trace_dir_base=/staff/fyyuan/hw_trace02
ppt_gpu_version="PPT-GPU"
use_ncu=1
profile_cpi=0
hw_prof_type=ncu-full
loop=1

run_name="paper"

# change part
GPU_PROFILE="TITANV"
model='ppt2'
time_out=1200
# time_out=7200

ppt_src='/staff/fyyuan/repo/PPT-GPU/ppt.py'
profling_extra_params=""
model_extra_params=''
# filter_app="micro"
# model_extra_params='<--no-overwrite>'
}

ppt_ori(){
default
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="cl32"
}

ppt_ori_fix_cycle(){
default
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="fix_cycle"
model_extra_params="<--fix-cycle>"
}

ppt_ori_ampere(){
default
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="cl32"
GPU_PROFILE="A100-40G"
}

ppt_ori_ampere_fix_cycle(){
default
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="fix_cycle"
GPU_PROFILE="A100-40G"
model_extra_params="<--fix-cycle>"
}

ppt2(){
default
run_name="paper"
}

ppt2_LDS(){
default
run_name="LDS"
model_extra_params="<--set-gpu-params num_LDS_units_per_SM:16>"
}

ppt2_old_memory(){
default
run_name="old_memory"
model_extra_params="-C l1::32::,l2::32:: --memory-model ppt-gpu"
}

ppt2_old_memory_LDS(){
default
run_name="old_memory_LDS"
model_extra_params="-C l1::32::,l2::32:: --memory-model ppt-gpu --set-gpu-params num_LDS_units_per_SM:16"
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

ppt2_no_KLL(){
default
run_name="paper_no_KLL"
model_extra_params="--no_KLL --set-gpu-params num_LDS_units_per_SM:16"
}

# Ampere
ppt2_ampere(){
default
GPU_PROFILE="A100-40G"
cuda_version="11.0"
run_name="paper"
}

ppt2_ampere_LDS(){
default
GPU_PROFILE="A100-40G"
cuda_version="11.0"
run_name="LDS"
model_extra_params="<--set-gpu-params num_LDS_units_per_SM:16>"
}

ppt2_ampere2(){
default
gpu="a100-40g"
GPU_PROFILE="A100-40G"
cuda_version="11.0"
run_name="paper"
}

# # expriment
# ppt2_KL(){
# default
# run_name="KL"
# }
# ppt2_ampere_KL(){
# default
# GPU_PROFILE="A100-40G"
# run_name="KL"
# }

ppt2_SS(){  # super scalar
default
run_name="SS_v2"
time_out=1200
}

ppt2_db(){
default
benchmarks="rodinia-3.1-full|polybench-full|pannotia|Tango|GPU_Microbenchmark|deepbench"
filter_app="deepbench"
run_name="db"
# model_extra_params='<--no-overwrite>'
}
ppt2_ampere_db(){
default
benchmarks="rodinia-3.1-full|polybench-full|pannotia|Tango|GPU_Microbenchmark|deepbench"
filter_app="deepbench"
GPU_PROFILE="A100-40G"
run_name="db"
# model_extra_params='<--no-overwrite>'
}

ppt_ori_db(){
default
filter_app="deepbench"
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="cl32_db"
time_out=7200
# model_extra_params='<--no-overwrite>'
}

# microbench
ppt2_micro(){
default
benchmarks="rodinia-3.1-full|polybench-full|pannotia|Tango|micro|GPU_Microbenchmark|deepbench"
filter_app="micro:LGThrottle|micro:LGThrottleM|micro:ShortScoreboard|micro:longScoreboard"
run_name="micro"
# run_name="micro_LDS"
# model_extra_params='<--AMAT_select const_2000>'
time_out=1200
}
# --AMAT_select: AMAT_ori, AMAT_sum, AMAT_foumula, const_1000
# --scale-opt: ori, float, ceil
# --ipc_select: tot_ipc, sm_ipc, smsp_ipc(用不了)
# --act_cycle_select
# --no-overwrite 

ppt_ori_micro(){
default
benchmarks="rodinia-3.1-full|polybench-full|pannotia|Tango|GPU_Microbenchmark|deepbench|micro"
filter_app="GPU_Microbenchmark:LGThrottle|GPU_Microbenchmark:LGThrottleM|GPU_Microbenchmark:ShortScoreboard|GPU_Microbenchmark:longScoreboard"
filter_app="micro:"
model='ppt-gpu'
ppt_src='/staff/fyyuan/repo/PPT-GPU-ori/ppt.py'
run_name="micro"
time_out=7200
# model_extra_params='<--AMAT_select const_2000>'
}

ppt2_ampere_micro(){
default
benchmarks="rodinia-3.1-full|polybench-full|pannotia|Tango|GPU_Microbenchmark|deepbench"
filter_app="GPU_Microbenchmark:LGThrottle|GPU_Microbenchmark:LGThrottleM|GPU_Microbenchmark:ShortScoreboard|GPU_Microbenchmark:longScoreboard"
GPU=1
hw_prof_type=ncu-rep
GPU_PROFILE="A100-40G"
run_name="micro"
# model_extra_params='<--no-overwrite>'
}
# --ncu-rep-dir /staff/fyyuan/ncu-rep/titanv-kernel-lat

# Trace/profile manual
trace_all(){
default
benchmarks="rodinia-3.1-full|polybench-full|pannotia|Tango|micro|GPU_Microbenchmark|deepbench"
# filter_app="GPU_Microbenchmark:LGThrottle|GPU_Microbenchmark:LGThrottleM|GPU_Microbenchmark:longScoreboard|GPU_Microbenchmark:ShortScoreboard"
# filter_app="pannotia"
# filter_app="GPU_Microbenchmark|deepbench"
filter_app=$benchmarks
# filter_app="GPU_Microbenchmark:ITVAL1|GPU_Microbenchmark:ITVALX_1|GPU_Microbenchmark:ITVAL_dep"
filter_app="micro"
GPU=1
trace_dir_base=/staff/fyyuan/hw_trace02
time_out=7200
hw_prof_type=ncu-full
loop=3
profling_extra_params="--no-overwrite"
source env.sh
}

trace_ampere(){
default
benchmarks="rodinia-3.1-full|polybench-full|pannotia|Tango|micro|GPU_Microbenchmark|deepbench"
# filter_app="GPU_Microbenchmark:LGThrottle|GPU_Microbenchmark:LGThrottleM|GPU_Microbenchmark:longScoreboard|GPU_Microbenchmark:ShortScoreboard"
filter_app=$benchmarks
# filter_app=deepbench:gemm_bench-tensor
filter_app="micro"
gpu=a100-40g
GPU_PROFILE="A100-40G"
GPU=1
hw_prof_type=ncu-full
trace_dir_base=/staff/fyyuan/hw_trace02
time_out=600
profling_extra_params="--no-overwrite"
source env.sh
}

sim(){
benchmarks="rodinia-3.1-full|polybench-full|GPU_Microbenchmark|deepbench|Tango|pannotia"
filter_app="GPU_Microbenchmark:LGThrottle|GPU_Microbenchmark:longScoreboard"
# filter_app=$benchmarks
time_out=900
source env.sh
}

parallel_settings=(
ppt2_LDS
ppt2
ppt2_ampere_LDS
ppt2_ampere
# ppt2_SS
# ppt2_ampere_SS
)
# parallel_settings=(
# ppt2_KL
# ppt2_ampere_KL
# )
sequential_settings=(
ppt_ori_fix_cycle
ppt_ori_ampere_fix_cycle
ppt_ori
ppt_ori_ampere
ppt2_old_memory_LDS
ppt2_old_memory
)

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