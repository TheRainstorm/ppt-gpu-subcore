#!/usr/bin/env bash
# set -x

# common
accel_sim_dir=/staff/fyyuan/repo/accel-sim-framework
GIMT_dir=/staff/fyyuan/repo/gpu-interval-model-tensor
ppt_gpu_dir=/staff/fyyuan/repo/PPT-GPU
benchmarks=my_benchmark

model=ppt-gpu
gpu=titanv
run_name=my_benchmark

# trace
trace_dir=${accel_sim_dir}/hw_run/traces/PPT-GPU-snode2-nofault/11.0
# run hw
res_hw_json=${GIMT_dir}/tmp/res_hw_${gpu}_${benchmarks}.json
# run sim
report_dir=tmp_output/${model}_${gpu}_${run_name}
res_sim_json=${ppt_gpu_dir}/res_${model}_${gpu}_${run_name}.json
# draw
draw_output=tmp_draw/draw_1_${model}_${gpu}_${run_name}

run_trace(){
    # trace
    cd ${accel_sim_dir}
    python3 ./util/tracer_nvbit/run_hw_trace.py -B ${benchmarks} -D 0 --trace_tool /staff/fyyuan/repo/PPT-GPU/tracing_tool/tracer.so

    # copy
    rsync -av hw_run/traces/device-0/11.0/ ${trace_dir}
}

run_hw(){
    # hw run
    cd ${GIMT_dir}
    python utils/run_model/run_hw.py -Y "${accel_sim_dir}/util/job_launching/apps/define-all-apps.yml" -B ${benchmarks} -T "${trace_dir}" --loop_cnt 3

    python utils/run_model/get_stat_hw.py -Y "${accel_sim_dir}/util/job_launching/apps/define-all-apps.yml" -B ${benchmarks} -T "${trace_dir}" -o ${res_hw_json}
}

run_sim(){
    cd ${ppt_gpu_dir}
    # run sim
    python scripts/run_simulation.py -M "mpiexec -n 2" -Y ${accel_sim_dir}/util/job_launching/apps/define-all-apps.yml -B ${benchmarks} -T ${trace_dir} -H TITANV --granularity 2 -R ${report_dir} 2>&1 | tee run_simulation.log

    # get stat
    python scripts/get_stat_sim.py -Y "${accel_sim_dir}/util/job_launching/apps/define-all-apps.yml" -B ${benchmarks} -T ${report_dir} -o ${res_sim_json}
}

draw(){
    # draw
    cd ${GIMT_dir}
    python utils/draw/draw_1.py -S ${res_sim_json} -H ${res_hw_json} -o ${draw_output} -D PPT-GPU
}

# run_trace
# run_hw
# run_sim
# draw