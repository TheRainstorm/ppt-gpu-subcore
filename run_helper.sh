#!/usr/bin/env bash

run_trace(){
# trace
cd ${accel_sim_dir}
python3 ./util/tracer_nvbit/run_hw_trace.py -B ${hw_new_benchmarks} -D 0 --trace_tool /staff/fyyuan/repo/PPT-GPU/tracing_tool/tracer.so

# copy
rsync -av hw_run/traces/device-0/11.0/ ${trace_dir}
}

run_hw(){
# hw run
cd ${GIMT_dir}
python ${GIMT_dir}/utils/run_model/run_hw.py -Y "${accel_sim_dir}/util/job_launching/apps/define-all-apps.yml" -B ${hw_new_benchmarks} -T "${trace_dir}" --loop_cnt 3

python ${GIMT_dir}/utils/run_model/get_stat_hw.py -Y "${accel_sim_dir}/util/job_launching/apps/define-all-apps.yml" -B ${benchmarks} -T "${trace_dir}" -o ${res_hw_json}
}

run_hw_ncu(){
# hw run
# cd ${GIMT_dir}
python ${GIMT_dir}/utils/run_model/run_hw.py -Y "${accel_sim_dir}/util/job_launching/apps/define-all-apps.yml" -B ${benchmarks} -T "${trace_dir}" --ncu --loop_cnt 1

python ${GIMT_dir}/utils/run_model/get_stat_hw.py -Y "${accel_sim_dir}/util/job_launching/apps/define-all-apps.yml" -B ${benchmarks} -T "${trace_dir}" --ncu -o ${res_hw_ncu_json} --loop 1

# convert to cpi stack
python ${GIMT_dir}/utils/draw/convert_cpi_stack.py -i ${res_hw_ncu_json} -I "ncu" -o ${res_hw_cpi_json}
}

run_sim(){
# cd ${ppt_gpu_dir}
# run sim
python ${ppt_gpu_dir}/scripts/run_simulation.py -M "mpiexec -n 2" -Y ${accel_sim_dir}/util/job_launching/apps/define-all-apps.yml -B ${sim_new_benchmarks} -T ${trace_dir} -H TITANV --granularity 2 -R ${report_dir} 2>&1 | tee run_simulation.log

# get stat
python ${ppt_gpu_dir}/scripts/get_stat_sim.py -Y "${accel_sim_dir}/util/job_launching/apps/define-all-apps.yml" -B ${benchmarks} -T ${report_dir} -o ${res_sim_json}

# convert to cpi stack
python ${GIMT_dir}/utils/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu" -o ${res_sim_cpi_json}

python ${GIMT_dir}/utils/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu_sched"  -o ${res_sim_sched_cpi_json}
}

draw(){
# draw
cd ${GIMT_dir}
rm -rf ${draw_output}
python ${GIMT_dir}/utils/draw/draw_1.py -S ${res_sim_json} -H ${res_hw_json} -o ${draw_output} -D PPT-GPU

# # draw hw cpi stack
# python ${GIMT_dir}/utils/draw/draw_cpi_stack.py -S ${res_hw_cpi_json} -o ${draw_cpi_ncu_output}

# # draw ppt gpu cpi stack
python ${GIMT_dir}/utils/draw/draw_cpi_stack.py -S ${res_sim_cpi_json} -o ${draw_output}

# draw side2sdie
python ${GIMT_dir}/utils/draw/draw_cpi_stack.py -S ${res_sim_cpi_json} -R ${res_hw_cpi_json} -o ${draw_output}

# save result 用于复现
cp ${res_hw_json} ${res_hw_ncu_json} ${res_hw_cpi_json} ${res_sim_json} ${res_sim_cpi_json} ${draw_output}
}

test(){
}
# run_trace
# run_hw
# run_sim
# draw