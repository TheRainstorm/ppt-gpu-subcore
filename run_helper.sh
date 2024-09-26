#!/usr/bin/env bash

run_trace(){
# trace
# rm -rf ${accel_sim_dir}/hw_run/traces/device-0/11.0
# python3 ${accel_sim_dir}/util/tracer_nvbit/run_hw_trace.py -B ${hw_new_benchmarks} -D 0 --trace_tool ${ppt_gpu_dir}/tracing_tool/tracer.so

pyrhon ${ppt_gpu_dir}/scripts/run_hw_trace.py -Y ${apps_yaml} -B ${hw_new_benchmarks} -T ${trace_dir} -D 0 --trace_tool ${ppt_gpu_dir}/tracing_tool/tracer.so
}

run_hw(){
# hw run
python ${ppt_gpu_dir}/scripts/run_hw_profling.py -Y ${apps_yaml} -B ${hw_new_benchmarks} -T ${trace_dir} --loop_cnt 3
python ${ppt_gpu_dir}/scripts/get_stat_hw.py -Y ${apps_yaml} -B ${benchmarks} -T ${trace_dir} -o ${res_hw_json}
}

run_hw_ncu(){
# hw run
python ${ppt_gpu_dir}/scripts/run_hw_profling.py -Y ${apps_yaml} -B ${hw_new_benchmarks} -T ${trace_dir} --ncu --loop_cnt 3

python ${ppt_gpu_dir}/scripts/get_stat_hw.py -Y ${apps_yaml} -B ${benchmarks} -T ${trace_dir} --ncu -o ${res_hw_ncu_json} --loop 3

# convert to cpi stack
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_hw_ncu_json} -I "ncu" -o ${res_hw_cpi_json}
}

run_sim(){
cd ${ppt_gpu_dir}
# run sim
python ${ppt_gpu_dir}/scripts/run_simulation.py -M "mpiexec -n 2" -Y ${apps_yaml} -B ${sim_new_benchmarks} -T ${trace_dir} -H TITANV --granularity 2 -R ${report_dir} 2>&1 | tee run_simulation.log

# get stat
python ${ppt_gpu_dir}/scripts/get_stat_sim.py -Y ${apps_yaml} -B ${benchmarks} -T ${report_dir} -o ${res_sim_json}

# convert to cpi stack
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu" -o ${res_sim_cpi_json}
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu_sched"  -o ${res_sim_sched_cpi_json}
}

draw(){
# draw
rm -rf ${draw_output}
python ${ppt_gpu_dir}/scripts/draw/draw_1.py -S ${res_sim_json} -H ${res_hw_json} -o ${draw_output} -D PPT-GPU

# # draw hw cpi stack
# python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -S ${res_hw_cpi_json} -o ${draw_cpi_ncu_output}

# # draw ppt gpu cpi stack
python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -S ${res_sim_cpi_json} -o ${draw_output}

# draw side2sdie
python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -S ${res_sim_cpi_json} -R ${res_hw_cpi_json} -o ${draw_output}

# save result 用于复现
cp ${res_hw_json} ${res_hw_ncu_json} ${res_hw_cpi_json} ${res_sim_json} ${res_sim_cpi_json} ${draw_output}
}

run_single(){

if [[ $1 != "0" ]]; then
echo "run trace and profiling"
# run hw trace
python ${ppt_gpu_dir}/scripts/run_hw_trace.py --apps ${single_app} -Y ${apps_yaml} -B ${benchmarks} -T ${trace_dir} -D 0 --trace_tool ${ppt_gpu_dir}/tracing_tool/tracer.so

# run_hw profling
python ${ppt_gpu_dir}/scripts/run_hw_profling.py --apps ${single_app} -Y ${apps_yaml} -B ${benchmarks} -T ${trace_dir} --loop_cnt 3
python ${ppt_gpu_dir}/scripts/get_stat_hw.py --apps ${single_app} -Y ${apps_yaml} -B ${benchmarks} -T ${trace_dir} -o ${res_hw_json}

# run_hw_ncu
python ${ppt_gpu_dir}/scripts/run_hw_profling.py --apps ${single_app} -Y ${apps_yaml} -B ${benchmarks} -T ${trace_dir} --ncu --loop_cnt 3
python ${ppt_gpu_dir}/scripts/get_stat_hw.py --apps ${single_app} -Y ${apps_yaml} -B ${benchmarks} -T ${trace_dir} --ncu -o ${res_hw_ncu_json} --loop 3
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_hw_ncu_json} -I "ncu" -o ${res_hw_cpi_json}
fi

# run_sim
python ${ppt_gpu_dir}/scripts/run_simulation.py --apps ${single_app} -Y ${apps_yaml} -B ${benchmarks} -T ${trace_dir} -H TITANV --granularity 2 -R ${single_report_dir}
python ${ppt_gpu_dir}/scripts/get_stat_sim.py --apps ${single_app} -Y ${apps_yaml} -B ${benchmarks} -T ${single_report_dir} -o ${res_sim_json}
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu" -o ${res_sim_cpi_json}
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu_sched"  -o ${res_sim_sched_cpi_json}

# draw
python ${ppt_gpu_dir}/scripts/draw/draw_1.py --apps ${single_app} -S ${res_sim_json} -H ${res_hw_json} -o ${single_draw_output} -D PPT-GPU
python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py --apps ${single_app} -S ${res_sim_cpi_json} -o ${single_draw_output}
python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py --apps ${single_app} -S ${res_sim_cpi_json} -R ${res_hw_cpi_json} -o ${single_draw_output}
}
# run_trace
# run_hw
# run_sim
# draw