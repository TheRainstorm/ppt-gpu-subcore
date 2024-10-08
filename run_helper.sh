#!/usr/bin/env bash

run_trace(){
# trace
python ${ppt_gpu_dir}/scripts/run_hw_trace.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} -D 0 --trace_tool ${ppt_gpu_dir}/tracing_tool/tracer.so
}

run_hw(){
# hw run
# python ${ppt_gpu_dir}/scripts/run_hw_profling.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} --loop_cnt 3
# python ${ppt_gpu_dir}/scripts/get_stat_hw.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} -o ${res_hw_nvprof_json}
# cp ${res_hw_nvprof_json} ${res_hw_json}

python ${ppt_gpu_dir}/scripts/run_hw_profling.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} --select ncu --loop_cnt 3
python ${ppt_gpu_dir}/scripts/get_stat_hw.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} --select ncu -o ${res_hw_ncu_json} --loop 3
python ${ppt_gpu_dir}/scripts/convert_hw_metrics.py -i ${res_hw_ncu_json} -o ${res_hw_json}
}

run_hw_ncu(){
# hw run
python ${ppt_gpu_dir}/scripts/run_hw_profling.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} --select ncu-cpi --loop_cnt 3

python ${ppt_gpu_dir}/scripts/get_stat_hw.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} --select ncu-cpi -o ${res_hw_ncu_cpi_json} --loop 3
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_hw_ncu_cpi_json} -I "ncu" -o ${res_hw_cpi_json} # convert to cpi stack
}

run_sim(){
cd ${ppt_gpu_dir}
# run sim
python ${ppt_gpu_dir}/scripts/run_simulation.py -M "mpiexec -n 2" -F ${filter_app} -B ${benchmarks} -T ${trace_dir} -H TITANV --granularity 2 -R ${report_dir} 2>&1 | tee run_simulation.log

# get stat
python ${ppt_gpu_dir}/scripts/get_stat_sim.py -B ${benchmarks} -F ${filter_app} -T ${report_dir} -o ${res_sim_json}

# convert to cpi stack
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu" -o ${res_sim_cpi_json}
# python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu_sched"  -o ${res_sim_sched_cpi_json}
}

draw(){
# draw
# rm -rf ${draw_output} 
# python ${ppt_gpu_dir}/scripts/draw/draw_1.py -F ${filter_app} -S ${res_sim_json} -H ${res_hw_nvprof_json} -o ${draw_output} -d app_nvprof app
python ${ppt_gpu_dir}/scripts/draw/draw_1.py -F ${filter_app} -S ${res_sim_json} -H ${res_hw_json} -o ${draw_output} app
# python ${ppt_gpu_dir}/scripts/draw/draw_1.py -F ubench -S ${res_sim_json} -H ${res_hw_json} -o ${draw_output} app  # draw specific app
python ${ppt_gpu_dir}/scripts/draw/draw_1.py -F rodinia-2.0-ft -S ${res_sim_json} -H ${res_hw_json} -o ${draw_output} -d kernel_rodinia2 kernel

# python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -S ${res_sim_cpi_json} -o ${draw_output} --subdir cpi_warp
# python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -S ${res_sim_cpi_json} -R ${res_hw_cpi_json} -o ${draw_output} --s2s --draw-subcore
python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -S ${res_sim_cpi_json} -R ${res_hw_cpi_json} -o ${draw_output} --s2s
# python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -F ubench -S ${res_sim_cpi_json} -R ${res_hw_cpi_json} -o ${draw_output} --s2s

}

run(){
if (( $1 & 8 )); then
echo "run trace"
run_trace
fi
if (( $1 & 4 )); then
echo "run profling"
run_hw
run_hw_ncu
fi

if (( $1 & 2 )); then
echo "run simulation"
run_sim
fi

if (( $1 & 1 )); then
echo "draw"
draw
fi
}

run_single(){
if (( $1 & 8 )); then
echo "run trace"
# run hw trace
python ${ppt_gpu_dir}/scripts/run_hw_trace.py -F ${single_app} -B ${benchmarks} -T ${trace_dir} -D 0 --trace_tool ${ppt_gpu_dir}/tracing_tool/tracer.so
fi

if (( $1 & 4 )); then
echo "run profling"
# run_hw profling
# python ${ppt_gpu_dir}/scripts/run_hw_profling.py -F ${single_app} -B ${benchmarks} -T ${trace_dir} --loop_cnt 3
# python ${ppt_gpu_dir}/scripts/get_stat_hw.py -F ${single_app} -B ${benchmarks} -T ${trace_dir} -o ${res_hw_json}
python ${ppt_gpu_dir}/scripts/run_hw_profling.py -F ${single_app} -B ${benchmarks} -T ${trace_dir} --select ncu --loop_cnt 3
python ${ppt_gpu_dir}/scripts/get_stat_hw.py -F ${single_app} -B ${benchmarks} -T ${trace_dir} --select ncu -o ${res_hw_ncu_json}
python ${ppt_gpu_dir}/scripts/convert_hw_metrics.py -i ${res_hw_ncu_json} -o ${res_hw_json}

# run_hw_ncu
python ${ppt_gpu_dir}/scripts/run_hw_profling.py -F ${single_app} -B ${benchmarks} -T ${trace_dir} --select ncu-cpi --loop_cnt 3
python ${ppt_gpu_dir}/scripts/get_stat_hw.py -F ${single_app} -B ${benchmarks} -T ${trace_dir} --select ncu-cpi -o ${res_hw_ncu_cpi_json} --loop 3
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_hw_ncu_cpi_json} -I "ncu" -o ${res_hw_cpi_json}
fi

if (( $1 & 2 )); then
echo "run simulation"
# run_sim
python ${ppt_gpu_dir}/scripts/run_simulation.py -F ${single_app} -M "mpiexec -n 2" -B ${benchmarks} -T ${trace_dir} -H TITANV --granularity 2 -R ${single_report_dir}
python ${ppt_gpu_dir}/scripts/get_stat_sim.py -F ${single_app} -B ${benchmarks} -T ${single_report_dir} -o ${res_sim_json}
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu" -o ${res_sim_cpi_json}
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu_sched"  -o ${res_sim_sched_cpi_json}
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu" -O "gsi-detail" -o ${res_sim_detail_cpi_json}
fi

if (( $1 & 1 )); then
echo "draw"
# draw error
python ${ppt_gpu_dir}/scripts/draw/draw_1.py -F ${single_app} -S ${res_sim_json} -H ${res_hw_json} -o ${single_draw_output} single

# draw cpi stack
python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -F ${single_app} -S ${res_sim_cpi_json} -R ${res_hw_cpi_json} -o ${single_draw_output} --seperate-dir --s2s --draw-subcore
python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -F ${single_app} -S ${res_sim_detail_cpi_json} -R ${res_hw_cpi_json} -o ${single_draw_output} --seperate-dir --s2s --subplot-s2s 
# python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -F ${single_app} -S ${res_sim_cpi_json} -o ${single_draw_output} --seperate-dir
python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -F ${single_app} -S ${res_sim_sched_cpi_json} -o ${single_draw_output} --seperate-dir --subdir "cpi_sched"
fi
}
