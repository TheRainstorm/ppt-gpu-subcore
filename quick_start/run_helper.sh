run_trace(){
nvbit_version=1.5.5
time_out=1200  # 超时跳过当前应用
python ${ppt_gpu_dir}/scripts/run_hw_trace.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} -D ${GPU} --trace_tool ${ppt_gpu_dir}/tracing_tool/tracer-${nvbit_version}.so --time-out ${time_out} > /dev/null
}

run_hw(){
hw_prof_type=ncu-full
loop=1
time_out=1200  # 超时跳过当前应用

#！！！自己采集应用性能时需要取消注释！！！
# python ${ppt_gpu_dir}/scripts/run_hw_profling.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} -D ${GPU} --select ${hw_prof_type} --loop-cnt ${loop} --time-out ${time_out} > /dev/null
python ${ppt_gpu_dir}/scripts/get_stat_hw.py -B ${benchmarks} -F ${filter_app} -T ${trace_dir} --select ${hw_prof_type} -o ${res_hw_ncu_json} --loop-cnt ${loop}
python ${ppt_gpu_dir}/scripts/convert_hw_metrics.py -i ${res_hw_ncu_json} -o ${res_hw_json}
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_hw_ncu_json} -I "ncu" -o ${res_hw_cpi_json} # convert to cpi stack
}

run_sim(){
time_out=1200  # 超时跳过当前应用
python ${ppt_gpu_dir}/scripts/run_simulation.py -F ${filter_app} -B ${benchmarks} -T ${trace_dir} -H ${GPU_PROFILE} --granularity 2 -R ${report_dir} --time-out ${time_out} --extra-params "${model_extra_params}" > /dev/null
python ${ppt_gpu_dir}/scripts/get_stat_sim2.py -B ${benchmarks} -F ${filter_app} -R ${report_dir} -T ${trace_dir} -o ${res_sim_json}

# convert to cpi stack
python ${ppt_gpu_dir}/scripts/draw/convert_cpi_stack.py -i ${res_sim_json} -I "ppt_gpu" -o ${res_sim_cpi_json}

# 模拟结果保存在 csv 中
python ${ppt_gpu_dir}/scripts/analysis_result.py -B ${benchmarks} -F ${filter_app} -S ${res_sim_json} -H ${res_hw_sim_json} -o res_${model}_${gpu}_${cuda_version}_${GPU_PROFILE}_${run_name}.xlsx
}

draw(){
python ${ppt_gpu_dir}/scripts/draw/draw_1.py -F ${filter_app} -B ${benchmarks} -S ${res_sim_json} -H ${res_hw_sim_json} -o ${draw_output} app_by_bench
python ${ppt_gpu_dir}/scripts/draw/draw_cpi_stack.py -F ${filter_app} -B ${benchmarks} -S ${res_sim_cpi_json} -R ${res_hw_cpi_sim_json} -o ${draw_output} --s2s
cp ${res_hw_sim_json} ${draw_output}
cp ${res_hw_cpi_sim_json} ${draw_output}
cp ${res_sim_json} ${draw_output}
}

print_summary(){
    date '+%Y-%m-%d %H:%M:%S'
    echo "Summary:\n"
    echo "[COMMON]:"
    echo "ppt_gpu_dir: $ppt_gpu_dir"
    echo "trace_dir: $trace_dir"
    echo "apps_yaml: $apps_yaml"
    echo ""

    echo "[Tracing]:"
    echo "cuda_version: $cuda_version"
    echo "HW GPU: $gpu [${GPU}]"
    echo ""

    echo "[Simulation]:"
    echo "benchmarks: $benchmarks"
    echo "filter_app: $filter_app"
    echo "GPU Profile: $GPU_PROFILE"
    echo "model: $model"
    echo "run_name: $run_name"
    echo "model_extra_params: $model_extra_params"
    echo ""

    echo "[Files]:"
    echo "res_hw_json: $res_hw_json"
    echo "res_hw_sim_json: $res_hw_sim_json"
    echo "res_sim_json: $res_sim_json"
    echo "report_dir: $report_dir"
    echo "draw_output: $draw_output"
    echo ""
}

run(){
print_summary >> ${log_file}
echo "run $1" >> ${log_file}

if (( $1 & 8 )); then
echo "========================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') run trace"
echo "========================================="
run_trace
fi
if (( $1 & 4 )); then
echo "========================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') run profling"
echo "========================================="
run_hw
fi

if (( $1 & 2 )); then
echo "========================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') run simulation"
echo "========================================="
run_sim
fi

if (( $1 & 1 )); then
echo "========================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') draw"
echo "========================================="
draw
fi
}