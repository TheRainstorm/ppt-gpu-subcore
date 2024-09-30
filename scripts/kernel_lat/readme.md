## Usage

1. run `generate_app_yaml.py`, move the generated `kernel_lat.yaml` to `../apps`
2. run hardware profiling and get the result json file (check `./run.sh`)

    ```shell
    python ${ppt_gpu_dir}/scripts/run_hw_profling.py -B ${benchmarks} -T ${trace_dir} --select ncu --loop_cnt 1
    python ${ppt_gpu_dir}/scripts/get_stat_hw.py -B ${benchmarks} -T ${trace_dir} --select ncu -o ${res_hw_ncu_json} --loop 1
    ```

3. run `draw.py` to generate the plot

    ```shell
    python draw.py -i ${res_hw_ncu_json} -o ${output_dir}
    ```
