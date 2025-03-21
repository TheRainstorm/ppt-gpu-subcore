## 脚本作用

基础：

- run_hw_trace
- run_hw_profling
    - get_stat_hw
    - convert_hw_metrics：只实现了从 ncu 数据，转换成 nvprof 数据
- run_simulation
    - get_stat_sim
    - run_simulation2 的唯一区别是 `--extra-params` 的处理方式不同，为了避免破坏原本的代码接口，新建了一个脚本
- analysis_result: 读取模拟和硬件结果，输出到一个 csv 文件中，用于分析

cpi

- draw/convert_cpi_stack
    - HW: {ncu, nvprof} to GSI
    - sim: {ppt_gpu, ppt_gpu_sched} to GSI


其他脚本：

- json2csv：把模拟结果和硬件结果输出到一个 csv 文件中
- rsync_trace: scan，扫描一个 trace 目录，输出 bench, app, trace 大小信息
- print_AMAT：过时了，不再使用

## hw tracing


## hw profiling

增加新的 bench，如何和之前的结果合并？

get_stat_hw 默认覆盖现有 hw res 文件。因为 ncu profile 的数据存储在 trace 目录中，因此每次重新 get 就行了，这保证了 app 出现顺序和 app yaml 一致。

需要把 benchmark_list 和 app-filter 设置为新的。

## 记录

参数记录

- fix_cycle：原本 ppt-gpu 输出的 cycle 错误加上了一个变量导致翻倍。该选项修正这个问题