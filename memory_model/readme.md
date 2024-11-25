- `run_benchmarks.py`：用于运行所有应用，模拟结果保存在 json 文件中。
- `caculate_MAEs.py`：用于读取 json 文件，计算所有应用的 MAE，Corr 值，并输出到 xlsx 文件中。xlsx 文件也保存了所有 kernel 访存的原始数据。
- `analysis_writeback.py`: 比较模拟的 dram 写事务数和通过 l2 写命中率计算的 dram 写事务数，输出到 xlsx 文件中。

