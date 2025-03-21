# PPT-GPU: Performance Prediction Toolkit for GPUs

本仓库包含一个 GPU 性能分析模型，从 [PPT-GPU]() 项目修改而来。项目优化了原本的计算模型和访存模型，包括：支持 SM 子核建模、使用自己实现的 Cache 功能模拟器替代原本的 SDCM 分析模型等、并且支持输出应用的 CPI 栈信息等。

## 要求

- CUDA 11, 使用 CUDA 12.x 需要迁移 NVBit Trace 代码，详见 https://github.com/NVlabs/NVBit/releases/tag/1.7
- Volta 架构之后的 GPU，Pascal SM 和 L1 架构均存在差异，建模效果很差

## 使用

- `env_jobs.sh, env.sh run_helper.sh` 脚本组合在一起，包含了使用 PPT-GPU 的所有步骤，包括 Trace 采集，硬件性能数据采集，模型预测。
- 进行 DSE，参考：`run_DSE*.sh` 脚本。
- memory_model 包含了实现的新内存模型封装，也包含了测试相关 shell 脚本
- `paper_scripts/paper.ipynb` 包含了论文中数据分析、画图的所有脚本

## References

本文实现的内存模型
```
Yuan F, Hao X, et al. Modeling GPU Memory Systems Based on Cache Functional Simulation[J]. Journal of Chinese Computer Systems,2025
```

对于原本的 PPT-GPU 模型，请参考：[SC' 21](https://doi.org/10.1145/3458817.3476221) paper ***(Hybrid, Scalable, Trace-Driven Performance Modeling of GPGPUs)***.

```
@inproceedings{Arafa2021PPT-GPU,
  author = {Y. {Arafa} and A. {Badawy} and A. {ElWazir} and A. {Barai} and A. {Eker} and G. {Chennupati} and N. {Santhi} and S. {Eidenbenz}},
  title = {Hybrid, Scalable, Trace-Driven Performance Modeling of GPGPUs},
  year = {2021},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  series = {SC '21}
}
```

对于 PPT-GPU 原本的内存模型，请参考：[ICS' 20](https://doi.org/10.1145/3392717.3392761) paper.
```
@inproceedings{Arafa2020PPT-GPU-MEM,
author = {Y. {Arafa} and A. {Badawy} and G. {Chennupati} and A. {Barai} and N. {Santhi} and S. {Eidenbenz}},
title = {Fast, Accurate, and Scalable Memory Modeling of GPGPUs Using Reuse Profiles},
year = {2020},
booktitle = {Proceedings of the 34th ACM International Conference on Supercomputing},
series = {ICS '20}
}
```

## License

按照 PPT-GPU 的 License 要求，保留原本的 License 信息：

&copy 2017. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
