The ***tracer_tool*** is used to extract the memory and SASS traces. This repo use and extend NVBit (NVidia Binary Instrumentation Tool) which is a research prototype of a dynamic binary instrumentation library for NVIDIA GPUs. Licence and agreement of NVBIT is found in the origianal [NVBIT repo](https://github.com/NVlabs/NVBit) (“This software contains source code provided by NVIDIA
    Corporation”)

NVBIT does not require application source code, any pre-compiled GPU application should work regardless of which compiler (or version) has been used (i.e. nvcc, pgicc, etc).

## Usage

* Please see our docker [wiki](https://github.com/NMSU-PEARL/PPT-GPU/wiki/Docker-Images-and-Usage) page for how to build the tracing tool and extract the traces using docker.

*  Setup the **MAX_KERNELS** variable in ***tracer.cu*** to define the limit on the number of kernels you want to instrument in the application 

* For stanalone building and running of the tracing_tool (no docker), please see below: 

### Building the tool

* Setup **ARCH** variable in the Makefile
* run make clean; make

### Extracting the traces

```shell
LD_PRELOAD=~/PPT-GPU/tracing_tool/tracer.so ./app.out
```

  The above command outputs two folders ***memory_traces*** and ***sass_traces*** each has the applications kernel traces. It also output ***app_config*** file whoch has information about the kernel executing inside the application. 

#### Memory tracing level control

Memory default tracing on block level, so that the trace is hardware independent (e.g SM number). When simulation, block is mapped to SM according block scheduling policy. It's sometimes useful to trace on SM level or warp level. The tracing tool provide a way to control the tracing level by setting the environment variable ***ENV_TRACING_LEVEL***. The value of ***ENV_TRACING_LEVEL*** is a bitmask, the bit 0 is for block level, bit 1 is for SM level, bit 2 is for warp_id in block level. The default value is 1.

```shell
export ENV_TRACING_LEVEL=1  # Block level tracing (default) --> memory_traces/kernel_1_block_0.mem
export ENV_TRACING_LEVEL=2  # SM level tracing   --> memory_traces/kernel_1_sm_0.mem 
export ENV_TRACING_LEVEL=5  # block level + warp id recorded
```
