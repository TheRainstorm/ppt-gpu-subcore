NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
GENCODE_SM60 := -gencode=arch=compute_60,code=sm_60
GENCODE_SM70 := -gencode=arch=compute_70,code=sm_70
GENCODE_SM75 := -gencode=arch=compute_75,code=sm_75
GENCODE_SM80 := -gencode=arch=compute_80,code=sm_80
ARCH := $(GENCODE_SM60) $(GENCODE_SM70) $(GENCODE_SM75) $(GENCODE_SM80)

NVCC_VER_REQ=11.0
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

NVBIT_PATH=nvbit_1.5.5/core
INCLUDES=-I$(NVBIT_PATH)

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=$(wildcard *.cu)

OBJECTS=$(SOURCES:.cu=.o)


mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL=tracer-1.5.5.so

all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) $(ARCH) -O3 $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

%.o: %.cu
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall $(ARCH) -O3 -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions $(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f *.o
