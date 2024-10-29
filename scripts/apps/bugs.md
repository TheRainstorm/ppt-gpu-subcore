

### hybridsort

trace 时会报错，不 trace 时正常

```
Sorting list of 4194304 floats
Sorting on GPU...Kernel #1

terminate called after throwing an instance of 'std::out_of_range'
  what():  _Map_base::at
Aborted (core dumped)
./run_tracing.sh  15.91s user 1.12s system 184% cpu 9.218 total
```

### mummergpu

不 trace 时也报错

```
release/mummergpu-rodinia-3.1 ./data/NC_003997.fna ./data/NC_003997_q100bp.fna > NC_00399.out
TWO_LEVEL_NODE_TREE is 0
TWO_LEVEL_CHILD_TREE is 0
QRYTEX is 0
COALESCED_QUERIES is 0
REFTEX is 0
REORDER_REF is 0
NODETEX is 1
CHILDTEX is 1
MERGETEX is 0
REORDER_TREE is 1
RENUMBER_TREE is 1
Loading ref: ./data/NC_003997.fna... 5227293 bp. [0.15195s]
Opening ./data/NC_003997_q100bp.fna...
Streaming reference pages against all queries
Stream will use 1 pages for 5227293 bases, page size = 5227293
Building reference texture...
  Creating Suffix Tree... 8606425 nodes [2.78018s]
  Renumbering tree... [0.56758s]
  Flattening Tree...  node: 4096x2112 children: 4096x2112 [1]    2121206 segmentation fault (core dumped)  /staff/fyyuan//repo/GPGPUs-Workloads/bin/11.0/release/mummergpu-rodinia-3.1
```

### particlefilter_native

trace 时会报错，不 trace 时正常

```
VIDEO SEQUENCE TOOK 0.026896
TIME TO GET NEIGHBORS TOOK: 0.000013
malloc(): invalid size (unsorted)
Aborted (core dumped)
./run_tracing.sh  0.03s user 0.00s system 12% cpu 0.293 total
```