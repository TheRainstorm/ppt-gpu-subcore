import json
import re

from enum import IntEnum

s = '''
inst_executed   smsp__inst_executed.sum
inst_issued smsp__inst_executed.sum
ipc smsp__inst_executed.avg.per_cycle_active
issued_ipc  smsp__inst_issued.avg.per_cycle_active
    
active_cycles   sm__cycles_active.sum
active_cycles_sys   sys__cycles_active.sum
    gpc__cycles_elapsed.avg
elapsed_cycles_pm   sm__cycles_elapsed.sum
elapsed_cycles_sm   sm__cycles_elapsed.sum
elapsed_cycles_sys  sys__cycles_elapsed.sum
active_warps    sm__warps_active.sum
achieved_occupancy  sm__warps_active.avg.pct_of_peak_sustained_active
    
tex_cache_hit_rate  l1tex__t_sector_hit_rate.pct
global_hit_rate (l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum) / (l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum)
l2_tex_hit_rate lts__t_sector_hit_rate.pct
l2_tex_read_hit_rate    lts__t_sector_op_read_hit_rate.pct
l2_tex_write_hit_rate   lts__t_sector_op_write_hit_rate.pct
    
global_load_requests    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
global_store_requests   l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
gld_transactions    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
gst_transactions    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
l2_read_transactions    lts__t_sectors_op_read.sum + lts__t_sectors_op_atom.sum + lts__t_sectors_op_red.sum
l2_write_transactions   lts__t_sectors_op_write.sum + lts__t_sectors_op_atom.sum + lts__t_sectors_op_red.sum
dram_read_transactions  dram__sectors_read.sum
dram_write_transactions dram__sectors_write.sum
gld_transactions_per_request    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
shared_load_transactions    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
shared_store_transactions   l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
l2_tex_read_transactions    lts__t_sectors_srcunit_tex_op_read.sum
tex_cache_transactions  l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active + l1tex__tex_writeback_active.avg.pct_of_peak_sustained_active
'''


lines = s.strip().split('\n')
for line in lines:
    # print(line)
    try:
        m = re.search(r'(\w+)\ +(\w.*)$', line)
        if m:
            nvprof, ncu = m.groups()
            # print(ncu)
            ncu_cmd = re.sub(r'([\w.]+)', "k['\g<1>']", ncu)
            # print(ncu_cmd)
            print(f"kernel_res['{nvprof}'] = {ncu_cmd}")
    except:
        pass

m = re.findall(r'(?P<metric>(\w+\.)+\w+)', s)
print(m)