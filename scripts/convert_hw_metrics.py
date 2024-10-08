import json
import argparse

def convert_ncu_to_nvprof(data):
    res_json = {}
    for app_arg, app_res in data.items():
        res_json[app_arg] = []
        print(f"{app_arg}: {len(app_res)}")
        for i, k in enumerate(app_res):
            kernel_res = {}
            kernel_res['kernel_name'] = k['kernel_name']
            
            kernel_res['inst_executed'] = k['smsp__inst_executed.sum']
            kernel_res['inst_issued'] = k['smsp__inst_executed.sum']
            kernel_res['ipc'] = k['smsp__inst_executed.avg.per_cycle_active']
            kernel_res['issued_ipc'] = k['smsp__inst_issued.avg.per_cycle_active']
            
            kernel_res['active_cycles'] = k['sm__cycles_active.sum']
            # kernel_res['active_cycles_sys'] = k['sys__cycles_active.sum']
            # kernel_res['active_cycles_sys'] = k['gpc__cycles_elapsed.avg']
            kernel_res['active_cycles_sys'] = k['gpc__cycles_elapsed.max']
            kernel_res['elapsed_cycles_pm'] = k['sm__cycles_elapsed.sum']
            kernel_res['elapsed_cycles_sm'] = k['sm__cycles_elapsed.sum']
            kernel_res['elapsed_cycles_sys'] = k['sys__cycles_elapsed.sum']
            kernel_res['active_warps'] = k['sm__warps_active.sum']
            kernel_res['achieved_occupancy'] = k['sm__warps_active.avg.pct_of_peak_sustained_active']/100
            
            try:
                kernel_res['global_hit_rate'] = (k['l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum'] + k['l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum'] + k['l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum'] + k['l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum']) / (k['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'] + k['l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'] + k['l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum'] + k['l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum'])
            except ZeroDivisionError:
                kernel_res['global_hit_rate'] = 0
            kernel_res['tex_cache_hit_rate'] = k['l1tex__t_sector_hit_rate.pct']
            kernel_res['l2_tex_hit_rate'] = k['lts__t_sector_hit_rate.pct']
            kernel_res['l2_tex_read_hit_rate'] = k['lts__t_sector_op_read_hit_rate.pct']
            kernel_res['l2_tex_write_hit_rate'] = k['lts__t_sector_op_write_hit_rate.pct']
            
            kernel_res['global_load_requests'] = k['l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum']
            kernel_res['global_store_requests'] = k['l1tex__t_requests_pipe_lsu_mem_global_op_st.sum']
            kernel_res['gld_transactions'] = k['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum']
            kernel_res['gst_transactions'] = k['l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum']
            kernel_res['l2_read_transactions'] = k['lts__t_sectors_op_read.sum'] + k['lts__t_sectors_op_atom.sum'] + k['lts__t_sectors_op_red.sum']
            kernel_res['l2_write_transactions'] = k['lts__t_sectors_op_write.sum'] + k['lts__t_sectors_op_atom.sum'] + k['lts__t_sectors_op_red.sum']
            kernel_res['dram_read_transactions'] = k['dram__sectors_read.sum']
            kernel_res['dram_write_transactions'] = k['dram__sectors_write.sum']
            kernel_res['gld_transactions_per_request'] = k['l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio']
            kernel_res['shared_load_transactions'] = k['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum']
            kernel_res['shared_store_transactions'] = k['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum']
            kernel_res['l2_tex_read_transactions'] = k['lts__t_sectors_srcunit_tex_op_read.sum']
            kernel_res['tex_cache_transactions'] = k['l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active'] + k['l1tex__tex_writeback_active.avg.pct_of_peak_sustained_active']

            res_json[app_arg].append(kernel_res)
    return res_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument("-i", "--input",
                        help="ori input")
    parser.add_argument("-o", "--output",
                        help="converted output")
    parser.add_argument("-O", "--out-type",
                        choices=["nvprof"],
                        default="nvprof",
                        help="classifiy model")
    parser.add_argument("-I", "--in-type",
                        choices=["ncu"],
                        default="ncu",
                        help="input json format")
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    detail = False
    if "detail" in args.out_type:
        detail = True
    
    if args.in_type == "ncu":
        if args.out_type == "nvprof":
            output_json = convert_ncu_to_nvprof(data)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    with open(args.output, 'w') as f:
        json.dump(output_json, f, indent=4)
