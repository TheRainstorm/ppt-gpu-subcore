import json
import argparse

gsi_stall_list = ['MemData', 'MemStruct', 'CompData', 'CompStruct', 'NotSelect', 'NoStall', 'Sync', 'Misc', 'Idle', 'IMC']
gsi_stall_list_detail = [
    'MemData',
    'MemStruct',
    'CompData',
    'CompStruct', 'CompStruct.iALU', 'CompStruct.fALU', 'CompStruct.hALU', 'CompStruct.dALU', 'CompStruct.SFU', 'CompStruct.dSFU', 'CompStruct.iTCU', 'CompStruct.hTCU', 'CompStruct.BRA', 'CompStruct.EXIT',
    'NotSelect', 'NoStall', 'Sync', 'Misc', 'Idle', 'IMC'
]
def convert_ncu_to_gsi(data):
    res_json = {}
    for app_arg, app_res in data.items():
        res_json[app_arg] = []
        print(f"{app_arg}: {len(app_res)}")
        for i, k in enumerate(app_res):
            # print(f"{i}: {k['kernel_name']}")
            kernel_cpi_res = {}
            kernel_cpi_res['MemData'] = k['smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio']
            kernel_cpi_res['MemStruct'] = k['smsp__average_warps_issue_stalled_drain_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio']
            kernel_cpi_res['CompData'] = k['smsp__average_warps_issue_stalled_wait_per_issue_active.ratio']
            kernel_cpi_res['CompStruct'] = k['smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio']
            kernel_cpi_res['NotSelect'] = k['smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio']
            kernel_cpi_res['NoStall'] = k['smsp__average_warps_issue_stalled_selected_per_issue_active.ratio']
            kernel_cpi_res['Sync'] = k['smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_membar_per_issue_active.ratio']
            kernel_cpi_res['Misc'] = k['smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_misc_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio']
            # warpgroup_arrive
            
            kernel_cpi_res['Idle'] = 0
            kernel_cpi_res['debug'] = {}
            kernel_cpi_res['debug']['sum'] = kernel_cpi_res['MemData'] + kernel_cpi_res['CompData'] + kernel_cpi_res['MemStruct'] + kernel_cpi_res['CompStruct'] + kernel_cpi_res['NotSelect'] + kernel_cpi_res['NoStall'] + kernel_cpi_res['Sync'] + kernel_cpi_res['Misc']
            kernel_cpi_res['debug']['smsp__average_warp_latency_per_inst_issued.ratio'] = k['smsp__average_warp_latency_per_inst_issued.ratio']
            
            kernel_cpi_res = fill_gsi(kernel_cpi_res)
            # sort by gsi stall list
            kernel_cpi_res = {k: kernel_cpi_res[k] for k in gsi_stall_list}
            res_json[app_arg].append(kernel_cpi_res)
    return res_json

def convert_nvprof_to_gsi(data):
    res_json = {}
    nvprof_stalls = ['stall_constant_memory_dependency','stall_exec_dependency','stall_inst_fetch','stall_memory_dependency','stall_memory_throttle','stall_not_selected','stall_other','stall_pipe_busy','stall_sleeping','stall_sync','stall_texture']
    for app_arg, app_res in data.items():
        res_json[app_arg] = []
        print(f"{app_arg}: {len(app_res)}")
        for i, k in enumerate(app_res):
            # print(f"{i}: {k['kernel_name']}")
            for stall in nvprof_stalls:
                if stall not in k:
                    k[stall] = 0
            kernel_cpi_res = {}
            kernel_cpi_res['MemData'] = k['stall_constant_memory_dependency'] + k['stall_memory_dependency']
            kernel_cpi_res['MemStruct'] = k['stall_memory_throttle'] + k['stall_texture']
            kernel_cpi_res['CompData'] = k['stall_exec_dependency']
            kernel_cpi_res['CompStruct'] = k['stall_pipe_busy']
            kernel_cpi_res['NotSelect'] = k['stall_not_selected']
            kernel_cpi_res['NoStall'] = 1
            kernel_cpi_res['Sync'] = k['stall_sync']
            kernel_cpi_res['Misc'] = k['stall_other'] + k['stall_sleeping'] + k['stall_inst_fetch']
            # warpgroup_arrive
            
            kernel_cpi_res['Idle'] = 0
            kernel_cpi_res['debug'] = {}
            kernel_cpi_res['debug']['sum'] = kernel_cpi_res['MemData'] + kernel_cpi_res['CompData'] + kernel_cpi_res['MemStruct'] + kernel_cpi_res['CompStruct'] + kernel_cpi_res['NotSelect'] + kernel_cpi_res['NoStall'] + kernel_cpi_res['Sync'] + kernel_cpi_res['Misc']
            
            kernel_cpi_res = fill_gsi(kernel_cpi_res)
            # sort by gsi stall list
            kernel_cpi_res = {k: kernel_cpi_res[k] for k in gsi_stall_list}
            res_json[app_arg].append(kernel_cpi_res)
    return res_json

def state_to_cpi(state_dict, set_debug=False, KL_ratio=0):
    cpi_stack = {}
    total_cycle = sum(state_dict.values())
    total_inst = state_dict.get('NoStall', 0)  # avoid zero
    if total_inst == 0:
        return {}
    
    for state in state_dict:
        cpi_stack[state] = state_dict[state]/total_inst
    
    if KL_ratio:
        other_sum = sum(cpi_stack.values())
        cpi_stack['IMC'] = KL_ratio*other_sum
    
    if set_debug:
        cpi_stack['debug'] = {}
        cpi_stack['debug']['total_cycle'] = total_cycle
        cpi_stack['debug']['total_inst'] = total_inst
        cpi_stack['debug']['average_warp_cycle_per_inst'] = total_cycle/total_inst
    return cpi_stack

def get_merged_cpi_stack(cpi_stack):
    cpi_stack_merged = {}
    for k,v in cpi_stack.items():
        if type(v) == dict:
            continue
        if '.' in k:
            stall_type, sub_type = k.split('.')
            cpi_stack_merged[stall_type] = cpi_stack_merged.get(stall_type, 0) + v
        else:
            cpi_stack_merged[k] = cpi_stack_merged.get(k, 0) + v
    return cpi_stack_merged

def fill_gsi(cpi_stack, stall_list=gsi_stall_list):
    cpi_stack_new = {}
    for k in stall_list:
        cpi_stack_new[k] = cpi_stack.get(k, 0)
    return cpi_stack_new
    
def get_cpi_stack_list(state_dict_list, detail=False, KL_ratio=0):
    if type(state_dict_list) == dict:
        # no subcore
        cpi_stack_list = [state_to_cpi(state_dict_list, KL_ratio=KL_ratio)]
    else:
        cpi_stack_list = [state_to_cpi(state_dict, KL_ratio=KL_ratio) for state_dict in state_dict_list]
    non_zero_num = len([cpi_stack for cpi_stack in cpi_stack_list if cpi_stack])
    # convert to gsi
    
    if not detail:
        cpi_stack_list = [fill_gsi(get_merged_cpi_stack(cpi_stack)) for cpi_stack in cpi_stack_list]
    else:
        # all_kernel_stall_list = get_all_kernel_stall_list(cpi_stack_list)
        # print(all_kernel_stall_list)
        cpi_stack_list = [fill_gsi(cpi_stack, stall_list=gsi_stall_list_detail) for cpi_stack in cpi_stack_list]
        
    def avg_dict(dict_list, non_zero_num):
        all_keys = set()
        for d in dict_list:
            all_keys.update(d.keys())
        non_zero = [d for d in dict_list if d]
        avg_dict = {}
        for k in all_keys:
            if k == 'debug':
                continue
            avg_dict[k] = sum([d.get(k, 0) for d in non_zero])/non_zero_num
        return avg_dict
    
    if not detail: # subcore may have different detail key, so avg is difficut
        # append avg dict to last if more than one subcore
        cpi_stack_list.append(avg_dict(cpi_stack_list, non_zero_num))
    return cpi_stack_list
        
def convert_ppt_gpu_to_gsi(data, select, detail=False):
    res_json = {}
    for app_arg, app_res in data.items():
        print(f"{app_arg}: {len(app_res)}")
        res_json[app_arg] = []
        for i, kernel_res in enumerate(app_res):
            KL_ratio = kernel_res.get('KL_ratio', 0)
            if select == "warp":
                cpi_stack_list = get_cpi_stack_list(kernel_res['warp_stats']['stall_types'], detail, KL_ratio=KL_ratio)
            else:
                cpi_stack_list = get_cpi_stack_list(kernel_res['scheduler_stats']['stall_types'], detail, KL_ratio=KL_ratio)
            res_json[app_arg].append(cpi_stack_list)
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
                        choices=["gsi", "gsi-detail"],
                        default="gsi",
                        help="classifiy model")
    parser.add_argument("-I", "--in-type",
                        choices=["ncu", "ppt_gpu", "ppt_gpu_sched", "nvprof"],
                        help="input json format")
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    detail = False
    if "detail" in args.out_type:
        detail = True
    
    if args.in_type == "ncu":
        output_json = convert_ncu_to_gsi(data)
    elif args.in_type == "ppt_gpu":
        output_json = convert_ppt_gpu_to_gsi(data, "warp", detail=detail)
    elif args.in_type == "ppt_gpu_sched":
        output_json = convert_ppt_gpu_to_gsi(data, "sched", detail=detail)
    elif args.in_type == "nvprof":
        output_json = convert_nvprof_to_gsi(data)
    else:
        raise NotImplementedError(f"Unknown input type: {args.in_type}")
    
    with open(args.output, 'w') as f:
        json.dump(output_json, f, indent=4)
