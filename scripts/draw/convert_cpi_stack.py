import json
import argparse

gsi_stall_list = ['MemData', 'MemStruct', 'CompData', 'CompStruct', 'NotSelect', 'NoStall', 'Sync', 'Misc', 'Idle']
def convert_ncu_to_gsi(data):
    res_json = {}
    for app_arg, app_res in data.items():
        res_json[app_arg] = []
        print(f"{app_arg}: {len(app_res)}")
        for i, k in enumerate(app_res):
            # print(f"{i}: {k['kernel_name']}")
            kernel_cpi_res = {}
            kernel_cpi_res['MemData'] = k['smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio']
            kernel_cpi_res['MemStruct'] = k['smsp__average_warps_issue_stalled_drain_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio']
            kernel_cpi_res['CompData'] = k['smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_wait_per_issue_active.ratio']
            kernel_cpi_res['CompStruct'] = k['smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio']
            kernel_cpi_res['NotSelect'] = k['smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio']
            kernel_cpi_res['NoStall'] = k['smsp__average_warps_issue_stalled_selected_per_issue_active.ratio']
            kernel_cpi_res['Sync'] = k['smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_membar_per_issue_active.ratio']
            kernel_cpi_res['Misc'] = k['smsp__average_warps_issue_stalled_misc_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio'] + k['smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio']
            # warpgroup_arrive
            
            kernel_cpi_res['Idle'] = 0
            kernel_cpi_res['debug'] = {}
            kernel_cpi_res['debug']['sum'] = kernel_cpi_res['MemData'] + kernel_cpi_res['CompData'] + kernel_cpi_res['MemStruct'] + kernel_cpi_res['CompStruct'] + kernel_cpi_res['NotSelect'] + kernel_cpi_res['NoStall'] + kernel_cpi_res['Sync'] + kernel_cpi_res['Misc']
            kernel_cpi_res['debug']['smsp__average_warp_latency_per_inst_issued.ratio'] = k['smsp__average_warp_latency_per_inst_issued.ratio']
            
            # sort by gsi stall list
            kernel_cpi_res = {k: kernel_cpi_res[k] for k in gsi_stall_list}
            res_json[app_arg].append(kernel_cpi_res)
    return res_json

def state_to_cpi(state_dict):
    cpi_stack = {}
    total_cycle = sum(state_dict.values())
    total_inst = state_dict.get('NoStall', 1)  # avoid zero
    average_warp_cycle_per_inst = total_cycle/total_inst
    
    for state in state_dict:
        cpi_stack[state] = state_dict[state]/total_inst  
    cpi_stack['debug'] = {}
    cpi_stack['debug']['total_cycle'] = total_cycle
    cpi_stack['debug']['total_inst'] = total_inst
    cpi_stack['debug']['average_warp_cycle_per_inst'] = average_warp_cycle_per_inst
    return cpi_stack

def get_cpi_stack_list(state_dict_list):
    cpi_stack_list = [state_to_cpi(state_dict) for state_dict in state_dict_list]
    # convert to gsi
    def fill_gsi(cpi_stack):
        cpi_stack_new = {}
        for k in gsi_stall_list:
            cpi_stack_new[k] = cpi_stack.get(k, 0)
        return cpi_stack_new
    cpi_stack_list_new = [fill_gsi(cpi_stack) for cpi_stack in cpi_stack_list]
    def avg_dict(dict_list):
        avg_dict = {}
        for k in dict_list[0]:
            avg_dict[k] = sum([d[k] for d in dict_list])/len(dict_list)
        return avg_dict
    # append avg dict to last
    cpi_stack_list_new.append(avg_dict(cpi_stack_list_new))
    return cpi_stack_list_new
            
def convert_ppt_gpu_to_gsi(data, select):
    res_json = {}
    for app_arg, app_res in data.items():
        print(f"{app_arg}: {len(app_res)}")
        res_json[app_arg] = []
        for i, kernel_res in enumerate(app_res):
            if select == "warp":
                cpi_stack_list = get_cpi_stack_list(kernel_res['warp_stats']['stall_types'])
            else:
                cpi_stack_list = get_cpi_stack_list(kernel_res['scheduler_stats']['stall_types'])
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
                        choices=["gsi"],
                        default="gsi",
                        help="classifiy model")
    parser.add_argument("-I", "--in-type",
                        choices=["ncu", "ppt_gpu", "ppt_gpu_sched"],
                        help="input json format")
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    if args.in_type == "ncu":
        output_json = convert_ncu_to_gsi(data)
    elif args.in_type == "ppt_gpu":
        output_json = convert_ppt_gpu_to_gsi(data, "warp")
    elif args.in_type == "ppt_gpu_sched":
        output_json = convert_ppt_gpu_to_gsi(data, "sched")
    else:
        raise NotImplementedError(f"Unknown input type: {args.in_type}")
    
    with open(args.output, 'w') as f:
        json.dump(output_json, f, indent=4)
