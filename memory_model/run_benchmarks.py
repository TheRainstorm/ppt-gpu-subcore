import argparse

import json
import os
import sys

from memory_model_warper import memory_model_warpper, BlockMapping, get_parser
from scripts.common import *

parser = get_parser()
args = parser.parse_args()
if args.use_sm_trace:
        args.block_mapping = BlockMapping.sm_block_mapping
        
apps = gen_apps_from_suite_list(args.benchmark_list)
app_and_arg_list = get_app_arg_list(apps)
args.apps = filter_app_list(app_and_arg_list, args.app_filter)

log_file = open(args.log_file, "a")
def logging(*args, **kwargs):
    args = (f"{now_timestamp()}: ", ) + args
    print(*args, **kwargs, file=log_file, flush=True)
    print(*args, **kwargs, file=sys.stderr)

failed_list = []
logging(f"CMD: {' '.join(sys.argv)}")
print(f"filter apps: {args.apps}")
logging(f"START")

sim_res = {}
# if os.path.exists(args.output):
#     with open(args.output, 'r') as f:
#         try:
#             sim_res = json.load(f)
#         except:
#             pass

for app_and_arg in app_and_arg_list:
    app = app_and_arg.split('/')[0]
    app_trace_dir = os.path.join(args.trace_dir, app_and_arg)
    
    if args.apps and app_and_arg not in args.apps:
        continue
    
    logging(f"{app_and_arg} start")
    app_res, gpu_config = memory_model_warpper(args.config, app_trace_dir, args.model, granularity=args.granularity, use_approx=args.use_approx,
                        filter_L2=args.filter_l2, block_mapping=args.block_mapping,
                        l1_dump_trace=False, l2_dump_trace='', overwrite_cache_params=args.overwrite_cache_params,
                        no_adaptive_cache=args.no_adaptive_cache)
    avg_l1_hit_rate = sum([res['l1_hit_rate'] for res in app_res]) / len(app_res)
    avg_l2_hit_rate = sum([res['l2_hit_rate'] for res in app_res]) / len(app_res)
    sim_res[app_and_arg] = app_res
    logging(f"{app_and_arg} finished")

    sim_res['0'] = gpu_config
    with open(args.output, 'w') as f:
        json.dump(sim_res, f, indent=4)

logging(f"failed list: {failed_list}")
logging(f"End")
log_file.close()