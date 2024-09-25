import argparse

import subprocess
import shlex
import re
import os
import yaml
import hashlib
from datetime import datetime

parser = argparse.ArgumentParser(
    description='Simulate all app defined'
)
parser.add_argument("-B", "--benchmark_list",
                 help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for the benchmark suite names.",
                 default="rodinia_2.0-ft")
parser.add_argument("--apps",
                    nargs="*",
                    help="only run specific apps")
parser.add_argument("-Y", "--benchmarks_yaml",
                    required=True,
                    help='benchmarks_yaml path')
parser.add_argument("-T", "--trace_dir",
                    required=True,
                    help="The root of all the trace file")
parser.add_argument("-D", "--device_num",
                 help="CUDA device number",
                 default="0")
parser.add_argument("-t", "--loop_cnt",
                 type=int,
                 default=3,
                 help="run multiple times")
parser.add_argument("-l", "--log_file",
                    default="run_hw.log",
                    )
parser.add_argument("-n", "--norun",
                 action="store_true")
parser.add_argument("-r", "--run_script",
                 default="run_profiling.sh")
parser.add_argument("-o", "--profiling_filename",
                 default="profiling.csv")
parser.add_argument("--no-overwrite", dest="overwrite",
                 action="store_false",
                 help="if overwrite=False, then don't profile when cvs exist")
parser.add_argument("--ncu",
                 action="store_true",
                 help="use ncu to profile")
args = parser.parse_args()

from common import *

defined_apps = {}
parse_app_definition_yaml(args.benchmarks_yaml, defined_apps)
apps = gen_apps_from_suite_list(args.benchmark_list.split(","), defined_apps)

log_file = open(args.log_file, "a")
def logging(*args, **kwargs):
    print(*args, **kwargs, file=log_file, flush=True)
