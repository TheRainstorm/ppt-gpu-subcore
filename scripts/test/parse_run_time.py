

import argparse
import re
from datetime import datetime
import sys

def caculate_delta(time1, time2):
    # 将字符串解析为 datetime 对象
    dt1 = datetime.strptime(time1, "%Y-%m-%d %H:%M:%S")
    dt2 = datetime.strptime(time2, "%Y-%m-%d %H:%M:%S")

    # 计算时间差
    time_diff = dt2 - dt1

    # 提取小时、分钟和秒
    hours, remainder = divmod(time_diff.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    # 格式化输出
    return time_diff.total_seconds(), f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def parse_run_time(file_path, sort=True):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # get content between 'START' and 'END' which is also the last
    content = content.split('START')[-1].split('END')[0]
    
    app_arg_times = re.findall(r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}): (?P<app_arg>.*) (?P<status>start|finished)', content)
    
    faild_app_args = []
    res = {}
    start_ts = {}
    end_ts = {}
    app_args = []
    for app_arg_time in app_arg_times:
        ts = app_arg_time[0]
        app_arg = app_arg_time[1]
        status = app_arg_time[2]
        if status == 'start':
            start_ts[app_arg] = ts
        else:
            end_ts[app_arg] = ts
        if app_arg not in app_args:
            app_args.append(app_arg)
    for app_arg in app_args:
        try:
            delta, delta_str = caculate_delta(start_ts[app_arg], end_ts[app_arg])
            res[app_arg] = (delta, delta_str, start_ts[app_arg], end_ts[app_arg])
        except:
            # print stderr
            print(f"Warnning: {app_arg} has no start or finished", file=sys.stderr)
            
            faild_app_args.append(app_arg)
    
    if sort:
        sorted_res = sorted(res.items(), key=lambda x: x[1][0], reverse=True)
    else:
        sorted_res = res.items()
    return sorted_res, faild_app_args

def print_run_time(run_time):
    for app_arg, (delta, delta_str, start_ts, end_ts) in run_time:
        print(f"{app_arg:50s}: {delta_str:20s} ({start_ts} -> {end_ts})")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, required=True)
    parser.add_argument('-s', '--sort', action='store_true')
    args = parser.parse_args()
    run_time, faild = parse_run_time(args.file_path, sort=args.sort)
    print_run_time(run_time)
    print(f"Faild: {faild}", file=sys.stderr)
