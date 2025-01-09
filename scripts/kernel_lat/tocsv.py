import argparse
import json
import re
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument("-i", "--res", required=True, help="ncu json file")
    parser.add_argument("-o", "--output", default="kernel_lat.xlsx", help="kernel result file")
    args = parser.parse_args()
    
    with open(args.res) as f:
        res = json.load(f)
    
    data = []
    for app_arg, app_res in res.items():
        kernel_name, params = app_arg.split("/")
        if 'kernel_lat' not in kernel_name:
            continue
        m = re.search(r'(\d+)_+(\d+)', params)
        if m:
            grid_size, block_size = map(int, m.groups())
        else:
            exit(1)
            
        cycle = int(app_res[0]['gpc__cycles_elapsed.max'])
        
        data.append({
            'grid_size': grid_size,
            'block_size': block_size,
            'cycle': cycle
        })
    
    df = pd.DataFrame(data, columns=["block_size", "grid_size", "cycle"])
    df.sort_values(['block_size', 'grid_size'], axis=0, ascending=[True, True], inplace=True, kind='quicksort', na_position='last', ignore_index=False, key=None)
    df.to_excel(args.output, sheet_name=f'raw', index=False)
    print(f"output to {args.output}")