

import json

def get_and_print(file_path, get_diverge=False):
    with open(file_path) as fp:
        data = json.load(fp)

    for app, app_data in data.items():
        AMAT_list = []
        diverge_list = []
        for kernel_data in app_data:
            AMAT_list.append(kernel_data["AMAT"])
            if get_diverge:
                try:
                    diverge_list.append(kernel_data["diverge_flag"])
                except:
                    diverge_list.append(0)
        
        avg_AMAT = sum(AMAT_list)/len(AMAT_list)
        if get_diverge:
            avg_diverge = sum(diverge_list)/len(diverge_list)

        if get_diverge:
            print(f"{app} [{len(app_data)}]: {avg_AMAT:.2f}, {avg_diverge:.2f}({int(sum(diverge_list))}/{len(diverge_list)})")
        else:
            print(f"{app} [{len(app_data)}]: {avg_AMAT:.2f}")

get_and_print("res_ppt-gpu-AMAT.json", get_diverge=True)

print("GIMT")
get_and_print("../gpu-interval-model-tensor/tmp/res_GIMT-AMAT.json")
