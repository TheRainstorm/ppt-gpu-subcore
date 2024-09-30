import yaml

# grid_sizes = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984, 2016, 2048, ]
# block_sizes = [32, 64, 128, 256, 512, 1024]

grid_sizes = [32, 64, 128, 256, 512, 1024]
block_sizes = [32, 64, 128, 256, 512, 1024]

data = {}

ubench = {}
ubench['exec_dir'] = "$UBENCH_ROOT/bin/"
ubench['data_dirs'] = "$GPUAPPS_ROOT/data_dirs/"
ubench['execs'] = []
data['ubench'] = ubench

kernel_lat = {}
kernel_lat['kernel_lat'] = []
ubench['execs'].append(kernel_lat)

arg_list = kernel_lat['kernel_lat']
for grid_size in grid_sizes:
    for block_size in block_sizes:
        arg = {}
        arg['args'] = f"{grid_size} {block_size}"
        arg_list.append(arg)

with open('kernel_lat.yml', 'w') as f:
    yaml.dump(data, f)