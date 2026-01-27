import os
import numpy as np
import pandas as pd
import matplotlib
import scipy.io as scipyio
import matplotlib.pyplot as plt

from tqdm import tqdm

# Absolute path of the folder containing the units' folders and scenarioX.csv
scenario_folder = '/home/changyuan/wireless_context/scenario9_dev'

# Fetch scenario CSV
try:
    csv_file = [f for f in os.listdir(scenario_folder) if f.endswith('csv')][0]
    csv_path = os.path.join(scenario_folder, csv_file)
except:
    raise Exception(f'No csv file inside {scenario_folder}.')

# Load CSV to dataframe
dataframe = pd.read_csv(csv_path)


import os
import scipy.io as scipyio
import numpy as np
import pandas as pd

# 假设前面的 dataframe 和 scenario_folder 已经定义好了
# 我们随机取第一行数据的 lidar 路径来分析
sample_idx = 0 
lidar_rel_path = dataframe['unit1_lidar_SCR'].values[sample_idx]
lidar_abs_path = os.path.join(scenario_folder, lidar_rel_path)

print(f"Checking file: {lidar_rel_path}")
print("-" * 30)

# 1. 加载整个 mat 文件
mat_data = scipyio.loadmat(lidar_abs_path)

# 2. 遍历打印 mat 文件里的所有变量
for key, value in mat_data.items():
    # 过滤掉系统自带的元数据 (以 __ 开头的通常是 header, version, globals)
    if key.startswith('__'):
        continue
        
    print(f"Variable Name: '{key}'")
    print(f"   Type: {type(value)}")
    
    # 如果是 numpy 数组，打印形状
    if isinstance(value, np.ndarray):
        print(f"   Shape: {value.shape}")
        print(f"   Data Sample (前5个值): {value.flatten()[:5]}")
    else:
        print(f"   Value: {value}")
    print("-" * 30)