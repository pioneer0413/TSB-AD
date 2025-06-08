import subprocess
import os

src_dataset_path = '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M'

file_names = os.listdir(src_dataset_path)
file_names.sort()

for file_name in file_names:
    # run subprocess command
    command = f'python benchmark_exp/Run_Custom_Detector.py --filename {file_name}'
    print(f'file_name: {file_name}')
    try:
        subprocess.run(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f'Error occurred while processing {file_name}: {e}')
        break  # 중단