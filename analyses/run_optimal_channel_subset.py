import os
import pandas as pd
# subprocess
import sys
import subprocess
import argparse

def channel_mask(num_channels):
    total = 2 ** num_channels
    masks = [format(i, f'0{num_channels}b') for i in range(1, total)]
    # remove zero-only mask
    return masks[0:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimal channel subset analysis.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file.')
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    data = df.iloc[:, :-1].values

    num_channels = data.shape[1]
    masks = channel_mask(num_channels)

    root_dir_path = '/home/hwkang/dev-TSB-AD'
    target_script = f'{root_dir_path}/TSB-AD/benchmark_exp/Run_Detector_M.py'

    dataset_name = args.data_path.split('/')[-1].split('.')[0].split('_')[1]
    ad_name = 'CNN'
    max_epochs = 5

    for mask in masks:
        postfix = f'mask-{mask}'

        command = [
            'python', target_script,
            '--data_path', args.data_path,
            '--dataset_name', dataset_name,
            '--AD_Name', ad_name,
            '--max_epochs', str(max_epochs),
            '--mask', mask,
            '--postfix', postfix,
            '--skip',
        ]

        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {result.stderr}")
        else:
            print(f"Command output: {result.stdout}")
