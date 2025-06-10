# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
from TSB_AD.snn.params import running_params
from TSB_AD.snn.utils import get_last_number

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

'''
python benchmark_exp/Run_Detector_M.py --AD_Name ParallelSNN --Encoder_Name receptive --postfix
python benchmark_exp/Run_Detector_M.py --AD_Name ParallelSNN --Encoder_Name conv --postfix
python benchmark_exp/Run_Detector_M.py --AD_Name ParallelSNN --Encoder_Name delta --postfix
python benchmark_exp/Run_Detector_M.py --AD_Name ParallelSNN --Encoder_Name repeat --postfix
python benchmark_exp/Run_Detector_M.py --AD_Name CNN --postfix
'''

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')

    # data
    parser.add_argument('--channel_swap', action='store_true', default=running_params['data']['swap'])
    parser.add_argument('--channel_shuffle', action='store_true', default=running_params['data']['shuffle'])

    # meta
    parser.add_argument('--AD_Name', type=str, required=True)
    parser.add_argument('--Encoder_Name', type=str, default=None, choices=['conv', 'repeat', 'delta', 'receptive', 'receptive2', 'gac'])
    parser.add_argument('--postfix', type=str, default='None')

    # model-common
    parser.add_argument('--device_type', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=running_params['model']['batch_size'])
    parser.add_argument('--window_size', type=int, default=running_params['model']['window_size'])

    # model-specific(ParallelSNN)
    parser.add_argument('--num_enc_features', type=int, default=int(running_params['ParallelSNNModel']['num_enc_features']))
    parser.add_argument('--norm_type', type=str, default=running_params['ParallelSNNModel']['norm_type'], choices=['bn', 'ln'])
    parser.add_argument('--dropout', action='store_true', default=running_params['ParallelSNNModel']['dropout'])
    parser.add_argument('--encoding_kernel', type=str, default='5n11n5', help='Format: "5n11n5" for [5, 11, 5]')
    parser.add_argument('--tt', action='store_true', default=running_params['ParallelSNNModel']['tt'])
    parser.add_argument('--delta_abs', action='store_true', default=running_params['ParallelSNNModel']['delta_abs'])
    parser.add_argument('--grad_spike', action='store_true', default=running_params['ParallelSNNModel']['grad_spike'])

    args = parser.parse_args()

    # Reset Independent Variable of running_params
    local_running_params = running_params.copy()

    # Data
    local_running_params['data']['swap'] = args.channel_swap
    local_running_params['data']['shuffle'] = args.channel_shuffle
    # Metadata
    local_running_params['meta']['AD_Name'] = args.AD_Name
    local_running_params['meta']['Encoder_Name'] = args.Encoder_Name
    local_running_params['meta']['postfix'] = args.postfix
    # model-common
    local_running_params['model']['device_type'] = args.device_type
    local_running_params['model']['batch_size'] = args.batch_size
    local_running_params['model']['window_size'] = args.window_size
    # model-specific
    local_running_params['ParallelSNNModel']['num_enc_features'] = args.num_enc_features
    local_running_params['ParallelSNNModel']['norm_type'] = args.norm_type
    local_running_params['ParallelSNNModel']['dropout'] = args.dropout
    local_running_params['ParallelSNNModel']['encoding_kernel'] = [int(x) for x in args.encoding_kernel.split('n')]
    local_running_params['ParallelSNNModel']['tt'] = args.tt
    local_running_params['ParallelSNNModel']['delta_abs'] = args.delta_abs
    local_running_params['ParallelSNNModel']['grad_spike'] = args.grad_spike

    # 로그 파일 경로 설정
    log_dir_path = os.path.join(local_running_params['meta']['root_dir_path'], 'logs')
    id_code = get_last_number(os.listdir(log_dir_path))
    local_running_params['meta']['id_code'] = id_code
    local_running_params['meta']['base_file_name'] = f'{id_code:03d}_{args.AD_Name}_{args.Encoder_Name}_{args.postfix}'
    log_file_path = os.path.join(log_dir_path, local_running_params['meta']['base_file_name'] + '.log')

    with open(log_file_path, 'w') as f:
        f.write('Running Configurations:\n')
        for key, value in local_running_params.items():
            f.write(f'{key}: {value}\n')
            print(f'{key}: {value}')

        # 실행 시작 시점 기록
        f.write('\n')
        f.write('Execution Start Time: {}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    # Check whether parameters are same as intended
    # get standard input from console
    # if input is 'y' or 'enter', continue the execution else remove log file and exit
    if input("Are the parameters correct? ([y]/n): ").strip().lower() not in ['y', '']:
        print("Parameters are not correct. Exiting...")
        os.remove(log_file_path)
        exit()

    file_list = pd.read_csv(local_running_params['data']['file_list'])['file_name'].values
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]

    write_csv = []
    for filename in file_list:
        
        file_path = os.path.join(local_running_params['data']['dataset_dir'], filename)
        df = pd.read_csv(file_path).dropna()

        assert not (local_running_params['data']['swap'] is True and local_running_params['data']['shuffle'] is True), "'swap' cannot co-exist with 'shuffle'."

        if local_running_params['data']['swap']:
            # Channel order alignment
            # Extract header from df
            header = df.columns.tolist()
            # If there exists 'HT_temperature.T' and 'C_temperature.T' in the header, swap order between them in df
            if 'HT_temperature.T' in header and 'C_temperature.T' in header:
                ht_index = header.index('HT_temperature.T')
                c_index = header.index('C_temperature.T')
                if ht_index > c_index:
                    # reorder the columns
                    new_order = header[:]
                    new_order[ht_index], new_order[c_index] = new_order[c_index], new_order[ht_index]
                    df = df[new_order]
                    # print new header
                    print("Before reordering columns:", header)
                    print("Reordered columns:", df.columns.tolist())

        # Shuffle channel order in df except for the last column
        if local_running_params['data']['shuffle']:
            original_columns = df.columns.tolist()
            columns = df.columns[:-1].tolist()
            random.shuffle(columns)
            df = df[columns + [df.columns[-1]]]
            print(f'Original columns: {original_columns}')
            print(f'Shuffled columns: {df.columns.tolist()}')
            
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        start_time = time.time()

        if args.AD_Name in Semisupervise_AD_Pool:
            output = run_Semisupervise_AD(data_train=data_train, data_test=data, TS_Name=filename, local_running_params=local_running_params, **Optimal_Det_HP)
        elif args.AD_Name in Unsupervise_AD_Pool:
            output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
        
        end_time = time.time()
        run_time = end_time - start_time

        if isinstance(output, np.ndarray):
            target_dir = os.path.join(local_running_params['data']['score_dir'], local_running_params['meta']['base_file_name'])
            os.makedirs(target_dir, exist_ok=True)
            np.save(target_dir+'/'+filename.split('.')[0]+'.npy', output)

        try:
            evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
            print('output: ', output.shape)
            print('label: ', label.shape)
            print('evaluation_result: ', evaluation_result)
            record = list(evaluation_result.values())
        except:
            record = [0]*9
        record.insert(0, run_time)
        record.insert(0, filename)

        ## Temp Save
        col_w = ['file', 'Time'] + list(evaluation_result.keys())
        csv_path = os.path.join(local_running_params['data']['result_dir'], local_running_params['meta']['base_file_name'] + '.csv')
        row_df = pd.DataFrame([record], columns=col_w)
        if not os.path.exists(csv_path):
            row_df.to_csv(csv_path, index=False)
        else:
            row_df.to_csv(csv_path, mode='a', header=False, index=False)
        
    # logging 설정
    with open(log_file_path, 'a') as f:
        # run_time 기록
        f.write('\n')
        f.write('End-to-End Run Time: {:.3f}s\n'.format(time.time() - Start_T))

    print(f"End-to-End Run Time: {time.time() - Start_T:.3f} seconds")