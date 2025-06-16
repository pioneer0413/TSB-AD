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
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from hummingbird.ml import convert
from TSB_AD.utils.torch_utility import get_gpu

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
python benchmark_exp/Run_Detector_M.py --AD_Name ParallelSNN --Encoder_Name dynamic --postfix
python benchmark_exp/Run_Detector_M.py --AD_Name ParallelSNN --Encoder_Name receptive --postfix
python benchmark_exp/Run_Detector_M.py --AD_Name ParallelSNN --Encoder_Name conv --postfix
python benchmark_exp/Run_Detector_M.py --AD_Name ParallelSNN --Encoder_Name delta --postfix
python benchmark_exp/Run_Detector_M.py --AD_Name ParallelSNN --Encoder_Name repeat --num_enc_features 1 --dataset_name GHL --batch_size 256 --postfix enc_features_1
python benchmark_exp/Run_Detector_M.py --AD_Name CNN --postfix
python benchmark_exp/Run_Detector_M.py --AD_Name SEWResNet --Encoder_Name conv --postfix
'''

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')

    # data
    parser.add_argument('--dataset_name', type=str, default='Tiny')
    parser.add_argument('--channel_swap', action='store_true', default=running_params['data']['swap'])
    parser.add_argument('--channel_shuffle', action='store_true', default=running_params['data']['shuffle'])
    parser.add_argument('--normalize', action='store_true', default=running_params['data']['normalize'], help='Apply channel-wise normalization.')
    parser.add_argument('--drop', action='store_true', default=running_params['data']['drop'], help='Apply PCA to drop less important channels.')

    # meta
    parser.add_argument('--AD_Name', type=str, required=True)
    parser.add_argument('--Encoder_Name', type=str, default=None, choices=['conv', 'repeat', 'delta', 'receptive', 'dynamic'])
    parser.add_argument('--postfix', type=str, default='None')

    # model-common
    parser.add_argument('--device_type', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=running_params['model']['batch_size'])
    parser.add_argument('--window_size', type=int, default=running_params['model']['window_size'])

    # model-specific(ParallelSNN)
    parser.add_argument('--num_enc_features', type=int, default=int(running_params['ParallelSNNModel']['num_enc_features']))
    parser.add_argument('--norm_type', type=str, default=running_params['ParallelSNNModel']['norm_type'], choices=['bn', 'ln'])
    parser.add_argument('--dropout', action='store_true', default=running_params['ParallelSNNModel']['dropout'])
    parser.add_argument('--encoding_kernel', type=str, default='3n3n3', help='Format: "5n11n5" for [5, 11, 5]')
    parser.add_argument('--tt', action='store_true', default=running_params['ParallelSNNModel']['tt'])
    parser.add_argument('--delta_abs', action='store_true', default=running_params['ParallelSNNModel']['delta_abs'])
    parser.add_argument('--grad_spike', action='store_true', default=running_params['ParallelSNNModel']['grad_spike'])

    # check skip
    parser.add_argument('--skip', action='store_true', default=False, help='Skip the confirmation of parameters before running the detector.')

    args = parser.parse_args()

    # Reset Independent Variable of running_params
    local_running_params = running_params.copy()

    # Data
    file_list = local_running_params['data']['file_list']
    src_file_name = file_list.split('/')[-1]
    target_file_name = f'TSB-AD-M-{args.dataset_name}-Eva.csv'
    file_list = file_list.replace(src_file_name, target_file_name)
    local_running_params['data']['file_list'] = file_list

    result_dir = local_running_params['data']['result_dir']
    src_name = result_dir.split('/')[-1]
    lower_name = args.dataset_name.lower()
    result_dir = result_dir.replace(src_name, lower_name)
    local_running_params['data']['result_dir'] = result_dir

    local_running_params['data']['swap'] = args.channel_swap
    local_running_params['data']['shuffle'] = args.channel_shuffle
    local_running_params['data']['normalize'] = args.normalize
    local_running_params['data']['drop'] = args.drop
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
    if args.skip is False:
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

        if local_running_params['data']['normalize']:
            # channel-wise normalization
            df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / (df.iloc[:, :-1].std() + 1e-8)


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

        if local_running_params['data']['drop']:
            # PCA
            '''
            C = data.shape[1]
            variance_threshold = 0.9
            pca_full = PCA(n_components=C)
            pca_full.fit(data)
            # 누적 설명력 계산
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            k = np.searchsorted(cumsum, variance_threshold) + 1

            # 상위 k개의 PC만 사용
            components = pca_full.components_[:k, :]                  # shape: (k, C)
            weights = pca_full.explained_variance_ratio_[:k]          # shape: (k,)

            # 채널별 importance 계산 (가중 합)
            importance = np.sum(np.abs(components) * weights[:, np.newaxis], axis=0)
            importance = importance / np.sum(importance)  # normalize to sum = 1

            # KMeans 클러스터링 (2개 클러스터)
            kmeans = KMeans(n_clusters=2, random_state=0)
            labels = kmeans.fit_predict(importance.reshape(-1, 1))  # shape: (C,)

            # sort indices
            indices = np.argsort(importance)[::-1]

            # 가장 중요한 채널이 속한 클러스터의 채널만 선택
            selected_indices = indices[labels[indices] == labels[indices[0]]]'''

            C = data.shape[1]
            # Kernel PCA with all C channels
            kpca = KernelPCA(n_components=C, kernel='rbf', gamma=1.0, fit_inverse_transform=False)
            kpca.fit(data)
            kpca_pytorch = convert(kpca, 'torch')
            device = get_gpu(cuda=True)
            kpca_pytorch.to(device)

            try:
                base_transformed = kpca_pytorch.transform(data)
            except:
                print("Using sklearn KernelPCA")
                base_transformed = kpca.transform(data)

            # 중요도 추정: 각 채널의 영향도를 주성분의 projection에서 유도
            # kpca는 components_가 없음. 대신 eigenvectors_를 이용하거나 transformed 기반으로 channel별 중요도를 유추

            # 대체 방법: 각 채널을 제거해보며 출력 변화량 측정 (approximation)
            base_var = np.var(base_transformed, axis=0).sum()
            importances = []

            for c in range(C):
                masked_data = data.copy()
                masked_data[:, c] = 0  # 채널 c 제거
                try:
                    try:
                        transformed_masked = kpca_pytorch.transform(masked_data)
                    except:
                        print("Using sklearn KernelPCA for masked data")
                        transformed_masked = kpca.transform(masked_data)
                    masked_var = np.var(transformed_masked, axis=0).sum()
                    importance = base_var - masked_var  # 제거 시 variance 감소량
                except:
                    importance = 0.0
                importances.append(importance)

            # cuda cleanup
            del kpca_pytorch
            torch.cuda.empty_cache()

            importances = np.array(importances)
            importances = importances / (np.sum(importances) + 1e-8)  # normalize

            # KMeans 클러스터링
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
            labels = kmeans.fit_predict(importances.reshape(-1, 1))  # shape: (C,)

            # 가장 중요한 채널이 속한 클러스터 선택
            sorted_indices = np.argsort(importances)[::-1]
            important_label = labels[sorted_indices[0]]
            selected_indices = np.where(labels == important_label)[0]

            # 선택된 채널만 남기기
            data = data[:, selected_indices]

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