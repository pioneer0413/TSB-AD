# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging, psutil
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
from TSB_AD.snn.params import running_params
from TSB_AD.snn.utils import get_last_number, measure_block
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
사용법:

python benchmark_exp/Run_Detector_M.py --AD_Name <your_ad_name> --Encoder_Name [your_encoder_name] --postfix [your_postfix]

참고 및 권장사항:

1. 결과 파일의 기본 양식은 다음과 같음: <id-code>_<AD_Name>_<Encoder_Name>_<postfix>.csv
2. postfix에 여러 정보를 담아야 하는 경우, 구분자인 '_'(underscore)와 구분하기 위해 '-'(hyphen)로(예: 'postfix1-postfix2') 사용하는 것을 권장
 2.1. 가급적 postfix에 세 자리 이상 연속된 숫자를 사용하지 않는 것을 권장 (ID 코드와 혼동될 수 있음)
3. AD_Name이 ParallelSNN인 경우를 제외하면 Encoder_Name은 사용할 필요 없으며, 미입력 시 None으로 설정됨
4. !!중요!! 주요 파라미터 기본값은 TSB_AD/snn/params.py에 정의되어 있음
5. dataset_name은 Datasets/File_List/TSB-AD-M-<keyword>-Eva.csv의 keyword와 일치해야 함(대소문자 구분 필요)

예시:

python benchmark_exp/Run_Detector_M.py --AD_Name CNN --postfix kpca-kmeans --dataset_name MSL
python benchmark_exp/Run_Detector_M.py --AD_Name ParallelSNN --Encoder_Name receptive --postfix k3n3n3
python benchmark_exp/Run_Detector_M.py --AD_Name ParallelSNN --Encoder_Name repeat --num_enc_features 1 --dataset_name GHL --batch_size 256 --postfix enc_features_1
'''
if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')

    # data
    parser.add_argument('--data_path', type=str, help='Path to the input data file.')
    parser.add_argument('--dataset_name', type=str, default='Tiny')
    parser.add_argument('--channel_swap', action='store_true', default=running_params['data']['swap'])
    parser.add_argument('--channel_shuffle', action='store_true', default=running_params['data']['shuffle'])
    parser.add_argument('--normalize', action='store_true', default=running_params['data']['normalize'], help='Apply channel-wise normalization.')
    parser.add_argument('--drop', action='store_true', default=running_params['data']['drop'], help='Apply PCA to drop less important channels.')
    parser.add_argument('--zero_pruning', action='store_true', default=False, help='Apply zero pruning to remove channels with zero variance.')
    parser.add_argument('--mask', type=str, default=None, help='Channel mask to apply. Format: "110" for channels 1 and 2 active.')

    # meta
    parser.add_argument('--AD_Name', type=str, required=True)
    parser.add_argument('--Encoder_Name', type=str, default=None, choices=['conv', 'repeat', 'delta', 'receptive', 'dynamic'])
    parser.add_argument('--postfix', type=str, default='None')

    # model-common
    parser.add_argument('--device_type', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=running_params['model']['batch_size'])
    parser.add_argument('--window_size', type=int, default=running_params['model']['window_size'])
    parser.add_argument('--max_epochs', type=int, default=running_params['model']['max_epochs'], help='Maximum number of epochs for training.')

    # model-specific(CNN)
    parser.add_argument('--perf_cuda', action='store_true', default=running_params['CNNModel']['cuda'], help='Use CUDA for performance evaluation.')
    parser.add_argument('--n_components', type=int, default=running_params['CNNModel']['n_components'], help='Number of diviends components for PCA.')
    parser.add_argument('--kernel', type=str, default=running_params['CNNModel']['kernel'], choices=['linear', 'rbf', 'sigmoid', 'poly'], help='Kernel type for SVM.')
    parser.add_argument('--n_clusters', type=int, default=running_params['CNNModel']['n_clusters'], help='Number of diviends for KMeans.')

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
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    local_running_params['data']['result_dir'] = result_dir
    local_running_params['data']['swap'] = args.channel_swap
    local_running_params['data']['shuffle'] = args.channel_shuffle
    local_running_params['data']['normalize'] = args.normalize
    local_running_params['data']['drop'] = args.drop
    local_running_params['data']['zero_pruning'] = args.zero_pruning
    local_running_params['data']['mask'] = args.mask
    # Metadata
    local_running_params['meta']['AD_Name'] = args.AD_Name
    local_running_params['meta']['Encoder_Name'] = args.Encoder_Name
    local_running_params['meta']['postfix'] = args.postfix
    # model-common
    local_running_params['model']['device_type'] = args.device_type
    local_running_params['model']['batch_size'] = args.batch_size
    local_running_params['model']['window_size'] = args.window_size
    local_running_params['model']['max_epochs'] = args.max_epochs
    # model-specific(CNN)
    local_running_params['CNNModel']['cuda'] = args.perf_cuda
    local_running_params['CNNModel']['n_components'] = args.n_components
    local_running_params['CNNModel']['kernel'] = args.kernel
    local_running_params['CNNModel']['n_clusters'] = args.n_clusters
    # model-specific(ParallelSNN)
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

    # 파라미터가 올바른지 확인
    if args.skip is False:
        if input("Are the parameters correct? ([y]/n): ").strip().lower() not in ['y', '']:
            print("Parameters are not correct. Exiting...")
            os.remove(log_file_path)
            exit()

    file_list = pd.read_csv(local_running_params['data']['file_list'])['file_name'].values
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]

    '''
    Performance 기록용
    '''
    ad_name = args.AD_Name.lower()
    use_cuda = 'cudaO' if local_running_params['CNNModel']['cuda'] else 'cudaX'
    save_path = f'/home/hwkang/dev-TSB-AD/TSB-AD/analyses/performance/{id_code}_{ad_name}_kpca_{lower_name}_{use_cuda}_perf.csv'
    process = psutil.Process(os.getpid())

    '''
    mask 설정
    '''
    channel_mask = [int(b) for b in local_running_params['data']['mask']]
    if local_running_params['data']['mask'] is not None:
        file_list = [args.data_path]

    write_csv = []
    for filename in file_list:
        
        file_path = os.path.join(local_running_params['data']['dataset_dir'], filename)
        if local_running_params['data']['mask'] is not None:
            # mask가 설정된 경우, 해당 파일만 사용
            file_path = args.data_path
            filename = os.path.basename(file_path)
        df = pd.read_csv(file_path).dropna()

        if local_running_params['data']['normalize']:
            # channel-wise normalization
            df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / (df.iloc[:, :-1].std() + 1e-8)

        '''
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
        '''

        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        if local_running_params['data']['mask'] is not None:
            bit_seq = local_running_params['data']['mask']
            selected_indices = [i for i, bit in enumerate(bit_seq) if bit == '1']
            data = data[:, selected_indices]

        if local_running_params['data']['zero_pruning']:
            selected_indices = [1, 2, 9] # Genesis
            #selected_indices = [-2] # CreditCard
            data = data[:, selected_indices]

        if local_running_params['data']['drop']:
            C = data.shape[1]
            # Kernel PCA with all C channels
            arg_n_components = max(C-1, local_running_params['CNNModel']['n_components'])
            kpca = KernelPCA(n_components=arg_n_components, kernel=local_running_params['CNNModel']['kernel'], gamma=1.0, fit_inverse_transform=False)

            # Measure Performace of KPCA fit
            _, (w_time, c_time, m_usage)= measure_block(process, kpca.fit, data, block_name="KPCA.fit")
            kpca_fit_tup = (w_time, c_time, m_usage)

            kpca_pytorch = convert(kpca, 'torch')
            device = get_gpu(cuda=True)
            kpca_pytorch.to(device)

            start_time = time.time()
            cpu_start = process.cpu_times()
            mem_start = process.memory_info().rss

            if use_cuda == 'cudaO':
                try:
                    base_transformed = kpca_pytorch.transform(data)
                except:
                    print("Using sklearn KernelPCA")
                    base_transformed = kpca.transform(data)
            else:
                base_transformed = kpca.transform(data)
            
            end_time = time.time()
            cpu_end = process.cpu_times()
            mem_end = process.memory_info().rss

            elapsed_time = end_time - start_time
            cpu_time = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
            mem_change = (mem_end - mem_start) / (1024 ** 2)

            kpca_transform_tup = (elapsed_time, cpu_time, mem_change)

            # 중요도 추정: 각 채널의 영향도를 주성분의 projection에서 유도
            # kpca는 components_가 없음. 대신 eigenvectors_를 이용하거나 transformed 기반으로 channel별 중요도를 유추

            # 대체 방법: 각 채널을 제거해보며 출력 변화량 측정 (approximation)
            base_var = np.var(base_transformed, axis=0).sum()
            importances = []

            start_time = time.time()
            cpu_start = process.cpu_times()
            mem_start = process.memory_info().rss

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

            end_time = time.time()
            cpu_end = process.cpu_times()
            mem_end = process.memory_info().rss

            elapsed_time = end_time - start_time
            cpu_time = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
            mem_change = (mem_end - mem_start) / (1024 ** 2)

            kpca_importance_tup = (elapsed_time, cpu_time, mem_change)

            # cuda cleanup
            if use_cuda == 'cudaO':
                del kpca_pytorch
                torch.cuda.empty_cache()

            importances = np.array(importances)
            importances = importances / (np.sum(importances) + 1e-8)  # normalize

            # KMeans 클러스터링
            arg_n_clusters = max(C-1, local_running_params['CNNModel']['n_clusters'])
            kmeans = KMeans(n_clusters=arg_n_clusters, random_state=0, n_init=10)
            labels, (w_time, c_time, m_usage) = measure_block(process, kmeans.fit_predict, importances.reshape(-1, 1), block_name="KMeans.fit_predict")

            kmeans_fit_tup = (w_time, c_time, m_usage)

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
        cpu_start = process.cpu_times()
        mem_start = process.memory_info().rss

        if args.AD_Name in Semisupervise_AD_Pool:
            output = run_Semisupervise_AD(data_train=data_train, data_test=data, TS_Name=filename, local_running_params=local_running_params, **Optimal_Det_HP)
        elif args.AD_Name in Unsupervise_AD_Pool:
            output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)

        cpu_end = process.cpu_times()
        mem_end = process.memory_info().rss
        end_time = time.time()

        elapsed_time = end_time - start_time
        cpu_time = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        mem_change = (mem_end - mem_start) / (1024 ** 2)

        model_tup = (elapsed_time, cpu_time, mem_change)

        if local_running_params['data']['drop']:
            record_dict = {
                    'file_name': filename,
                    'length': data.shape[0],
                    'kpca_fit_time': kpca_fit_tup[0],
                    'kpca_transform_time': kpca_transform_tup[0],
                    'importance_time': kpca_importance_tup[0],
                    'kmeans_time': kmeans_fit_tup[0],
                    'model_time': model_tup[0],
                    'kpca_fit_cpu_time': kpca_fit_tup[1],
                    'kpca_transform_cpu_time': kpca_transform_tup[1],
                    'importance_cpu_time': kpca_importance_tup[1],
                    'kmeans_cpu_time': kmeans_fit_tup[1],
                    'model_cpu_time': model_tup[1],
                    'kpca_fit_mem': kpca_fit_tup[2],
                    'kpca_transform_mem': kpca_transform_tup[2],
                    'importance_mem': kpca_importance_tup[2],
                    'kmeans_mem': kmeans_fit_tup[2],
                    'model_mem': model_tup[2],
                }
            
            # if there is no saved csv file, create a new one
            if not os.path.exists(save_path):
                # 'file_name' is the index column
                save_df = pd.DataFrame(columns=record_dict.keys())
                save_df.set_index('file_name', inplace=True)
            else:
                # append to the existing csv file
                save_df = pd.read_csv(save_path, index_col='file_name')

            row_df = pd.DataFrame([record_dict])
            row_df.set_index('file_name', inplace=True)
            save_df = pd.concat([save_df, row_df], axis=0)
            save_df.to_csv(save_path)

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

        if local_running_params['data']['mask'] is not None:
            tokens = filename.split('.')[0].split('_')
            ts_name = '_'.join(tokens[:2])
            # save_dir에 가장 하위 디렉터리를 제외하고 mask를 추가한 디렉터리 생성
            temp = local_running_params['data']['result_dir'].split('/')[:-1]
            save_dir = '/'.join(temp) + '/mask'
            save_path = os.path.join(save_dir, f'{ts_name}.csv')
            # if there is no directory, create it
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # remove the first element (filename) and insert mask at the beginning
            record = record[1:]  # remove filename
            record.insert(0, local_running_params['data']['mask'])
            col_w = ['mask', 'Time'] + list(evaluation_result.keys())
            row_df = pd.DataFrame([record], columns=col_w)
            if not os.path.exists(save_path):
                row_df.to_csv(save_path, index=False)
            else:
                row_df.to_csv(save_path, mode='a', header=False, index=False)

        else:
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