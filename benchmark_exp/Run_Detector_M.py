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

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

root_dir_path = '/home/hwkang/dev-TSB-AD/TSB-AD/'

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M')
    parser.add_argument('--file_list', type=str, default='/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Tiny-Eva.csv')
    parser.add_argument('--score_dir', type=str, default='/home/hwkang/dev-TSB-AD/TSB-AD/eval/score/multi/')
    parser.add_argument('--save_dir', type=str, default='/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--AD_Name', type=str, required=True)
    parser.add_argument('--Encoder_Name', type=str, default=None, choices=['conv', 'repeat', 'delta', 'convmlp'])
    parser.add_argument('--postfix', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true', default=False) # False=skip, True=overwrite
    parser.add_argument('--off_cuda', action='store_true', default=False)

    # Load
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--load_file_code', type=str, default=None)

    # Visualization
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--save_encoding', action='store_true', default=False)
    parser.add_argument('--trace_threshold', action='store_true', default=False)
    parser.add_argument('--early_stop_off', action='store_false', default=True)

    # Independent Variables
    parser.add_argument('--activation_type', type=str, default='ternary', choices=['binary', 'ternary'])
    parser.add_argument('--learn_threshold', action='store_true', default=False)
    parser.add_argument('--granularity', type=str, default='neuron')
    parser.add_argument('--threshold_init', type=str, default='scalar')  # updated default value
    parser.add_argument('--bntt', action='store_true', default=False)
    parser.add_argument('--second_chance', action='store_true', default=False)

    # Adversarial
    parser.add_argument('--adversarial_type', type=str, default=None, choices=['fgsm', 'pgd'])

    args = parser.parse_args()

    # Reset Independent Variable of running_params
    local_running_params = running_params.copy()

    local_running_params['load'] = args.load
    local_running_params['postfix'] = args.postfix
    local_running_params['off_cuda'] = args.off_cuda

    local_running_params['verbose'] = args.verbose
    local_running_params['save_encoding'] = args.save_encoding
    local_running_params['trace_threshold'] = args.trace_threshold
    local_running_params['early_stop'] = args.early_stop_off

    local_running_params['activations']['activation'] = args.activation_type
    local_running_params['activations']['binary']['learn_threshold'] = args.learn_threshold
    local_running_params['activations']['binary']['granularity'] = args.granularity
    local_running_params['activations']['binary']['threshold_init'] = args.threshold_init
    local_running_params['activations']['binary']['bntt'] = args.bntt
    local_running_params['activations']['binary']['second_chance'] = args.second_chance

    local_running_params['adversarial']['type'] = args.adversarial_type

    if args.Encoder_Name is not None:
        target_dir = os.path.join(args.score_dir, args.AD_Name, args.Encoder_Name)
    else:
        target_dir = os.path.join(args.score_dir, args.AD_Name)

    if args.postfix is not None:
        target_dir = f'{target_dir}_{args.postfix}'
    os.makedirs(target_dir, exist_ok = True)

    # 같은 파일 코드 생성 방지를 위해 0.1~0.5초 사이의 랜덤한 숫자 생성
    random_num = random.uniform(0.1, 0.3)
    # random_num 동안 대기
    time.sleep(random_num)
    random.seed(seed)

    log_dir_path = os.path.join(root_dir_path, 'logs')
    os.makedirs(log_dir_path, exist_ok=True)
    # log_dir_path의 파일 리스트 가져오기
    log_files = os.listdir(log_dir_path)
    # log_dir_path에 있는 파일 중에서 가장 큰 숫자를 찾기
    max_num = 0
    id_code = 0
    if len(log_files) > 0:
        for file in log_files:
            segments = file.split('_')
            file_number = int(segments[0])
            if file_number > max_num:
                max_num = file_number
        file_number = max_num + 1
        # 새로운 파일 이름 생성
        if args.Encoder_Name is not None:
            file_name = f'{file_number:03d}_run_{args.AD_Name}_{args.Encoder_Name}_{args.postfix}.log'
        else:
            file_name = f'{file_number:03d}_run_{args.AD_Name}_{args.postfix}.log'
    else:
        file_number = 0
        # 새로운 파일 이름 생성
        if args.Encoder_Name is not None:
            file_name = f'{file_number:03d}_run_{args.AD_Name}_{args.Encoder_Name}_{args.postfix}.log'
        else:
            file_name = f'{file_number:03d}_run_{args.AD_Name}_{args.postfix}.log'
    log_file_path = os.path.join(log_dir_path, file_name)
    id_code = file_number

    with open(log_file_path, 'w') as f:
        f.write('Arguments:\n')
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
        
        f.write('\n')
        f.write('Running Configurations:\n')
        for key, value in local_running_params.items():
            f.write(f'{key}: {value}\n')

        # 실행 시작 시점 기록
        f.write('\n')
        f.write('Execution Start Time: {}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    if args.Encoder_Name is not None:
        logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}_{args.Encoder_Name}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    file_list = pd.read_csv(args.file_list)['file_name'].values
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]
    #print('Optimal_Det_HP: ', Optimal_Det_HP)

    write_csv = []
    for filename in file_list:
        if os.path.exists(target_dir+'/'+filename.split('.')[0]+'.npy') and args.overwrite is False : continue
        print('Processing:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        start_time = time.time()

        try:
            if args.AD_Name in Semisupervise_AD_Pool:

                parts = filename.split('_')
                
                if local_running_params['save']:
                    if args.Encoder_Name is not None:
                        file_name = f'{args.AD_Name}_{id_code:03d}_{args.Encoder_Name}_{args.postfix}_{parts[0]}_{parts[1]}.pt'
                    else:
                        file_name = f'{args.AD_Name}_{id_code:03d}_{args.postfix}_{parts[0]}_{parts[1]}.pt'
                    local_running_params['save_file_path'] = os.path.join(root_dir_path, 'weights', file_name)

                if local_running_params['load']:
                    if args.Encoder_Name is not None:
                        load_file_list = os.listdir(os.path.join(root_dir_path, 'weights'))
                        # load_file_list에서 AD_Name과 args.load_file_code, parts[0], parts[1]를 포함하는 파일 찾기
                        file_name = [f for f in load_file_list if args.AD_Name in f and args.load_file_code in f and parts[0] in f and parts[1] in f]
                    local_running_params['load_file_path'] = os.path.join(root_dir_path, 'weights', file_name[0])

                output = run_Semisupervise_AD(data_train=data_train, data_test=data, TS_Name=filename, AD_Name=args.AD_Name, Encoder_Name=args.Encoder_Name, local_running_params=local_running_params, **Optimal_Det_HP)
                
            elif args.AD_Name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
            else:
                raise Exception(f"{args.AD_Name} is not defined")
        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                last_call = tb[-1]
                print(f'Run_Detector_M error in {last_call.filename} at line {last_call.lineno}: {e}')
            else:
                print(f'Run_Detector_M error: {e}')
            # log_file_path에 에러 메시지 기록
            with open(log_file_path, 'a') as f:
                f.write(f'\nRun_Detector_M error in {last_call.filename} at line {last_call.lineno}: {e}\n')

        end_time = time.time()
        run_time = end_time - start_time

        if isinstance(output, np.ndarray):
            logging.info(f'Success at {filename} using {args.AD_Name} | Time cost: {run_time:.3f}s at length {len(label)}')
            np.save(target_dir+'/'+filename.split('.')[0]+'.npy', output)
        else:
            logging.error(f'At {filename}: '+output)

        ### whether to save the evaluation result
        if args.save:
            try:
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                #print('evaluation_result: ', evaluation_result)
                list_w = list(evaluation_result.values())
            except:
                list_w = [0]*9
            list_w.insert(0, run_time)
            list_w.insert(0, filename)
            write_csv.append(list_w)

            ## Temp Save
            col_w = list(evaluation_result.keys())
            col_w.insert(0, 'Time')
            col_w.insert(0, 'file')
            w_csv = pd.DataFrame(write_csv, columns=col_w)
            os.makedirs(args.save_dir, exist_ok=True)
            if args.Encoder_Name is None:
                if args.postfix is not None:
                    w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}_{id_code:03d}_{args.postfix}.csv', index=False)
                else:
                    w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}_{id_code:03d}.csv', index=False)
            else:
                if args.postfix is not None:
                    w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}_{id_code:03d}_{args.Encoder_Name}_{args.postfix}.csv', index=False)
                else:
                    w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}_{id_code:03d}_{args.Encoder_Name}.csv', index=False)

    # logging 설정
    with open(log_file_path, 'a') as f:
        # run_time 기록
        f.write('\n')
        f.write('Total Run Time: {:.3f}s\n'.format(time.time() - Start_T))