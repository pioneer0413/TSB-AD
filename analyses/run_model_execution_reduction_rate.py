import os, argparse, time
import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from TSB_AD.models.CNN import CNN
from TSB_AD.snn.params import running_params

def get_selected_indices(data, n_components, kernel, n_clusters):

    n_samples, n_features = data.shape
    #print(f"Number of samples: {n_samples}, Number of features: {n_features}")

    temp_n_components = max(1, n_features//n_components)
    #print(f"Using n_components: {n_components}, kernel: {kernel}, n_clusters: {n_clusters}")
    kpca = KernelPCA(n_components=temp_n_components, kernel=kernel, gamma=1.0, fit_inverse_transform=False)
    kpca.fit(data)

    # --- Block 2: KPCA transform ---
    base_transform = kpca.transform(data)
    base_var = np.var(base_transform, axis=0).sum()

    importances = []
    for selected_feature in range(n_features):
        masked_data = data.copy()
        masked_data[:, selected_feature] = 0
        try:
            transformed_masked = kpca.transform(masked_data)
            masked_var = np.var(transformed_masked, axis=0).sum()
            importance = base_var - masked_var
        except:
            importance = 0.0
        importances.append(importance)

    importances = np.array(importances)
    importances = importances / (np.sum(importances) + 1e-8)

    # --- Block 4: KMeans clustering ---
    temp_n_clusters = max(1, n_features // n_clusters)
    kmeans = KMeans(n_clusters=temp_n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(importances.reshape(-1, 1))

    # --- Post-processing ---
    sorted_indices = np.argsort(importances)[::-1]
    important_label = labels[sorted_indices[0]]
    selected_indices = np.where(labels == important_label)[0]

    return selected_indices, importances, labels, sorted_indices, important_label

'''
Usage:

python analyses/run_model_execution_reduction_rate.py [--use_kpca] [--dataset_name MSL|SMAP] [--measurement_type model_execution|reduction_rate] [--new]

Objectives:

- 제안 방법 적용 유무에 따른 '모델 실행 시간' 측정
- 제안 방법의 하이퍼파라미터 변화에 따른 '채널 축소율' 측정
'''
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='KPCA Model Execution')
    parser.add_argument('--use_kpca', action='store_true', default=False)
    parser.add_argument('--dataset_name', type=str, default='MSL')
    parser.add_argument('--measurement_type', type=str, default='model_execution', choices=['model_execution', 'reduction_rate'])
    parser.add_argument('--new', action='store_true', default=False, help='If true, create a new csv file. If false, append to the existing csv file.')
    parser.add_argument('--min_run_time', type=int, default=30, help='Minimum run time for model execution measurement in seconds.')
    args = parser.parse_args()

    root_dir_path = '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/'
    src_file_path = f'/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-{args.dataset_name}-Eva.csv'
    file_df = pd.read_csv(src_file_path)
    file_names = file_df.iloc[:, 0].values.tolist()
    file_names.sort()
    file_paths = [os.path.join(root_dir_path, file_name) for file_name in file_names]
    lower_dataset_name = args.dataset_name.lower()
    if args.measurement_type == 'model_execution':
        save_path = f'/home/hwkang/dev-TSB-AD/TSB-AD/analyses/performance/{lower_dataset_name}_model_execution.csv'
    else:
        save_path = f'/home/hwkang/dev-TSB-AD/TSB-AD/analyses/performance/{lower_dataset_name}_reduction_rate.csv'
    if args.use_kpca:
        save_path = save_path.replace('.csv', '_kpca.csv')
    if args.new:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_path = save_path.replace('.csv', f'_{timestamp}.csv')

    measurements = {}
    for idx, file_path in enumerate(file_paths):
        print(f"Processing file({idx+1}/{len(file_paths)}): {file_path}")

        '''
        파라미터별 채널 축소율 측정용
        '''
        if args.dataset_name == 'MSL':
            # MSL
            params = [
                # Top-3
                (16, 'poly', 8),
                (8, 'poly', 4),
                (16, 'rbf', 4),
                #(4, 'poly', 8),
                #(16, 'poly', 2),

                # Mid-3
                (2, 'rbf', 8),
                (16, 'sigmoid', 16),
                (32, 'sigmoid', 4),
                #(2, 'rbf', 4),
                #(2, 'poly', 16),

                # Bottom-3 except (n_clusters=32)
                (32, 'linear', 2),
                (4, 'linear', 2),
                (16, 'linear', 4),
                #(32, 'poly', 4),
                #(32, 'poly', 8),
            ]
        elif args.dataset_name == 'SMAP':
            # SMAP
            params = [
                # Top-3
                (8, 'sigmoid', 8),
                (8, 'rbf', 2),
                (4, 'sigmoid', 8),
                #(8, 'sigmoid', 4),
                #(8, 'rbf', 4),

                # Mid-3
                #(2, 'sigmoid', 4),
                (2, 'sigmoid', 2),
                (2, 'poly', 2),
                (2, 'sigmoid', 8),
                #(8, 'poly', 4),

                # Bot-3
                #(4, 'linear', 8),
                #(8, 'linear', 2),
                (16, 'linear', 2),
                (16, 'poly', 2),
                (16, 'linear', 4),
            ]

        df = pd.read_csv(file_path)
        data = df.iloc[:, :-1].values
        label = df.iloc[:, -1].values
        original_features = data.shape[1]

        if args.use_kpca:
            if args.measurement_type == 'reduction_rate':
                # 파라미터별 채널 축소율 측정용
                record = {'file_name': os.path.basename(file_path),}
                for idx, (param) in enumerate(params):
                    n_components, kernel, n_clusters = param
                    
                    selected_indices, importances, labels, sorted_indices, important_label = get_selected_indices(data, n_components=n_components, kernel=kernel, n_clusters=n_clusters)
                    reduced_features = len(selected_indices)
                    reduction_rate = (original_features - reduced_features) / original_features * 100
                    
                    if idx >= 0 and idx < 3:
                        col_name = f'top{idx+1}'
                    elif idx >= 3 and idx < 6:
                        col_name = f'mid{idx-3+1}'
                    else:
                        col_name = f'bot{idx-6+1}'

                    print(f'{col_name} | params {param} | selected_indices {selected_indices} | {reduction_rate:.2f}%')
                    record[col_name] = reduction_rate
            else:
                selected_indices, importances, labels, sorted_indices, important_label = get_selected_indices(data, n_components=16, kernel='poly', n_clusters=8)
                data = data[:, selected_indices]

        if args.measurement_type == 'model_execution':
            model = CNN(
                TS_Name=os.path.basename(file_path),
                num_raw_features=data.shape[1],
                local_running_params=running_params
            )

            measurement = model.measure_time(data, min_run_time=args.min_run_time)
            #measurements[file_path] = (data.shape[0], measurement.times)

            record = {
                'file_name': os.path.basename(file_path),
                'use_kpca': 0 if not args.use_kpca else 1,
                'length': data.shape[0],
                'time_distribution': measurement.times
            }

        '''
        레코드 저장
        '''
        # if there is no saved csv file, create a new one
        if not os.path.exists(save_path):
            # 'file_name' is the index column
            save_df = pd.DataFrame(columns=record.keys())
            save_df.set_index('file_name', inplace=True)
        else:
            # append to the existing csv file
            save_df = pd.read_csv(save_path, index_col='file_name')

        row_df = pd.DataFrame([record])
        row_df.set_index('file_name', inplace=True)
        save_df = pd.concat([save_df, row_df], axis=0)
        save_df.to_csv(save_path)