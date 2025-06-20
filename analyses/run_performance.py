import os
import pandas as pd
import numpy as np
import argparse
import time
import psutil

from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans

def measure_block(process, func, *args, block_name=""):
    print(f"\n--- Measuring Block: {block_name} ---")
    start_time = time.time()
    cpu_start = process.cpu_times()
    mem_start = process.memory_info().rss

    result = func(*args)

    end_time = time.time()
    cpu_end = process.cpu_times()
    mem_end = process.memory_info().rss

    print(f"Time elapsed: {end_time - start_time:.4f} sec")
    print(f"CPU time (user + system): {(cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system):.4f} sec")
    print(f"Memory change: {(mem_end - mem_start) / (1024 ** 2):.4f} MB")

    elpased_time = end_time - start_time
    cpu_time = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
    mem_change = (mem_end - mem_start) / (1024 ** 2)  # Convert to MB
    return result, (elpased_time, cpu_time, mem_change)

'''
현재는 미사용
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel PCA performance evaluation")
    parser.add_argument('--dataset_name', type=str, default='MSL', help='Name of the dataset to evaluate')
    args = parser.parse_args()

    root_dir_path = '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/'
    src_file_path = f'/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-{args.dataset_name}-Eva.csv'
    file_df = pd.read_csv(src_file_path)
    file_names = file_df.iloc[:, 0].values.tolist()
    file_paths = [os.path.join(root_dir_path, file_name) for file_name in file_names]

    '''
    save_df = pd.DataFrame(columns=['file_name', 
                                    'kpca_fit_time', 'kpca_transform_time', 'importance_time', 'kmeans_time',
                                    'kpca_fit_cpu_time', 'kpca_transform_cpu_time', 'importance_cpu_time', 'kmeans_cpu_time',
                                    'kpca_fit_mem', 'kpca_transform_mem', 'importance_mem', 'kmeans_mem',
                                    'selected_indices'])
    '''
    
    lower_dataset_name = args.dataset_name.lower()
    save_path = f'/home/hwkang/dev-TSB-AD/TSB-AD/tests/kpca_{lower_dataset_name}_perf.csv'

    for idx, file_path in enumerate(file_paths):
        record = []
        df = pd.read_csv(file_path)
        data = df.iloc[:, :-1].values
        label = df.iloc[:, -1].values

        n_samples, n_features = data.shape
        print(f"Number of samples: {n_samples}, Number of features: {n_features}")

        process = psutil.Process(os.getpid())
        kpca = KernelPCA(n_components=n_features, kernel='rbf', gamma=1.0, fit_inverse_transform=False)

        # --- Block 1: KPCA fit ---
        _, (a, b, c) = measure_block(process, kpca.fit, data, block_name="KPCA.fit")
        tup = (a, b, c)
        record.append(tup)

        # --- Block 2: KPCA transform ---
        base_transform, (a, b, c) = measure_block(process, kpca.transform, data, block_name="KPCA.transform (original)")
        tup = (a, b, c)
        record.append(tup)
        base_var = np.var(base_transform, axis=0).sum()

        # --- Block 3: Importance evaluation ---
        importances = []
        start_time = time.time()
        cpu_start = process.cpu_times()
        mem_start = process.memory_info().rss

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

        end_time = time.time()
        cpu_end = process.cpu_times()
        mem_end = process.memory_info().rss

        print(f"\n--- Measuring Block: Importance computation ---")
        print(f"Time elapsed: {end_time - start_time:.4f} sec")
        print(f"CPU time (user + system): {(cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system):.4f} sec")
        print(f"Memory change: {(mem_end - mem_start) / (1024 ** 2):.4f} MB")

        elapsed_time = end_time - start_time
        cpu_time = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
        mem_change = (mem_end - mem_start) / (1024 ** 2)
        record.append((elapsed_time, cpu_time, mem_change))

        importances = np.array(importances)
        importances = importances / (np.sum(importances) + 1e-8)

        # --- Block 4: KMeans clustering ---
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        labels, (a, b, c) = measure_block(process, kmeans.fit_predict, importances.reshape(-1, 1), block_name="KMeans.fit_predict")
        tup = (a, b, c)
        record.append(tup)

        # --- Post-processing ---
        '''
        sorted_indices = np.argsort(importances)[::-1]
        important_label = labels[sorted_indices[0]]
        selected_indices = np.where(labels == important_label)[0]
        '''
        '''
        for i, tup in enumerate(record):
            print(tup)
        '''

        record_dict = {
            'file_name': file_names[idx],
            'length': n_samples,
            'kpca_fit_time': record[0][0],
            'kpca_transform_time': record[1][0],
            'importance_time': record[2][0],
            'kmeans_time': record[3][0],
            'kpca_fit_cpu_time': record[0][1],
            'kpca_transform_cpu_time': record[1][1],
            'importance_cpu_time': record[2][1],
            'kmeans_cpu_time': record[3][1],
            'kpca_fit_mem': record[0][2],
            'kpca_transform_mem': record[1][2],
            'importance_mem': record[2][2],
            'kmeans_mem': record[3][2]
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