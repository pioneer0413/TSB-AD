import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse, time

from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans

from TSB_AD.snn.utils import get_selected_indices

'''
Usage:

python analyses/run_feature_importance_single.py
'''
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Run feature importance analysis')
    parser.add_argument('--dataset_name', type=str, default='MSL', help='Dataset name (e.g., MSL, SMAP)')
    parser.add_argument('--n_components', type=int, default=8, help='Number of components for KPCA')
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel type for KPCA (e.g., linear, poly, rbf, sigmoid)')
    parser.add_argument('--n_clusters', type=int, default=2, help='Number of clusters for KMeans')
    parser.add_argument('--new', action='store_true', default=False, help='Use new dataset structure')
    args = parser.parse_args()

    root_dir_path = '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/'
    src_file_path = f'/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-{args.dataset_name}-Eva.csv'
    file_df = pd.read_csv(src_file_path)
    file_names = file_df.iloc[:, 0].values.tolist()
    file_names.sort()
    file_paths = [os.path.join(root_dir_path, file_name) for file_name in file_names]

    for idx, file_path in enumerate(file_paths):
        if idx > 0:
            break
        print(f"Processing file({idx+1}/{len(file_paths)}): {file_path}")
        # only one figure
        plt.figure(figsize=(8, 5))

        df = pd.read_csv(file_path)
        data = df.iloc[:, :-1].values
        label = df.iloc[:, -1].values

        selected_indices, importances, labels, sorted_indices, important_label = get_selected_indices(data, args.n_components, args.kernel, args.n_clusters,
                                                                                                        measure_time=False)
        #importances = (importances - np.min(importances)) / (np.max(importances) - np.min(importances) + 1e-8)
        colors = ['blue' if labels[i] != important_label else 'red' for i in range(len(importances))]
        plt.bar(range(len(importances)), importances, color=[colors[i] for i in range(len(importances))], alpha=0.7)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title(f'Feature Importance for {os.path.basename(file_path)}\nKernel: {args.kernel}, n_components: {args.n_components}, n_clusters: {args.n_clusters}')
        plt.xticks(range(len(importances)), rotation=0 if len(importances) < 30 else 45)
        if important_label is not None:
            for i in range(len(importances)):
                if labels[i] == important_label:
                    plt.axvline(x=i, color='red', linestyle='--', linewidth=1)

        file_name = os.path.basename(file_path).split('.')[0]
        plt.tight_layout()
        if args.new:
            # timestamp를 사용해서 새로운 파일을 저장
            # 이전 비교가 필요할 때 사용
            timestamp = time.strftime("%Y%m%d-%H%M%S") 
            plt.savefig(f'/home/hwkang/dev-TSB-AD/TSB-AD/analyses/feature_importance/feature_importance_single_{file_name}_{timestamp}.png', dpi=600)
        else:
            plt.savefig(f'/home/hwkang/dev-TSB-AD/TSB-AD/analyses/feature_importance/feature_importance_single_{file_name}.png', dpi=600)
        plt.close()