import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse, time

from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans

def get_selected_indices(data, n_components, kernel, n_clusters):
    n_samples, n_features = data.shape

    temp_n_components = max(1, n_features//n_components)
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

python analyses/run_feature_importance.py
'''
if __name__=='__main__':
    # MSL
    '''
    params = [
        # Top-5
        (16, 'poly', 8),
        (8, 'poly', 4),
        (16, 'rbf', 4),
        #(4, 'poly', 8),
        #(16, 'poly', 2),

        # Mid-5
        (2, 'rbf', 8),
        (16, 'sigmoid', 16),
        (32, 'sigmoid', 4),
        #(2, 'rbf', 4),
        #(2, 'poly', 16),

        # Bottom-5 except (n_clusters=32)
        (32, 'linear', 2),
        (4, 'linear', 2),
        (16, 'linear', 4),
        #(32, 'poly', 4),
        #(32, 'poly', 8),
    ]'''

    # SMAP
    params = [
        # Top-5
        (8, 'sigmoid', 8),
        (8, 'rbf', 2),
        (4, 'sigmoid', 8),
        #(8, 'sigmoid', 4),
        #(8, 'rbf', 4),

        # Mid-5
        #(2, 'sigmoid', 4),
        (2, 'sigmoid', 2),
        (2, 'poly', 2),
        (2, 'sigmoid', 8),
        #(8, 'poly', 4),

        # Bot-5
        #(4, 'linear', 8),
        #(8, 'linear', 2),
        (16, 'linear', 2),
        (16, 'poly', 2),
        (16, 'linear', 4),
    ]

    parser = argparse.ArgumentParser(description='Run feature importance analysis')
    parser.add_argument('--dataset_name', type=str, default='MSL', help='Dataset name (e.g., MSL, SMAP)')
    parser.add_argument('--new', action='store_true', default=False, help='Use new dataset structure')
    args = parser.parse_args()

    root_dir_path = '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/'
    src_file_path = f'/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-{args.dataset_name}-Eva.csv'
    file_df = pd.read_csv(src_file_path)
    file_names = file_df.iloc[:, 0].values.tolist()
    file_names.sort()
    file_paths = [os.path.join(root_dir_path, file_name) for file_name in file_names]

    for idx, file_path in enumerate(file_paths):
        print(f"Processing file({idx+1}/{len(file_paths)}): {file_path}")
        fig, axes = plt.subplots(3, 3, figsize=(18, 5), sharex=True, sharey=True)
        axes = axes.reshape(3, 3)
        for row in range(3):
            for col in range(3):
                idx = row * 3 + col
                n_components, kernel, n_clusters = params[idx]

                df = pd.read_csv(file_path)
                data = df.iloc[:, :-1].values
                label = df.iloc[:, -1].values

                selected_indices, importances, labels, sorted_indices, important_label = get_selected_indices(data, n_components, kernel, n_clusters)
                importances = (importances - np.min(importances)) / (np.max(importances) - np.min(importances) + 1e-8)

                ax = axes[row, col]
                #print(important_label)
                #print(labels)
                #print(importances)
                colors = ['red' if labels[i] == important_label else 'blue' for i in range(len(importances))]
                ax.bar(range(len(importances)), importances, color=colors, alpha=0.7)
                
                # 막대 색이 빨간색인 경우 해당 막대그래프에 수직선 추가
                for i in range(len(importances)):
                    if labels[i] == important_label:
                        ax.axvline(x=i, color='red', linestyle='--', linewidth=0.5)

                ax.set_title(f'({n_components}, {kernel}, {n_clusters})', fontsize=10)
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels(range(len(importances)), rotation=45, fontsize=4)

                # ✅ Only 2nd row → show xlabel
                if row == 2:
                    ax.set_xlabel("Feature Index", fontsize=8)

                # ✅ Only 1st column → show ylabel
                if col == 0:
                    ax.set_ylabel("Importance", fontsize=8)

        file_name = os.path.basename(file_path).split('.')[0]
        plt.suptitle(f'{file_name}', fontsize=16)
        plt.tight_layout()
        if args.new:
            # timestamp를 사용해서 새로운 파일을 저장
            # 이전 비교가 필요할 때 사용
            timestamp = time.strftime("%Y%m%d-%H%M%S") 
            plt.savefig(f'/home/hwkang/dev-TSB-AD/TSB-AD/analyses/feature_importance/feature_importance_t-m-b_{file_name}_{timestamp}.png', dpi=600)
        else:
            plt.savefig(f'/home/hwkang/dev-TSB-AD/TSB-AD/analyses/feature_importance/feature_importance_t-m-b_{file_name}.png', dpi=600)
        plt.close()