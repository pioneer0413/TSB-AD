import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from fastdtw.fastdtw import fastdtw
from scipy.spatial.distance import euclidean



def compute_dtw_matrix(data):
    """
    data: numpy array of shape (T, N) → time x feature
    Returns: DTW distance matrix of shape (N, N)
    """
    N = data.shape[1]
    distance_matrix = np.zeros((N, N))

    print(N, data.shape)
    for i in range(N):
        for j in range(i + 1, N):
            A = data[:, i]
            B = data[:, j]
            # check if A and B are 1D arrays and numpy arrays
            if A.ndim == 1 and B.ndim == 1:
                A = A.reshape(-1, 1)
                B = B.reshape(-1, 1)
            dist, _ = fastdtw(A, B, dist=euclidean)
            distance_matrix[i, j] = distance_matrix[j, i] = dist
    return distance_matrix

def plot_dtw_heatmap(dtw_matrix, title="DTW Distance Heatmap"):
    """
    dtw_matrix: (N, N) DTW 거리 행렬
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(dtw_matrix, interpolation='nearest', cmap='viridis')
    plt.title(title)
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.colorbar(label="DTW Distance")
    plt.tight_layout()
    plt.savefig(f'/home/hwkang/dev-TSB-AD/TSB-AD/figures/20250615/dtw_{title}.png')
    plt.show()
# pw
src_file_paths = [
    #'/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/137_CreditCard_id_1_Finance_tr_500_1st_541.csv'
    #'/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/042_GHL_id_11_Sensor_tr_50000_1st_150001.csv'
    #'/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/074_SMD_id_18_Facility_tr_7174_1st_21230.csv'
    '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/042_GHL_id_11_Sensor_tr_50000_1st_150001.csv',

    # PW
    '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/065_SMD_id_9_Facility_tr_737_1st_837.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/073_SMD_id_17_Facility_tr_5926_1st_10620.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/078_SMD_id_22_Facility_tr_500_1st_326.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/115_PSM_id_1_Facility_tr_50000_1st_129872.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/130_OPPORTUNITY_id_2_HumanActivity_tr_1045_1st_1145.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/144_SMAP_id_1_Sensor_tr_2052_1st_5300.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/173_GECCO_id_1_Sensor_tr_16165_1st_16265.csv',
]
dfs = [pd.read_csv(src_file_path) for src_file_path in src_file_paths]
datas = [df.iloc[:, :-1].values for df in dfs]
#label = df.iloc[:, -1].values

for i, (data, file_paths) in enumerate(zip(datas, src_file_paths)):
    print(f"Processing dataset {i + 1}/{len(datas)}")
    
    file_name = os.path.basename(file_paths).split('.')[0]

    # Compute DTW distance matrix
    dtw_matrix = compute_dtw_matrix(data)
    
    # Plot DTW distance heatmap
    plot_dtw_heatmap(dtw_matrix, title=file_name)