import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_paths = [
    '/home/hwkang/TSB-AD/Datasets/TSB-AD-M/025_MITDB_id_7_Medical_tr_37500_1st_88864.csv',
    '/home/hwkang/TSB-AD/Datasets/TSB-AD-M/058_SMD_id_2_Facility_tr_1087_1st_1187.csv',
    '/home/hwkang/TSB-AD/Datasets/TSB-AD-M/074_SMD_id_18_Facility_tr_7174_1st_21230.csv',
    '/home/hwkang/TSB-AD/Datasets/TSB-AD-M/092_SVDB_id_9_Medical_tr_2674_1st_2774.csv',
    '/home/hwkang/TSB-AD/Datasets/TSB-AD-M/097_SVDB_id_14_Medical_tr_1031_1st_1131.csv',
    '/home/hwkang/TSB-AD/Datasets/TSB-AD-M/115_PSM_id_1_Facility_tr_50000_1st_129872.csv',
    '/home/hwkang/TSB-AD/Datasets/TSB-AD-M/141_CATSv2_id_4_Sensor_tr_41727_1st_41827.csv',
    '/home/hwkang/TSB-AD/Datasets/TSB-AD-M/162_SMAP_id_19_Sensor_tr_1908_1st_4690.csv',
    '/home/hwkang/TSB-AD/Datasets/TSB-AD-M/172_SWaT_id_2_Sensor_tr_23700_1st_23800.csv',
]
save_dir_path = '/home/hwkang/TSB-AD/figures/odd_data'

for file_path in file_paths:
    df = pd.read_csv(file_path)
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    num_attrs = data.shape[1]

    # subplot 사용을 위한 행과 열의 수를 num_attrs에 맞게, 정사각형에 가깝게 설정
    num_rows = int(np.ceil(np.sqrt(num_attrs)))
    num_cols = int(np.ceil(num_attrs / num_rows))

    print(num_rows, num_cols)

    parts_of_file_name = file_path.split('/')[-1].split('_')

    # 각 subplot의 크기 설정
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10), squeeze=False)
    fig.suptitle(f'{parts_of_file_name[0]}_{parts_of_file_name[1]}', fontsize=16)
    for i in range(num_attrs):
        ax = axes[i // num_cols, i % num_cols]
        ax.plot(data[:, i], label=f'Attribute {i+1}')
        ax.set_title(f'Attribute {i+1}')
        #ax.legend()
        #ax.grid()
        # label의 경우, 0은 정상, 1은 이상치로 가정
        # 1인 경우 plot에 빨간 점으로 표시 그 외는 푸른 색
        anomaly_indices = np.where(label == 1)[0]
        ax.scatter(anomaly_indices, data[anomaly_indices, i], color='red', label='Anomaly', s=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 제목과 서브플롯 간격 조정
    plt.savefig(os.path.join(save_dir_path, f'{parts_of_file_name[0]}_{parts_of_file_name[1]}.png') , dpi=300)