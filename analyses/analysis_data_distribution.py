import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns

# 출력 디렉터리 생성
output_dir = '/home/hwkang/TSB-AD/snn/data_distributions/'
os.makedirs(output_dir, exist_ok=True)

def visualize_windows(df, file_name, overlap=False):
    """
    DataFrame의 특성들에 대해 윈도우 기반 시각화를 수행
    
    Args:
        df: 입력 DataFrame
        file_name: 파일 이름
        overlap: 윈도우 중첩 여부
    """
    n_features = df.shape[1] - 1  # 라벨 열 제외
    
    if overlap:
        # 처음 20,000개 행만 선택 (또는 전체 데이터의 20%를 사용)
        n_samples = min(20000, len(df))
        df_subset = df.iloc[:n_samples]
        window_size = n_samples // 6
        stride = (n_samples - window_size) // 9  # 10개의 중첩 윈도우 생성
    else:
        window_size = len(df) // 10  # 중첩 없는 10개 윈도우
        stride = window_size
        df_subset = df
    
    # subplot 생성
    fig, axes = plt.subplots(n_features * 2, 10, figsize=(25, 4 * n_features))
    if n_features == 1:
        axes = axes.reshape(2, 10)
    
    # 각 feature별 색상 지정
    colors = sns.color_palette("husl", n_features)  # feature 수만큼 다른 색상 생성
    
    # 각 window별 시각화
    for idx in range(10):
        start_idx = idx * stride
        end_idx = start_idx + window_size
        window = df_subset.iloc[start_idx:end_idx]
        
        for feature_idx in range(n_features):
            feature_data = window.iloc[:, feature_idx]
            
            # 시계열 플롯
            ax_idx = feature_idx * 2
            axes[ax_idx, idx].plot(feature_data, 
                                 color=colors[feature_idx])
            axes[ax_idx, idx].set_title(f'Window {idx+1}\nFeature {feature_idx+1}')
            axes[ax_idx, idx].set_xlabel('Time')
            axes[ax_idx, idx].set_ylabel('Value')
            
            # 히스토그램
            axes[ax_idx + 1, idx].hist(feature_data, bins=50, 
                                     color=colors[feature_idx], 
                                     alpha=0.7)
            axes[ax_idx + 1, idx].set_title(f'Histogram {idx+1}\nFeature {feature_idx+1}')
            axes[ax_idx + 1, idx].set_xlabel('Value')
            axes[ax_idx + 1, idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # 파일 저장
    suffix = '_overlapped' if overlap else '_non_overlapped'
    output_path = os.path.join(output_dir, f'{file_name}{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_csv_files():
    """모든 CSV 파일에 대해 시각화 수행 (파일 크기 순으로)"""
    csv_path = '/home/hwkang/TSB-AD/Datasets/TSB-AD-M/*.csv'
    
    # 파일 경로와 크기를 튜플로 저장
    file_sizes = []
    for file_path in glob.glob(csv_path):
        file_size = os.path.getsize(file_path)
        file_sizes.append((file_path, file_size))
    
    # 파일 크기 순으로 정렬
    file_sizes.sort(key=lambda x: x[1])  # 두 번째 요소(파일 크기)로 정렬
    
    # 정렬된 순서대로 처리
    for file_path, size in file_sizes:
        # 파일 이름 추출
        file_name = os.path.basename(file_path).replace('.csv', '')
        size_mb = size / (1024 * 1024)  # 크기를 MB 단위로 변환
        print(f"Processing {file_name}... (Size: {size_mb:.2f} MB)")
        
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        
        # 중첩 없는 윈도우 시각화
        visualize_windows(df, file_name, overlap=False)
        
        # 중첩 있는 윈도우 시각화
        visualize_windows(df, file_name, overlap=True)
        
        print(f"Completed processing {file_name}")

if __name__ == "__main__":
    process_csv_files()