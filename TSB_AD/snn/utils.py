from torch import nn
import torch
import torch.nn.functional as F
import os
import numpy as np
import time
import psutil
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()
    
class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, : -self.chomp_size].contiguous()
    
def get_last_number(log_files):
    max_num = 0
    if len(log_files) > 0:
        for file in log_files:
            segments = file.split('_')
            file_number = int(segments[0])
            if file_number > max_num:
                max_num = file_number
        file_number = max_num + 1
    else:
        file_number = 0
    return file_number

def loss_visualization(train_loss, valid_loss, save_dir_path='/home/hwkang/dev-TSB-AD/TSB-AD/figures/loss_evolution', TS_Name=None, AD_Name=None, Encoder_Name=None, hyperparameters: list=[]):
    """
    Visualize training and validation loss.
    
    Args:
        train_loss: Training loss values
        valid_loss: Validation loss values
    """
    import matplotlib.pyplot as plt
    
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{TS_Name}')

    # hyperparameters의 모든 값을 _을 구분자로 붙여서 변수명 hps로 문자열로 저장
    hps = '_'.join([str(hp) for hp in hyperparameters])

    if Encoder_Name is not None:
        plt.savefig(f'{save_dir_path}/{AD_Name}_{Encoder_Name}_{TS_Name}_loss_{hps}.png')
    else:
        plt.savefig(f'{save_dir_path}/{AD_Name}_{TS_Name}_loss_{hps}.png')
    plt.close()

def measure_energy_and_time(function):
    import time
    import pyRAPL
    pyRAPL.setup()
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        m = pyRAPL.Measurement('energy')
        m.start()
        result = function(*args, **kwargs)
        m.stop()
        end = time.perf_counter()
        elapsed = end - start
        energy = m.result.pkg  # energy consumption in Joules
        print(f"Time consumed: {elapsed:.6f} seconds, Energy consumed: {energy} Joules")
        return result, elapsed, energy
    return wrapper

def calculate_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Calculate the output size of a convolutional layer.
    
    Args:
        input_size: Size of the input
        kernel_size: Size of the kernel
        stride: Stride of the convolution
        padding: Padding added to both sides
        dilation: Dilation rate
        
    Returns:
        Output size after convolution
    """
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

class BaseTracer():
    def __init__(self, local_running_params):
        self.local_running_params = local_running_params
        self.dst_dir_path = f'/home/hwkang/dev-TSB-AD/TSB-AD/analyses/exports'
        self.file_name = f'{self.id_code}_{self.AD_Name}_{self.Encoder_Name}_{self.postfix}'

class EncodingTracer(BaseTracer):
    def __init__(self, local_running_params):
        super().__init__(local_running_params=local_running_params)
        self.spike_rate_trend = []
        self.threshold_trend = []
        self.normalization_gamma_trend = []
        self.normalization_beta_trend = []
    
    def export_spike_rate_trend(self):
        pass
    
    def export_threshold_trend(self):
        pass
    
    def export_normalization_trend(self):
        pass

class LearningTracer(BaseTracer):
    def __init__(self, local_running_params):
        super().__init__(local_running_params=local_running_params)
        self.train_loss_tracer = []
        self.valid_loss_tracer = []
        #self.gradient_trend = []
        #self.gradient_distribution = []
    
    def export_loss(self):
        target = 'loss.npz'
        file_name = f'{self.file_name}_{target}'
        save_file_path = os.path.join(self.dst_dir_path, file_name)
        np.savez(save_file_path, train_loss_trend=self.train_loss_tracer, valid_loss_trend=self.valid_loss_tracer)
    
    def export_gradient(self):
        pass

def measure_block(process, func, *args, block_name="", verbose=False):
    if verbose:
        print(f"\n--- Measuring Block: {block_name} ---")
    start_time = time.time()
    cpu_start = process.cpu_times()
    mem_start = process.memory_info().rss

    result = func(*args)

    end_time = time.time()
    cpu_end = process.cpu_times()
    mem_end = process.memory_info().rss

    if verbose:
        print(f"Time elapsed: {end_time - start_time:.4f} sec")
        print(f"CPU time (user + system): {(cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system):.4f} sec")
        print(f"Memory change: {(mem_end - mem_start) / (1024 ** 2):.4f} MB")

    elpased_time = end_time - start_time
    cpu_time = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
    mem_change = (mem_end - mem_start) / (1024 ** 2)  # Convert to MB
    return result, (elpased_time, cpu_time, mem_change)

def get_selected_indices(data, n_components, kernel, n_clusters, measure_time=False, file_name=''):

    n_samples, n_features = data.shape

    arg_n_components = min(n_features-1, n_components)
    kpca = KernelPCA(n_components=arg_n_components, kernel=kernel, gamma=1.0, fit_inverse_transform=False)
    
    # Stage 1
    if measure_time:
        process = psutil.Process(os.getpid())
        _, (fit_time, fit_cpu_time, fit_mem_change) = measure_block(process, kpca.fit, data, block_name="KPCA Fit")
    else:
        kpca.fit(data)

    # Stage 2: get base transformation and variance
    if measure_time:
        base_transform, (transform_time, transform_cpu_time, transform_mem_change) = measure_block(process, kpca.transform, data, block_name="KPCA Transform (original)")
    else:
        base_transform = kpca.transform(data)
    base_var = np.var(base_transform, axis=0).sum()

    # Stage 3: evaluate importance of each feature
    importances = []

    # Measure time manually if required
    if measure_time:
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
    if measure_time:
        end_time = time.time()
        cpu_end = process.cpu_times()
        mem_end = process.memory_info().rss

        elapsed_time = end_time - start_time
        cpu_time = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
        mem_change = (mem_end - mem_start) / (1024 ** 2)

    importances = np.array(importances)
    importances = importances / (np.sum(importances) + 1e-8)

    # Stage 4: KMeans clustering
    arg_n_clusters = min(n_features-1, n_clusters)
    kmeans = KMeans(n_clusters=arg_n_clusters, random_state=0, n_init=10)
    if measure_time:
        labels, (kmeans_time, kmeans_cpu_time, kmeans_mem_change) = measure_block(process, kmeans.fit_predict, importances.reshape(-1, 1), block_name="KMeans Fit Predict")
    else:
        labels = kmeans.fit_predict(importances.reshape(-1, 1))

    # --- Post-processing ---
    sorted_indices = np.argsort(importances)[::-1]
    important_label = labels[sorted_indices[0]]
    selected_indices = np.where(labels == important_label)[0]

    if measure_time:
        assert file_name != '', "File name must be provided when measure_time is True"
        record_dict = {
            'file_name': file_name,
            'length': data.shape[0],

            'kpca_fit_time': fit_time,
            'kpca_transform_time': transform_time,
            'importance_time': elapsed_time,
            'kmeans_time': kmeans_time,

            'kpca_fit_cpu_time': fit_cpu_time,
            'kpca_transform_cpu_time': transform_cpu_time,
            'importance_cpu_time': cpu_time,
            'kmeans_cpu_time': kmeans_cpu_time,

            'kpca_fit_mem': fit_mem_change,
            'kpca_transform_mem': transform_mem_change,
            'importance_mem': mem_change,
            'kmeans_mem': kmeans_mem_change,
        }

    if measure_time:
        return selected_indices, importances, labels, sorted_indices, important_label, record_dict
    return selected_indices, importances, labels, sorted_indices, important_label

if __name__ == "__main__":
    input_size = 50
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    chomp_size = 0
    output_size = calculate_output_size(input_size, kernel_size, stride, padding, dilation) - chomp_size
    print(f"Output size: {output_size}")