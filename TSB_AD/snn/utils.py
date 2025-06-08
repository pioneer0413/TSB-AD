from torch import nn
import torch
import torch.nn.functional as F
import os
import numpy as np

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

        


if __name__ == "__main__":
    input_size = 50
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    chomp_size = 0
    output_size = calculate_output_size(input_size, kernel_size, stride, padding, dilation) - chomp_size
    print(f"Output size: {output_size}")