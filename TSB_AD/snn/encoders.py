import torch
import torch.nn as nn
import torch.nn.init as init
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
import math

from TSB_AD.snn.activations import SpikeActivation, TernarySpikeActivation, DynamicReceptiveSpikeActivation
from TSB_AD.snn.utils import EncodingTracer

class BaseEncoder(nn.Module):
    def __init__(self, local_running_params, num_raw_features):
        super().__init__()

        assert local_running_params is not None, "local_running_params cannot be None."
        self.local_running_params = local_running_params

        self.num_raw_features = num_raw_features
        self.num_enc_features = self.local_running_params['num_enc_features']
        self.window_size = self.local_running_params['window_size']

        if self.local_running_params['activations']['activation'] == 'binary':
            self.act = SpikeActivation(
                local_running_params=self.local_running_params,
                num_features=self.num_raw_features,
                ndim=4,
                layer_type='encoding'
            )
        elif self.local_running_params['activations']['activation'] == 'ternary':
            self.act = TernarySpikeActivation(
                local_running_params=self.local_running_params,
                num_features=self.num_raw_features,
                ndim=4,
                layer_type='encoding'
            )

        if self.local_running_params['analysis']:
            self.tracer = EncodingTracer(local_running_params=self.local_running_params)

    def forward(self):
        raise NotImplementedError
    
class DynamicReceptiveEncoder(BaseEncoder):
    def __init__(self, local_running_params, num_raw_features: int):
        super().__init__(local_running_params=local_running_params, num_raw_features=num_raw_features)

        kernel_size = self.local_running_params['encoders']['conv']['kernel_size']
        stride = self.local_running_params['encoders']['conv']['stride']
        #dilation = self.local_running_params['encoders']['conv']['dilation']

        # Conv1d
        self.transducer = nn.Conv1d(
            in_channels=self.num_raw_features,
            out_channels=self.num_enc_features,
            kernel_size=kernel_size,
            stride=stride,
            padding='same'
        )
        # Weight initialization
        init.kaiming_normal_(self.transducer.weight, mode='fan_in', nonlinearity='relu')
        # LayerNorm
        self.normalizer = nn.LayerNorm(
            normalized_shape=(self.num_enc_features, self.window_size),
        )
        self.act = DynamicReceptiveSpikeActivation(
            local_running_params=self.local_running_params,
            num_features=self.num_raw_features,
            ndim=3,
            layer_type='encoding'
        )
        
    def forward(self, inputs: torch.Tensor): # <- (B, L, C)
        inputs = inputs.permute(0, 2, 1) # (B, C, L)
        
        activations = self.transducer(inputs)  # (B, E, L) <- (B, C, L)
        
        activations = self.normalizer(activations)  # (B, E, L) <- (B, E, L)
        
        spikes = self.act(activations)

        return spikes

class ConvEncoder(BaseEncoder):
    def __init__(self, local_running_params, num_raw_features: int):
        super().__init__(local_running_params=local_running_params, num_raw_features=num_raw_features)

        kernel_size = self.local_running_params['encoders']['conv']['kernel_size']
        stride = self.local_running_params['encoders']['conv']['stride']
        dilation = self.local_running_params['encoders']['conv']['dilation']
        
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.num_enc_features,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, kernel_size // 2),
                dilation=dilation,
            ),
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # batch, 1, C, L

        activations = self.encoder(inputs)  # batch, num_enc_features, C, L

        spikes = self.act(activations)
        
        return spikes # (B, E, C, L)
    
class ReLUEncoder(BaseEncoder):
    def __init__(self, local_running_params, num_raw_features: int):
        super().__init__(local_running_params=local_running_params, num_raw_features=num_raw_features)

        kernel_size = self.local_running_params['encoders']['conv']['kernel_size']
        stride = self.local_running_params['encoders']['conv']['stride']
        dilation = self.local_running_params['encoders']['conv']['dilation']

        self.conv = nn.Conv1d(
            in_channels=self.num_raw_features,
            out_channels=self.num_enc_features,
            kernel_size=kernel_size,
            stride=stride,
            padding='same'
        )
        # Weight initialization
        init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        # LayerNorm
        self.normalizer = nn.LayerNorm(
            normalized_shape=(self.num_enc_features, self.window_size),
        )
        self.relu = nn.ReLU()
    def forward(self, inputs: torch.Tensor):
        #inputs = inputs.permute(0, 2, 1) # (B, C, L)
        activations = self.conv(inputs)
        activations = self.normalizer(activations)  # (B, E, L) <- (B, E, L)
        activations = self.relu(activations)  # (B, E, L) <- (B, E, L)
        return activations # (B, E, L) <- (B, E, L)

class PoissonEncoder(nn.Module):
    def __init__(self, num_enc_features: int=10):
        super().__init__()
        self.num_enc_features= num_enc_features

    def forward(self, inputs: torch.Tensor):
        # inputs: (B, L, C)
        spk_rec = []
        for i in range(self.num_enc_features): # num_enc_features(=num_steps)만큼 반복
            # Poisson spike generation
            spk = self.PoissonGen(inputs)
            spk_rec.append(spk) # (B, L, C) * T <-
        spikes = torch.stack(spk_rec, dim=1).permute(0, 1, 3, 2) # (B, T, C, L) <- (B, L, C) * T
        return spikes # (B, T, C, L) <-

    def PoissonGen(self, inputs, rescale_fac=1.0):
        random_input = torch.rand_like(inputs)
        # 입력의 부호까지 고려, 입력값이 크고 음수인 경우 음의 스파이크 발생
        return torch.mul(torch.le(random_input*rescale_fac, torch.abs(inputs)).float(), torch.under_sign(inputs))

class RepeatEncoder(BaseEncoder):
    def __init__(self, local_running_params, num_raw_features: int):
        super().__init__(local_running_params=local_running_params, num_raw_features=num_raw_features)

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.repeat(
            tuple([self.num_enc_features] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )  # num_enc_features batch L C
        inputs = inputs.permute(1, 0, 3, 2)  # (B, E, C, L)

        spikes = self.act(inputs)

        return spikes # (B, E, C, L) <-

class DeltaEncoder(BaseEncoder):
    def __init__(self, local_running_params, num_raw_features: int):
        super().__init__(local_running_params=local_running_params, num_raw_features=num_raw_features)
        
        self.encoder = nn.Linear(1, self.num_enc_features)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, inputs: torch.Tensor, outputs=False):
        # inputs: batch, L, C
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)  # batch, 1, C, L
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1)  # batch, C, L, 1
        activations = self.encoder(delta)  # batch, C, L, num_enc_features
        activations = activations.permute(0, 3, 1, 2)  # batch, num_enc_features, C, L

        spikes = self.act(activations)

        if outputs:
            return spikes, activations  # (B, E, C, L), (B, E, C, L)
        else:
            return spikes # (B, E, C, L)

def main():
    '''
    시각화용 main 함수
    '''
    poisson_encoder = PoissonEncoder(num_enc_features=10)
    repeat_encoder = RepeatEncoder(num_enc_features=10)
    delta_encoder = DeltaEncoder(num_enc_features=10)
    conv_encoder = ConvEncoder(num_enc_features=10)
    
    # Dummy input
    inputs = torch.randn(1, 10, 3)  # (batch, length, num_features)

    out_poisson = poisson_encoder(inputs)
    out_repeat = repeat_encoder(inputs)
    out_delta = delta_encoder(inputs)
    out_conv = conv_encoder(inputs)
    
    print("Poisson Encoder Output Shape:", out_poisson.shape) 
    print("Repeat Encoder Output Shape:", out_repeat.shape) 
    print("Delta Encoder Output Shape:", out_delta.shape)
    print("Conv Encoder Output Shape:", out_conv.shape)
    
    # (B*E*C, L)로 변환
    enc_shape = out_conv.shape
    out_conv = out_conv.reshape(-1, enc_shape[3]) # (B*E*C, L)
    out_delta = out_delta.reshape(-1, enc_shape[3]) # (B*E*C, L)
    out_repeat = out_repeat.reshape(-1, enc_shape[3]) 
    out_poisson = out_poisson.reshape(-1, enc_shape[3])

    viz = False
    if viz:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].imshow(out_poisson.detach().numpy(), aspect='auto', cmap='gray')
        axs[0, 0].set_title('Poisson Encoder')
        axs[0, 1].imshow(out_repeat.detach().numpy(), aspect='auto', cmap='gray')
        axs[0, 1].set_title('Repeat Encoder')
        axs[0, 2].imshow(out_delta.detach().numpy(), aspect='auto', cmap='gray')
        axs[0, 2].set_title('Delta Encoder')
        axs[1, 0].imshow(out_conv.detach().numpy(), aspect='auto', cmap='gray')
        axs[1, 0].set_title('Conv Encoder')
        
        # inputs은 히트맵 형식으로 출력
        axs[1, 2].imshow(inputs.detach().numpy(), aspect='auto', cmap='hot')
        axs[1, 2].set_title('Input')
        plt.tight_layout()
        plt.savefig('encoder_outputs.png')
        plt.close()

if __name__ == "__main__":
    main()