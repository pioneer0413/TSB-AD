import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
import snntorch as snn
from snntorch import surrogate as snn_surrogate
import torch.nn.functional as F
from sklearn.decomposition import KernelPCA, PCA
from sklearn.cluster import KMeans
import numpy as np

tau = 2.0
backend = "torch"
detach_reset = True

class BaseEncoder(nn.Module):
    def __init__(self, local_running_params):
        super().__init__()
        #assert local_running_params is not None, "local_running_params must be provided"
        self.local_running_params = local_running_params
    def forward(self):
        raise NotImplementedError("Subclasses should implement this method.")

class RepeatEncoder(BaseEncoder):
    def __init__(self, local_running_params, output_size: int, neuron_type: str = 'spikingjelly'):
        super().__init__(local_running_params)
        self.out_size = output_size
        self.neuron_type = neuron_type
        if neuron_type == 'spikingjelly':
            self.lif = neuron.LIFNode(
                tau=tau,
                step_mode="m",
                detach_reset=detach_reset,
                surrogate_function=surrogate.ATan(),
            )
        else:
            self.lif = snn.Leaky(beta=0.99, spike_grad=snn_surrogate.atan(), init_hidden=True, output=False)

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        inputs = inputs.repeat(
            tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )  # T B L C
        inputs = inputs.permute(0, 1, 3, 2)  # T B C L
        if self.neuron_type == 'spikingjelly':
            spks = self.lif(inputs)  # T B C L
            return spks.permute(1, 0, 2, 3)  # B T C L
        else:
            inputs = inputs.permute(1, 0, 2, 3)  # B T C L
            spks = self.lif(inputs)  # B T C L
            return spks
        
class DeltaEncoder(BaseEncoder):
    def __init__(self, local_running_params, num_raw_features, output_size: int, neuron_type: str = 'spikingjelly'):
        super().__init__(local_running_params)
        self.norm = nn.BatchNorm2d(1) if local_running_params['ParallelSNNModel']['norm_type'] == 'bn' else nn.LayerNorm((1, num_raw_features, local_running_params['model']['window_size']))
        self.enc = nn.Linear(1, output_size)
        self.neuron_type = neuron_type
        if neuron_type == 'spikingjelly':
            self.lif = neuron.LIFNode(
                tau=tau,
                step_mode="m",
                detach_reset=detach_reset,
                surrogate_function=surrogate.ATan(),
            )
        else:
            self.lif = snn.Leaky(beta=0.99, spike_grad=snn_surrogate.atan(), init_hidden=True, output=False)

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)  # B, 1, C, L
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1)  # B, C, L, 1
        enc = self.enc(delta)  # B, C, L, T
        enc = enc.permute(3, 0, 1, 2)  # T, B, C, L
        if self.neuron_type == 'spikingjelly':
            spks = self.lif(enc)
            return spks.permute(1, 0, 2, 3)  # B, T, C, L
        else:
            enc = enc.permute(1, 0, 2, 3)
            spks = self.lif(enc)
            return spks  # B, T, C, L
        
class ConvEncoder(BaseEncoder):
    def __init__(self, local_running_params, output_size: int, kernel_size: int = 3, neuron_type: str = 'spikingjelly'):
        super().__init__(local_running_params)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.neuron_type = neuron_type
        if neuron_type == 'spikingjelly':
            self.lif = neuron.LIFNode(
                tau=tau,
                step_mode="m",
                detach_reset=detach_reset,
                surrogate_function=surrogate.ATan(),
            )
        else:
            self.lif = snn.Leaky(beta=0.99, spike_grad=snn_surrogate.atan(), init_hidden=True, output=False)

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # B, 1, C, L
        enc = self.encoder(inputs)  # B, T, C, L
        enc = enc.permute(1, 0, 2, 3)  # T, B, C, L
        if self.neuron_type == 'spikingjelly':
            spks = self.lif(enc)
            return spks.permute(1, 0, 2, 3)
        else:
            enc = enc.permute(1, 0, 2, 3) # B, T, C, L <
            spks = self.lif(enc)
            return spks

class ReceptiveEncoder(BaseEncoder):
    def __init__(self, local_running_params, num_raw_features, num_enc_features):
        super().__init__(local_running_params)

        self.kernel_size = 3

        self.transducer = nn.Conv1d(
            in_channels=num_raw_features,
            out_channels=num_enc_features,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        if self.local_running_params['ParallelSNNModel']['norm_type'] == 'bn':
            self.norm = nn.BatchNorm1d(num_enc_features)
        else:
            self.norm = nn.LayerNorm((num_enc_features, local_running_params['model']['window_size']))

        self.dropout = nn.Dropout1d(p=0.5)

        # S-LIF
        self.s_lif = neuron.SlidingPSN(
            k=self.local_running_params['ParallelSNNModel']['encoding_kernel'][0],
            step_mode='m' if self.local_running_params['ParallelSNNModel']['step_mode'] == 'm' else 's',
            backend='conv',
        )
        
        # F-LIF
        self.f_lif = neuron.SlidingPSN(
            k=self.local_running_params['ParallelSNNModel']['encoding_kernel'][1],
            step_mode='m' if self.local_running_params['ParallelSNNModel']['step_mode'] == 'm' else 's',
            backend='conv',
        )
        # N-LIF
        self.n_lif = neuron.SlidingPSN(
            k=self.local_running_params['ParallelSNNModel']['encoding_kernel'][2],
            step_mode='m' if self.local_running_params['ParallelSNNModel']['step_mode'] == 'm' else 's',
            backend='conv',
        )

        if self.local_running_params['ParallelSNNModel']['tt']:
            self.s_lif.bias = nn.Parameter(-0.5*torch.ones(self.local_running_params['model']['window_size']).view(self.local_running_params['model']['window_size'], 1, 1))
            self.f_lif.bias = nn.Parameter(-1.0*torch.ones(self.local_running_params['model']['window_size']).view(self.local_running_params['model']['window_size'], 1, 1))
            self.n_lif.bias = nn.Parameter(-2.0*torch.ones(self.local_running_params['model']['window_size']).view(self.local_running_params['model']['window_size'], 1, 1))
        else:
            self.s_lif.bias = nn.Parameter(torch.as_tensor(-0.5))
            self.f_lif.bias = nn.Parameter(torch.as_tensor(-1.0))
            self.n_lif.bias = nn.Parameter(torch.as_tensor(-2.0))

        # Conv2d for graded spike
        self.conv2d = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=(1, 1)
        )

    def forward(self, inputs: torch.Tensor, vizualize: bool = False):
        # IN: B, L, C
        inputs = inputs.permute(0, 2, 1) # B, C, L <
        activations = self.transducer(inputs) # B, E, L <
        activations = self.norm(activations)

        if self.local_running_params['ParallelSNNModel']['dropout']:
            activations = self.dropout(activations)  # B, E, L <

        delta_activations = torch.zeros_like(activations) # B, E, L <
        delta_activations[:, :, 1:] = activations[:, :, 1:] - activations[:, :, :-1]
        delta_activations[:, :, 0] = 0.0

        if self.local_running_params['ParallelSNNModel']['delta_abs']:
            delta_activations = torch.abs(delta_activations)

        # F-LIF
        f_lif_out = self.f_lif(delta_activations.permute(2, 0, 1)).permute(1, 2, 0)
        # S-LIF
        s_lif_out = self.s_lif(activations.permute(2, 0, 1)).permute(1, 2, 0)  # B, E, L < L, B, E
        # N-LIF
        n_lif_out = self.n_lif(activations.permute(2, 0, 1)).permute(1, 2, 0) # B, E, L < L, B, E
                
        lif_out = torch.stack([s_lif_out, f_lif_out, n_lif_out], dim=1)  # B, 3, E, L

        if self.local_running_params['ParallelSNNModel']['grad_spike']:
            lif_out = self.conv2d(lif_out)  # B, 1, E, L
        else:
            lif_out = lif_out.sum(dim=1).unsqueeze(1)  # B, 1, E, L

        if vizualize:
            return lif_out, activations, delta_activations # B, 1, E, L < B, E, L < B, E, L
        return lif_out # B, 1, E, L <
    
class DynamicReceptiveEncoder(BaseEncoder):
    def __init__(self, local_running_params, num_raw_features, num_enc_features):
        super().__init__(local_running_params)

        self.kernel_size = 3
        '''self.transducer = nn.Conv2d(
            in_channels=1,
            out_channels=num_enc_features,
            kernel_size=(1, self.kernel_size),
            stride=1,
            padding=(0, self.kernel_size // 2),
        )'''
        self.transducer = nn.Conv1d(
            in_channels=num_raw_features,
            out_channels=num_enc_features,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        self.norm = nn.BatchNorm1d(num_enc_features)

        # SA-I
        #self.sa_i = neuron.SimpleLIFNode(tau=2., decay_input=True)
        self.sa_i = neuron.SlidingPSN(k=3, step_mode='m', backend='conv')
        self.sa_i.bias = nn.Parameter(torch.as_tensor(-0.5))  # SlidingPSN bias
        self.adapt_gain = 1e-5
        # FA-I
        #self.fa_i = neuron.SimpleLIFNode(tau=2., decay_input=False)
        self.fa_i = neuron.SlidingPSN(k=3, step_mode='m', backend='conv')
        self.fa_i.bias = nn.Parameter(torch.as_tensor(-1.0))  # SlidingPSN bias

    def forward(self, inputs: torch.Tensor):
        # IN: B, L, C
        #inputs = inputs.permute(0, 2, 1).unsqueeze(1) # B, 1, C, L < B, L, C
        inputs = inputs.permute(0, 2, 1)  # B, C, L < B, L, C

        activations = self.transducer(inputs)  # B, E, L < B, C, L
        activations = self.norm(activations)
        delta_activations = torch.zeros_like(activations)
        delta_activations[:, :, 1:] = activations[:, :, 1:] - activations[:, :, :-1]  # B, E, L < B, C, L
        delta_activations[:, :, 0] = 0.0
        delta_activations = torch.abs(delta_activations)

        sa_i_out = self.sa_i(activations.permute(2, 0, 1))  # L, B, E < B, E, L
        fa_i_out = self.fa_i(delta_activations.permute(2, 0, 1))  # L, B, E < B, E, L

        outputs = sa_i_out + fa_i_out
        outputs = outputs.permute(1, 2, 0).unsqueeze(1)
        '''
        cumsum = torch.cumsum(sa_i_out.permute(1,2,3,0), dim=-1)  # B, E, C, L < L, B, E, C
        adapt = self.sa_i_threshold + self.adapt_gain * cumsum  # B, E, C, L
        activations = activations - adapt
        sa_i_out = self.sa_i(activations.permute(3, 0,1,2))  # L, B, E, C < B, E, C, L
        outputs = sa_i_out + fa_i_out
        outputs = outputs.permute(1, 2, 3, 0)
        '''
        return outputs, activations, delta_activations # B, 1, E, L <
    
    def reset_adapt(self):
        #print(self.adapt_strength)
        self.adapt_strength = 0.05
        self.recovery_strength = 0.02

    def split_low_high(self, x, cutoff_ratio=0.25):
        """
        x : (B, C, L) real tensor
        cutoff_ratio : 0~1, 비율로 구분 (예: 0.25 → Nyquist의 25% 이하를 저주파로)
        반환값       : low, high  (둘 다 (B, C, L))
        """
        B, C, L = x.shape
        N_fft   = L // 2 + 1                     # rfft length
        k_c     = int(cutoff_ratio * (N_fft - 1))

        # 1) FFT
        Xf = torch.fft.rfft(x, dim=-1)           # (B, C, N_fft), complex64/128

        # 2) 주파수 마스크
        mask = torch.zeros(N_fft, device=x.device, dtype=torch.bool)
        mask[:k_c + 1] = True                   # 0~k_c 포함 → low
        mask_low  = mask
        mask_high = ~mask

        # broadcast masks to (B,C,N_fft)
        mask_low  = mask_low.view(1, 1, -1)
        mask_high = mask_high.view(1, 1, -1)

        # 3) 필터링
        X_low  = Xf * mask_low
        X_high = Xf * mask_high

        # 4) IFFT
        x_low  = torch.fft.irfft(X_low,  n=L, dim=-1)   # (B,C,L)
        x_high = torch.fft.irfft(X_high, n=L, dim=-1)

        return x_low, x_high
    
    def temporal_attention(self, x):
        # x: (B, L, C)
        mp = F.adaptive_max_pool1d(x, 1) # (B, L, 1)
        mp = self.fc1(mp.squeeze(-1))
        mp = F.relu(mp)
        mp = self.fc2(mp).unsqueeze(-1)
        
        ap = F.adaptive_avg_pool1d(x, 1)
        ap = self.fc1(ap.squeeze(-1))
        ap = F.relu(ap)
        ap = self.fc2(ap).unsqueeze(-1)

        normalized = F.normalize(mp + ap, dim=1)
        attention = F.softmax(normalized, dim=-1)
        return attention