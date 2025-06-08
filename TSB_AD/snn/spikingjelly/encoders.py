import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
import snntorch as snn
from snntorch import surrogate as snn_surrogate
import torch.nn.functional as F

tau = 2.0
backend = "torch"
detach_reset = True

class BaseEncoder(nn.Module):
    def __init__(self, local_running_params):
        super().__init__()
        #assert local_running_params is not None, "local_running_params must be provided"
        self.local_running_params = local_running_params

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
    def __init__(self, local_running_params, output_size: int, neuron_type: str = 'spikingjelly'):
        super().__init__(local_running_params)
        self.norm = nn.BatchNorm2d(1)
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

class DynamicReceptiveEncoder(BaseEncoder):
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

        # S-LIF
        self.s_lif = neuron.SlidingPSN(
            k=self.local_running_params['ParallelSNNModel']['encoding_kernel'][0],
            step_mode='m' if self.local_running_params['ParallelSNNModel']['step_mode'] == 'm' else 's',
            backend='conv',
        )
        self.s_lif.bias = nn.Parameter(
            torch.as_tensor(-0.5)
        )
        self.adapt_rate = 0.5

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
        self.n_lif.bias = nn.Parameter(
            torch.as_tensor(-2.0)
        )
        self.sensitize_rate = 0.5

        # Conv2d for graded spike
        self.conv2d = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=(1, 1)
        )

    def forward(self, inputs: torch.Tensor):
        # IN: B, L, C
        inputs = inputs.permute(0, 2, 1) # B, C, L <
        activations = self.transducer(inputs) # B, E, L <
        x = activations = self.norm(activations)

        delta_activations = torch.zeros_like(activations)
        delta_activations[:, :, 1:] = activations[:, :, 1:] - activations[:, :, :-1] # B, E, L-1 <
        delta_activations[:, :, 0] = 0.0
        
        if self.local_running_params['ParallelSNNModel']['step_mode'] == 'm':
            # S-LIF
            s_lif_out = self.s_lif(activations.permute(2, 0, 1)).permute(1, 2, 0)  # L, B, E < L, B, E
            # F-LIF
            f_lif_out = self.f_lif(delta_activations.permute(2, 0, 1)).permute(1, 2, 0)
            # N-LIF
            n_lif_out = self.n_lif(activations.permute(2, 0, 1)).permute(1, 2, 0) # L, B, E < L, B, E
        else:
            num_steps = activations.size(2)
            shape = activations.shape
            activations = activations.reshape(shape[2], shape[0], shape[1])  # L, B, E <
            delta_activations = delta_activations.reshape(shape[2], shape[0], shape[1])  # L, B, E <
            temp_s_lif_out = []
            temp_f_lif_out = []
            temp_n_lif_out = []
            prev_s_lif_out = torch.zeros_like(activations[0])  # B, E
            #prev_f_lif_out = torch.zeros_like(delta_activations[0])
            prev_n_lif_out = torch.zeros_like(activations[0])
            for step in range(num_steps):
                # S-LIF
                s_lif_out = self.s_lif(activations[step] - prev_s_lif_out * self.adapt_rate)
                temp_s_lif_out.append(s_lif_out)
                prev_s_lif_out = s_lif_out
                
                # F-LIF
                f_lif_out = self.f_lif(delta_activations[step])
                temp_f_lif_out.append(f_lif_out)
                #prev_f_lif_out = f_lif_out
                
                # N-LIF
                n_lif_out = self.n_lif(activations[step] + prev_n_lif_out * self.sensitize_rate)
                temp_n_lif_out.append(n_lif_out)
                prev_n_lif_out = n_lif_out

            s_lif_out = torch.stack(temp_s_lif_out, dim=0).permute(1, 2, 0)  # B, E, L < L, B, E
            f_lif_out = torch.stack(temp_f_lif_out, dim=0).permute(1, 2, 0)
            n_lif_out = torch.stack(temp_n_lif_out, dim=0).permute(1, 2, 0)
                
        lif_out = torch.stack([s_lif_out, f_lif_out, n_lif_out], dim=1)  # B, 3, E, L
        #lif_out = lif_out.sum(dim=1).unsqueeze(1)  # B, 1, E, L

        lif_out = self.conv2d(lif_out)  # B, 1, E, L

        #lif_out = lif_out + activations.unsqueeze(1) + delta_activations.unsqueeze(1) # B, 1, E, L <

        return lif_out # B, 1, E, L <