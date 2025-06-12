import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer
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

    def forward(self, inputs: torch.Tensor):
        # IN: B, L, C
        inputs = inputs.permute(0, 2, 1) # B, C, L <
        activations = self.transducer(inputs) # B, E, L <
        #activations = self.norm(activations)

        if self.local_running_params['ParallelSNNModel']['dropout']:
            activations = self.dropout(activations)  # B, E, L <

        delta_activations = torch.zeros_like(activations)
        delta_activations[:, :, 1:] = activations[:, :, 1:] - activations[:, :, :-1] # B, E, L-1 <
        delta_activations[:, :, 0] = 0.0

        if self.local_running_params['ParallelSNNModel']['delta_abs']:
            delta_activations = torch.abs(delta_activations)

        # F-LIF
        f_lif_out = self.f_lif(delta_activations.permute(2, 0, 1)).permute(1, 2, 0)
        # S-LIF
        s_lif_out = self.s_lif(activations.permute(2, 0, 1)).permute(1, 2, 0)  # L, B, E < L, B, E
        # N-LIF
        n_lif_out = self.n_lif(activations.permute(2, 0, 1)).permute(1, 2, 0) # L, B, E < L, B, E
                
        lif_out = torch.stack([s_lif_out, f_lif_out, n_lif_out], dim=1)  # B, 3, E, L

        if self.local_running_params['ParallelSNNModel']['grad_spike']:
            lif_out = self.conv2d(lif_out)  # B, 1, E, L
        else:
            lif_out = lif_out.sum(dim=1).unsqueeze(1)  # B, 1, E, L

        return lif_out # B, 1, E, L <
    
class DynamicReceptiveEncoder(BaseEncoder):
    def __init__(self, local_running_params, num_raw_features, num_enc_features):
        super().__init__(local_running_params)

        kernel_3 = 3
        kernel_7 = 7
        self.transducer_3 = nn.Conv2d(
            in_channels=1,
            out_channels=num_enc_features,
            kernel_size=kernel_3,
            stride=1,
            padding=(kernel_3 // 2, kernel_3 // 2),
        )
        self.transducer_7 = nn.Conv2d(
            in_channels=1,
            out_channels=num_enc_features,
            kernel_size=kernel_7,
            stride=1,
            padding=(kernel_7 // 2, kernel_7 // 2),
        )
        self.norm = nn.LayerNorm((local_running_params['model']['window_size']))

        # SA-I
        self.sa_i = neuron.SimpleLIFNode(tau=20., decay_input=False, v_threshold=1.)
        # SA-II
        self.sa_ii = neuron.SimpleLIFNode(tau=50., decay_input=False, v_threshold=1.)
        # FA-I
        self.fa_i = neuron.SimpleLIFNode(tau=2., decay_input=False, v_threshold=0.8)
        # FA-II
        self.fa_ii = neuron.SimpleLIFNode(tau=0.91, decay_input=False, v_threshold=0.8) # TODO: Izhikevich
        

    def forward(self, inputs: torch.Tensor):
        # IN: B, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1) # B, 1, C, L <
        activations_3 = self.transducer_3(inputs) # B, E, C, L <
        activations_7 = self.transducer_7(inputs) # B, E, C, L <
        #activations = self.norm(activations)

        delta_activations_3 = torch.zeros_like(activations_3)
        delta_activations_3[:, :, :, 1:] = activations_3[:, :, :, 1:] - activations_3[:, :, :, :-1]
        delta_activations_3[:, :, :, 0] = 0.0
        delta_activations_3 = torch.abs(delta_activations_3)

        delta_activations_7 = torch.zeros_like(activations_7)
        delta_activations_7[:, :, :, 1:] = activations_7[:, :, :, 1:] - activations_7[:, :, :, :-1] # B, E, L-1 <
        delta_activations_7[:, :, :, 0] = 0.0
        delta_activations_7 = torch.abs(delta_activations_7)

        activations_3 = activations_3.permute(3, 0, 1, 2)
        activations_7 = activations_7.permute(3, 0, 1, 2)
        delta_activations_3 = delta_activations_3.permute(3, 0, 1, 2)
        delta_activations_7 = delta_activations_7.permute(3, 0, 1, 2)

        sa_i_rec = []
        sa_ii_rec = []
        fa_i_rec = []
        fa_ii_rec = []

        for step in range(inputs.size(3)):
            sa_i_out = self.sa_i(activations_3[step])
            sa_ii_out = self.sa_ii(activations_7[step])
            fa_i_out = self.fa_i(delta_activations_3[step])
            fa_ii_out = self.fa_ii(delta_activations_7[step])

            sa_i_rec.append(sa_i_out)
            sa_ii_rec.append(sa_ii_out)
            fa_i_rec.append(fa_i_out)
            fa_ii_rec.append(fa_ii_out)
            
        sa_i_out = torch.stack(sa_i_rec, dim=-1)  # B, E, C, L
        sa_ii_out = torch.stack(sa_ii_rec, dim=-1) # B, E, C, L
        fa_i_out = torch.stack(fa_i_rec, dim=-1)
        fa_ii_out = torch.stack(fa_ii_rec, dim=-1)

        #print(f"SA-I outputs shape: {sa_i_out.shape}")

        outputs = sa_i_out + sa_ii_out + fa_i_out + fa_ii_out # B, E, C, L

        #print(f"DynamicReceptiveEncoder outputs shape: {outputs.shape}")

        return outputs