import torch
from torch import nn
from torch.nn import functional as F

import snntorch as snn
from snntorch import surrogate
from spikingjelly.activation_based import neuron

import math

class BaseSpikeActivation(nn.Module):
    def __init__(self, local_running_params, num_features, ndim=3, layer_type='core'):
        super(BaseSpikeActivation, self).__init__()
        
        assert local_running_params is not None, "local_running_params cannot be None."
        self.local_running_params = local_running_params

        self.num_features = num_features
        self.num_enc_features = self.local_running_params['num_enc_features']
        self.window_size = self.local_running_params['window_size']

        self.num_steps = self.local_running_params['activations']['common']['num_steps']
        self.beta = self.local_running_params['activations']['common']['beta']

        self.ndim = ndim
        self.layer_type = layer_type

        # Normalization layer initialization
        if self.ndim == 3:
            C, L = self.num_features, self.window_size
            norm_type = self.local_running_params['normalization_layer']['type']
            if norm_type == 'bn':
                self.norm = nn.BatchNorm1d(num_features=self.num_features)
            elif norm_type == 'gn':
                self.norm = nn.GroupNorm(num_groups=self.local_running_params['normalization_layer']['gn']['num_groups'], num_channels=self.num_features)
            elif norm_type == 'ln':
                self.norm = nn.LayerNorm(normalized_shape=(C, L))
            elif norm_type == 'bntt':
                self.norm = snn._layers.BatchNormTT1d(input_features=self.num_features, time_steps=self.num_steps)
        elif self.ndim == 4:
            E, C, L = self.num_enc_features, self.num_features, self.window_size
            norm_type = self.local_running_params['normalization_layer']['type']
            if norm_type == 'bn':
                self.norm = nn.BatchNorm2d(num_features=self.num_enc_features)
            elif norm_type == 'gn':
                self.norm = nn.GroupNorm(num_groups=self.local_running_params['normalization_layer']['gn']['num_groups'], num_channels=self.num_enc_features)
            elif norm_type == 'ln':
                self.norm = nn.LayerNorm(normalized_shape=(E, C, L))
            elif norm_type == 'bntt':
                self.norm = snn._layers.BatchNormTT2d(input_features=self.num_enc_features, time_steps=self.num_steps)

    def forward(self):
        raise NotImplementedError("The forward method must be implemented in the derived class.")
    
class DynamicReceptiveSpikeActivation(BaseSpikeActivation):
    def __init__(self, local_running_params, num_features, ndim=3, layer_type='core'):
        super(DynamicReceptiveSpikeActivation, self).__init__(local_running_params, num_features, ndim, layer_type)

        assert local_running_params is not None, "local_running_params cannot be None."

        self.learn_threshold = self.local_running_params['activations']['dynamic_receptive']['learn_threshold']
        self.learn_beta = self.local_running_params['activations']['dynamic_receptive']['learn_beta']
        self.reset_mechanism = self.local_running_params['activations']['dynamic_receptive']['reset_mechanism']
        self.integration = self.local_running_params['activations']['dynamic_receptive']['integration']

        #self.register_buffer('base_thr_s', torch.ones((self.batch_size, self.num_enc_features)) * 0.5) # (E)
        #self.register_buffer('base_thr_f', torch.ones((self.batch_size, self.num_enc_features)) * 0.9)
        #self.register_buffer('base_thr_n', torch.ones((self.batch_size, self.num_enc_features)) * 3.0)

        if self.local_running_params['activations']['dynamic_receptive']['granularity'] == 'scalar':
            self.base_thr_s = 0.5
            self.base_thr_f = 0.9
            self.base_thr_n = 3.0
        elif self.local_running_params['activations']['dynamic_receptive']['granularity'] == 'neuron':
            self.base_thr_s = 0.5 * torch.ones((self.num_enc_features,))  # (E)
            self.base_thr_f = 0.9 * torch.ones((self.num_enc_features,))
            self.base_thr_n = 3.0 * torch.ones((self.num_enc_features,))

        self.base_beta_s = 0.85
        self.base_beta_f = 0.7
        self.base_beta_n = 0.95
        
        # Temporarily excluded
        #self.adapt_gain = 1.6
        #self.adapt_recovery = 0.95
        #self.sensitization_decay = 0.9
        #self.sensitization_recovery = 0.95

        self.s_lif = snn.Leaky(beta=self.base_beta_s, threshold=self.base_thr_s, spike_grad=surrogate.atan(alpha=2.0),
                               reset_mechanism=self.reset_mechanism, init_hidden=True, output=False,
                               learn_threshold=self.learn_threshold, learn_beta=self.learn_beta)
        self.f_lif = snn.Leaky(beta=self.base_beta_f, threshold=self.base_thr_f, spike_grad=surrogate.atan(alpha=2.0),
                               reset_mechanism=self.reset_mechanism, init_hidden=True, output=False,
                               learn_threshold=self.learn_threshold, learn_beta=self.learn_beta)
        self.n_lif = snn.Leaky(beta=self.base_beta_n, threshold=self.base_thr_n, spike_grad=surrogate.atan(alpha=2.0),
                               reset_mechanism=self.reset_mechanism, init_hidden=True, output=False,
                               learn_threshold=self.learn_threshold, learn_beta=self.learn_beta)
        
        self.register_buffer('burst_kernel', torch.tensor([1, 1, 1], dtype=torch.float32).view(1, 1, 3)) # shape: (1, 1, 3)
        self.burst_kernel = self.burst_kernel.repeat(self.num_enc_features, 1, 1)  # shape: (E, 1, 3)

        self.integration_kernel = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.integration_kernel.weight.data.fill_(1.0)  # Initialize weights to 1
        self.integration_kernel.bias.data.fill_(0.0)  # Initialize bias to 0

    def forward(self, activations): # <- (B, E, L)

        activations = activations.permute(2, 0, 1) # (L, B, E) <-

        num_steps = activations.size(0)

        #self.threshold_reset()

        s_spk_rec, f_spk_rec, n_spk_rec = [], [], []
        for step in range(num_steps):
            s_spike = self.s_lif(activations[step]) # (B, E) <-
            
            if step == 0:
                delta_activation = activations[step]
            else:
                delta_activation = activations[step] - activations[step-1]

            if self.local_running_params['activations']['dynamic_receptive']['delta_activation']:
                delta_activation = torch.abs(delta_activation)

            f_spike = self.f_lif(delta_activation)

            n_spike = self.n_lif(activations[step])

            s_spk_rec.append(s_spike)
            f_spk_rec.append(f_spike)
            n_spk_rec.append(n_spike)

            # Adaptation (Temporaily excluded)
            if self.local_running_params['activations']['dynamic_receptive']['adaptation']:
                self.adaptation(s_spike)
            # Sesntization (Temporaily excluded)
            if self.local_running_params['activations']['dynamic_receptive']['sensitization']:
                self.sensitization(n_spike)

            s_spike = torch.stack(s_spk_rec, dim=-1) # (B, E, L) <- (B, E) * T
            f_spike = torch.stack(f_spk_rec, dim=-1)
            n_spike = torch.stack(n_spk_rec, dim=-1)

        # Burst modeling over each channels
        if self.local_running_params['activations']['dynamic_receptive']['burst']:
            n_spike = F.conv1d(n_spike, self.burst_kernel, padding=1, groups=n_spike.shape[1]) # (B, E, L) <- (B, E, L) * (1, 3)
            n_spike = (n_spike > 0).float() # (B, E, L) <- (B, E, L) * (1, 3)

        # Spike integration
        if self.integration == 'sum':
            spike = s_spike + f_spike + n_spike # (B, E, L) <- (B, E, L) + (B, E, L) + (B, E, L)
        elif self.integration == 'concat':
            spike = torch.cat((s_spike, f_spike, n_spike), dim=1) # (B, E*3, L) <- (B, E, L) + (B, E, L) + (B, E, L)
            spike_reshaped = spike.view(spike.shape[0], 3, self.num_enc_features, self.window_size) # (B, 3, E, L)
            spike_reshaped = self.integration_kernel(spike_reshaped) # (B, 1, E, L) <- (B, 3, E, L)
            spike = spike_reshaped.squeeze(1) # (B, E, L) <- (B, 1, E, L)

        return spike

    def adaptation(self, spike): # (B, E)

        thresholds = self.s_lif.threshold # (E)
        base_thr_s = self.base_thr_s # (E)

        # flatten
        thresholds_flat = thresholds.view(-1) # (B*E,)
        base_thr_s_flat = base_thr_s.view(-1) # (B*E,)
        spike_flat = spike.view(-1) # (B*E,)

        # Get the indices where spike_reduce > 0.5
        #spike_reduce = torch.mean(spike, dim=0) # (E,)
        #indices_1 = (spike_reduce > 0.5).int() # (E,)
        #indices_0 = (spike_reduce <= 0.5).int() # (E,)

        indices_1 = torch.nonzero(spike_flat == 1, as_tuple=False).squeeze()  # index of spikes (1)
        indices_0 = torch.nonzero(spike_flat == 0, as_tuple=False).squeeze()  # index of no spikes (0)

        # handle case when only one index (squeeze makes it scalar)        
        if indices_1.dim() == 0:
            indices_1 = indices_1.unsqueeze(0)
        if indices_0.dim() == 0:
            indices_0 = indices_0.unsqueeze(0)
        
        thresholds_flat[indices_1] = thresholds_flat[indices_1] * self.adapt_gain
        recovery = base_thr_s_flat[indices_0] + (thresholds_flat[indices_0] - base_thr_s_flat[indices_0]) * self.adapt_recovery
        thresholds_flat[indices_0] = torch.max(thresholds_flat[indices_0], recovery)  # 선택적으로 최대값 제한

        self.s_lif.threshold = thresholds_flat.view(self.batch_size, self.num_enc_features) # (B, E)
        #self.s_lif.threshold = thresholds # (E)

    def sensitization(self, spike):
        
        thresholds = self.n_lif.threshold # (B, E)
        base_thr_n = self.base_thr_n # (B, E)
        
        # flatten
        thresholds_flat = thresholds.view(-1) # (B*E,)
        base_thr_n_flat = base_thr_n.view(-1) # (B*E,)
        spike_flat = spike.view(-1) # (B*E,)

        # Get the indices where spike_reduce > 0.5
        #spike_reduce = torch.mean(spike, dim=0) # (E,)
        #indices_1 = (spike_reduce > 0.5).int() # (E,)
        #indices_0 = (spike_reduce <= 0.5).int() # (E,)

        indices_1 = torch.nonzero(spike_flat == 1, as_tuple=False).squeeze()  # index of spikes (1)
        indices_0 = torch.nonzero(spike_flat == 0, as_tuple=False).squeeze()  # index of no spikes (0)

        # handle case when only one index (squeeze makes it scalar)
        if indices_1.dim() == 0:
            indices_1 = indices_1.unsqueeze(0)
        if indices_0.dim() == 0:
            indices_0 = indices_0.unsqueeze(0)
        
        # Sensitize the threshold for 1s
        thresholds_flat[indices_1] = thresholds_flat[indices_1] * self.sensitization_decay
        # Recover the threshold for 0s
        recovery = base_thr_n_flat[indices_0] + (thresholds_flat[indices_0] - base_thr_n_flat[indices_0]) * self.sensitization_recovery
        thresholds_flat[indices_0] = torch.min(thresholds_flat[indices_0], recovery)  # 선택적으로 최대값 제한

        self.n_lif.threshold = thresholds_flat.view(self.batch_size, self.num_enc_features) # (B, E)
        #self.n_lif.threshold = thresholds # (E)

    def threshold_reset(self):
        self.s_lif.threshold = self.base_thr_s
        self.f_lif.threshold = self.base_thr_f
        self.n_lif.threshold = self.base_thr_n

class PSNActivation(BaseSpikeActivation):
    def __init__(self, local_running_params, num_features, ndim=3, layer_type='core'):
        super(PSNActivation, self).__init__(local_running_params=local_running_params, num_features=num_features, ndim=ndim, layer_type=layer_type)
        
        self.lif = neuron.PSN()

    def forward(self, activations):
        pass


class SpikeActivation(BaseSpikeActivation):
    def __init__(self, local_running_params, num_features, ndim=3, layer_type='core'):
        super(SpikeActivation, self).__init__(local_running_params, num_features, ndim, layer_type)

        assert local_running_params is not None, "local_running_params cannot be None."

        self.learn_threshold = self.local_running_params['activations']['binary']['learn_threshold']

        def shape_by_granularity(granularity: str, ndim: int):
            if ndim == 3:
                if granularity == 'channel':
                    return (self.num_features, 1)
                elif granularity == 'neuron':
                    return (self.num_features, self.window_size)
                else:
                    raise ValueError(f"Invalid granularity: {granularity}. Choose ['channel', 'neuron'].")
            elif ndim == 4:
                if granularity == 'channel':
                    return (self.num_enc_features, self.num_features, 1)
                elif granularity == 'neuron':
                    return (self.num_enc_features, self.num_features, self.window_size)
                else:
                    raise ValueError(f"Invalid granularity: {granularity}. Choose ['channel', 'neuron'].")

        shape = shape_by_granularity(self.local_running_params['activations']['binary']['granularity'], ndim)
        if self.local_running_params['activations']['binary']['threshold_init'] == 'all-1s':
            self.init_threshold = torch.ones(size=shape)
        elif self.local_running_params['activations']['binary']['threshold_init'] == 'all-0s':
            self.init_threshold = torch.zeros(size=shape)
        elif self.local_running_params['activations']['binary']['threshold_init'] == 'random':
            self.init_threshold = torch.rand(size=shape)
        elif self.local_running_params['activations']['binary']['threshold_init'] == 'he':
            if ndim == 3:
                C = shape[0] # num_features
                L = shape[1] # window_size
                fan_in = C * L
                std = math.sqrt(2.0 / fan_in)
                self.init_threshold = torch.empty(C, L).normal_(mean=0.0, std=std)
            elif ndim == 4:
                E = shape[0] # num_enc_features
                C = shape[1] # num_features
                L = shape[2] # window_size
                fan_in = C * L
                std = math.sqrt(2.0 / fan_in)
                self.init_threshold = torch.empty(C, L).normal_(mean=0.0, std=std)
        elif self.local_running_params['activations']['binary']['threshold_init'] == 'scalar':
            self.init_threshold = 1.0
        assert self.init_threshold is not None, "Invalid threshold_init value. Choose ['all-1s', 'all-0s', 'random', 'he', 'scalar']."
        
        self.lif = snn.Leaky(beta=self.beta, 
                             spike_grad=surrogate.atan(alpha=2.0), 
                             init_hidden=True,
                             output=False,
                             threshold=self.init_threshold,
                             learn_threshold=self.learn_threshold
                             )
    
    def forward(self, activations): # <- (B, C, L) or (B, T, C, L)
        
        if self.local_running_params['normalization_layer']['type'] in ['bn', 'ln', 'gn']:
            activations = self.norm(activations)
            if self.local_running_params['activations']['binary']['adaptive']:
                gamma = self.norm.weight.abs().mean()
                beta = self.norm.bias.mean()
                self.lif.threshold = beta + gamma

        if self.layer_type == 'core':
            spk_rec = []
            for step in range(self.num_steps):
                # BNTT
                if self.local_running_params['normalization_layer']['type'] == 'bntt':
                    activations = self.norm[step](activations)
                    
                spike = self.lif(activations)
                # SCoF
                if self.local_running_params['activations']['binary']['second_chance']:
                    spike = self.second_chance_of_firing(activations, spike)
                spk_rec.append(spike)
            spike = torch.stack(spk_rec, dim=-1) # <- (B, C, L, T)
            spike = torch.mean(spike, dim=-1) # <- (B, C, L)
            return spike
        else:
            spike = self.lif(activations)
            return spike
    
    def second_chance_of_firing(self, activations: torch.Tensor, spike: torch.Tensor):
        threshold = self.lif.threshold

        distance = torch.abs(activations - threshold)
        sign = torch.sign(activations - threshold)
        
        # sub-threshold firing
        ''' exponential
        alpha_sub = 5.0
        sub_threshold_fire_prob = torch.exp(-alpha_sub * distance)
        '''
        
        sub_threshold_fire_prob = sub_threshold_fire_prob * (sign < 0).float()
        second_spike = torch.bernoulli(sub_threshold_fire_prob)
        spike = spike + second_spike

        # supra-threshold firing
        ''' exponential
        alpha_supra = 5.0
        supra_threshold_fire_prob = 1.0 - torch.exp(-alpha_supra * distance)
        '''

        supra_threshold_fire_prob = supra_threshold_fire_prob * (sign > 0).float()
        ternary_spike = torch.bernoulli(supra_threshold_fire_prob)
        spike = spike + ternary_spike

        spike = torch.clamp(spike, 0, 2)

        return spike
    
class TernarySpikingNeuron(snn._neurons.leaky.Leaky):
    def __init__(self, beta, pos_threshold, neg_threshold, spike_grad=None,
                 surrogate_disable=False, init_hidden=True, inhibition=False,
                 learn_beta=False, reset_mechanism="subtract",
                 state_quant=False, output=False, graded_spikes_factor=1.0,
                 learn_graded_spikes_factor=False, reset_delay=True):
        super(TernarySpikingNeuron, self).__init__(beta=beta,
                                                     threshold=1.0,
                                                     spike_grad=spike_grad,
                                                     surrogate_disable=surrogate_disable,
                                                     init_hidden=init_hidden,
                                                     inhibition=inhibition,
                                                     learn_beta=learn_beta,
                                                     learn_threshold=False,
                                                     reset_mechanism=reset_mechanism,
                                                     state_quant=state_quant,
                                                     output=output,
                                                     graded_spikes_factor=graded_spikes_factor,
                                                     learn_graded_spikes_factor=learn_graded_spikes_factor,
                                                     reset_delay=reset_delay)
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
    
    def fire(self, mem):
        if self.state_quant:
            mem = self.state_quant(mem)

        mem_shift_pos = mem - self.pos_threshold
        mem_shift_neg = self.neg_threshold - mem

        spk_pos = self.spike_grad(mem_shift_pos)
        spk_neg = self.spike_grad(mem_shift_neg)

        spk = spk_pos - spk_neg
        spk = spk * self.graded_spikes_factor
        return spk
    
class TernarySpikeActivation(BaseSpikeActivation):
    def __init__(self, local_running_params, num_features, ndim=3, layer_type='core'):
        super(TernarySpikeActivation, self).__init__(local_running_params, num_features, ndim, layer_type)
        
        assert local_running_params is not None, "local_running_params cannot be None."

        self.pos_threshold = self.local_running_params['activations']['ternary']['pos_threshold'] # initial positive threshold
        self.neg_threshold = self.local_running_params['activations']['ternary']['neg_threshold'] # initial negative threshold

        # Initialize the TernarySpikingNeuron
        self.t_lif = TernarySpikingNeuron(beta=self.beta,
                                        pos_threshold=self.pos_threshold,
                                        neg_threshold=self.neg_threshold)
        
    def forward(self, activations, reduce=True): # <- (B, C, L) or (B, E, C, L)

        if self.ndim != -1:
            if self.local_running_params['normalization_layer']['type'] in ['bn', 'ln', 'gn']:
                activations = self.norm(activations)
                if self.local_running_params['activations']['ternary']['adaptive']:
                    gamma = self.norm.weight.abs().mean()
                    beta = self.norm.bias.mean()
                    self.t_lif.pos_threshold = beta + gamma
                    self.t_lif.neg_threshold = beta - gamma

        if self.layer_type == 'core':
            spk_rec, mem_rec = [], []
            for step in range(self.num_steps):
                if self.local_running_params['normalization_layer']['type'] == 'bntt':
                    activations = self.norm[step](activations)
                spike = self.t_lif(activations)
                spk_rec.append(spike)
                mem_rec.append(self.t_lif.mem)

            spike = torch.stack(spk_rec, dim=-1) # <- (B, C, L, T)
            mem = torch.stack(mem_rec, dim=-1)
            if reduce:
                spike = torch.mean(spike, dim=-1) # <- (B, C, L)
                mem = torch.mean(mem, dim=-1)
            return spike
        else:
            spike = self.t_lif(activations)
            return spike
        

if __name__ == "__main__":
    
    # print spikingjelly version from pip
    from importlib.metadata import version
    print("spikingjelly version:", version("spikingjelly"))