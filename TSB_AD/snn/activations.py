import torch
from torch import nn

import snntorch as snn
from snntorch import surrogate

import math

class SpikeActivation(nn.Module):
    def __init__(self, local_running_params, num_features, ndim=3):
        super(SpikeActivation, self).__init__()

        assert local_running_params is not None, "local_running_params cannot be None."
        self.local_running_params = local_running_params

        self.num_features = num_features
        self.num_enc_features = self.local_running_params['num_enc_features']
        self.window_size = self.local_running_params['window_size']
        self.num_steps = self.local_running_params['activations']['common']['num_steps']
        self.beta = self.local_running_params['activations']['common']['beta']
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
        
        self.bntt = snn._layers.BatchNormTT1d(input_features=self.num_features, time_steps=self.num_steps)
        self.lif = snn.Leaky(beta=self.beta, 
                             spike_grad=surrogate.atan(alpha=2.0), 
                             init_hidden=True,
                             output=False,
                             threshold=self.init_threshold,
                             learn_threshold=self.learn_threshold
                             )
    
    def forward(self, activations): # <- (B, C, L) or (B, T, C, L)
        spk_rec = []
        for step in range(self.num_steps):
            # BNTT
            if self.local_running_params['activations']['binary']['bntt']:
                activations = self.bntt[step](activations)
            spike = self.lif(activations)
            # SCoF
            if self.local_running_params['activations']['binary']['second_chance']:
                spike = self.second_chance_of_firing(activations, spike)
            spk_rec.append(spike)
        spike = torch.stack(spk_rec, dim=-1) # <- (B, C, L, T)
        spike = torch.mean(spike, dim=-1) # <- (B, C, L)
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
    
class TernarySpikeActivation(nn.Module):
    def __init__(self, local_running_params):
        super(TernarySpikeActivation, self).__init__()
        
        assert local_running_params is not None, "local_running_params cannot be None."
        self.local_running_params = local_running_params

        self.num_steps = self.local_running_params['activations']['common']['num_steps']
        self.beta = self.local_running_params['activations']['common']['beta']
        self.pos_threshold = self.local_running_params['activations']['ternary']['pos_threshold'] # initial positive threshold
        self.neg_threshold = self.local_running_params['activations']['ternary']['neg_threshold'] # initial negative threshold

        # Initialize the TernarySpikingNeuron
        self.t_lif = TernarySpikingNeuron(beta=self.beta,
                                        pos_threshold=self.pos_threshold,
                                        neg_threshold=self.neg_threshold)
        
    def forward(self, activations, reduce=True, mem_out=False):
        spk_rec, mem_rec = [], []
        for step in range(self.num_steps):
            spike = self.t_lif(activations)
            spk_rec.append(spike)
            mem_rec.append(self.t_lif.mem)

        spike = torch.stack(spk_rec, dim=-1) # <- (B, C, L, T)
        mem = torch.stack(mem_rec, dim=-1)
        if reduce:
            spike = torch.mean(spike, dim=-1) # <- (B, C, L)
            mem = torch.mean(mem, dim=-1)
        if mem_out:
            return spike, mem
        return spike


if __name__ == "__main__":
    '''
    ternary_lif = TernarySpikingNeuron(beta=0.3, reset_mechanism="subtract", init_hidden=True,)
    x_1 = torch.tensor([-1.5 , -1.1, 0.5, 1.3, 0.0])
    x_2 = torch.tensor([-0.5 , -0.1, 0.5, 1.3, 0.0])
    x_3 = torch.tensor([5.0 , -0.1, 0.5, 1.3, 0.0])
    x_4 = torch.tensor([-2.1 , -0.1, 0.5, 1.3, 0.0])
    x = [x_1, x_2, x_3, x_4]
    
    for i in range(len(x)):
        spk = ternary_lif(x[i])
        #print(f"Input: {x[i]}")
        print(f"Spike: {spk} | Mem: {ternary_lif.mem}")
    '''

    from params import running_params
    t_lif = TernarySpikeActivation(local_running_params=running_params)
    inputs = torch.randn(1, 2, 10) # (B, C, L)
    outputs = t_lif(inputs)
    
    print(outputs)