import torch
from torch import nn

import snntorch as snn
from snntorch import surrogate

class SpikeActivation(nn.Module):
    def __init__(self, num_steps=10, beta=0.99, spike_grad=surrogate.atan(alpha=2.0), init_hidden=True, threshold=1.0):
        super(SpikeActivation, self).__init__()
        self.num_steps = num_steps
        self.beta = beta
        self.spike_grad = spike_grad
        self.init_hidden = init_hidden
        self.threshold = threshold
        self.lif = snn.Leaky(beta=self.beta, 
                             spike_grad=self.spike_grad, 
                             init_hidden=self.init_hidden, 
                             threshold=self.threshold)
    def forward(self, x): # <- (B, C, L)
        spk_rec = []
        for _ in range(self.num_steps):
            x = self.lif(x)
            spk_rec.append(x)
        x = torch.stack(spk_rec, dim=-1) # <- (B, C, L, T)
        x = torch.mean(x, dim=-1) # <- (B, C, L)
        return x