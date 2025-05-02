import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import math

class BaseEncoder(nn.Module):
    def __init__(self, num_raw_features: int, window_size: int = 50, local_running_params=None):
        super().__init__()

        self.local_running_params = local_running_params

        self.num_raw_features = num_raw_features
        self.num_enc_features = self.local_running_params['num_enc_features']
        self.learn_threshold = self.local_running_params['encoders']['learn_threshold']
        self.window_size = window_size

        if self.local_running_params['encoders']['normalization_layer']['type'] == 'bn':
            self.norm = nn.BatchNorm2d(self.num_enc_features)
        elif self.local_running_params['encoders']['normalization_layer']['type'] == 'ln':
            E, C, L = self.num_enc_features, self.num_raw_features, self.window_size
            self.norm = nn.LayerNorm((E, C, L))
        elif self.local_running_params['encoders']['normalization_layer']['type'] == 'gn':
            self.norm = nn.GroupNorm(self.local_running_params['encoders']['normalization_layer']['gn']['num_groups'], self.num_enc_features)
        else:
            raise ValueError(f"Invalid normalization layer: {self.local_running_params['encoders']['normalization_layer']}. Choose 'bn', 'ln', or 'gn'.")

        def shape_by_granularity(granularity: str):
            if granularity == 'channel':
                return (self.num_enc_features, self.num_raw_features, 1)
            elif granularity == 'neuron':
                return (self.num_enc_features, self.num_raw_features, self.window_size)
            else:
                raise ValueError(f"Invalid granularity: {granularity}. Choose ['channel', 'neuron'].")

        if self.local_running_params['encoders']['threshold_init'] == 'all-1s':
            shape = shape_by_granularity(self.local_running_params['encoders']['granularity'])
            self.init_threshold = torch.ones(size=shape)

        elif self.local_running_params['encoders']['threshold_init'] == 'all-0s':
            shape = shape_by_granularity(self.local_running_params['encoders']['granularity'])
            self.init_threshold = torch.zeros(size=shape)

        elif self.local_running_params['encoders']['threshold_init'] == 'random':
            shape = shape_by_granularity(self.local_running_params['encoders']['granularity'])
            self.init_threshold = torch.rand(size=shape)

        elif self.local_running_params['encoders']['threshold_init'] == 'he':
            shape = shape_by_granularity(self.local_running_params['encoders']['granularity'])
            E = shape[0] # num_enc_features
            C = shape[1] # num_raw_features
            L = shape[2] # window_size
            fan_in = C * L
            std = math.sqrt(2.0 / fan_in)
            self.init_threshold = torch.empty(E, C, L).normal_(mean=0.0, std=std)

        elif self.local_running_params['encoders']['threshold_init'] == 'scalar':
            self.init_threshold = 1.0

        self.lif = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=False,
            threshold=self.init_threshold,
            learn_threshold=self.learn_threshold,
        )

        self.second_chance = self.local_running_params['encoders']['second_chance']
        self.sub_threshold_type = self.local_running_params['encoders']['sub_threshold']['type']
        self.supra_threshold_type = self.local_running_params['encoders']['supra_threshold']['type']

    def forward(self, inputs: torch.Tensor):
        raise NotImplementedError

    def second_chance_of_firing(self, activations: torch.Tensor, spikes: torch.Tensor):
        """
        Args:
            activations: (B, E, C, L) feature map after Conv/BN
            mode: 'linear', 'exponential', 'gaussian'
            ternary: whether to enable ternary spike (+1, +2)
            kwargs: mode-specific parameters
                - simoid
                    - eps: small value to avoid division by zero
                - exponential
                    - alpha: scaling factor for exponential decay
                - gaussian
                    - sigma: standard deviation for Gaussian distribution
                - ternary
                    - adaptive_margin: whether to use adaptive margin based on batch std
                    - scale_factor: scaling factor for adaptive margin
                    - margin: fixed margin for ternary spike
        """
        threshold = self.lif.threshold

        distance = torch.abs(activations - threshold) 
        sign = torch.sign(activations - threshold) # 음수: threshold 미달, 양수: threshold 초과
        sub_threshold_fire_prob = torch.zeros_like(activations)

        if self.sub_threshold_type == 'linear':
            # margin 설정
            if self.local_running_params['encoders']['sub_threshold']['linear']['adaptive_margin']:
                batch_std = activations.std()
                scale_factor = self.local_running_params['encoders']['sub_threshold']['linear']['scale_factor']
                margin = batch_std * scale_factor
            else:
                margin = self.local_running_params['encoders']['sub_threshold']['linear']['margin']
            eps = self.local_running_params['encoders']['sub_threshold']['linear']['eps']
            sub_threshold_fire_prob = (distance / (margin + eps)).clamp(0, 1)

        elif self.sub_threshold_type == 'exponential':
            alpha = self.local_running_params['encoders']['sub_threshold']['exponential']['alpha']
            sub_threshold_fire_prob = torch.exp(-alpha * distance)

        elif self.sub_threshold_type == 'gaussian':
            sigma = self.local_running_params['encoders']['sub_threshold']['gaussian']['sigma']
            sub_threshold_fire_prob = torch.exp(-0.5 * (distance / sigma) ** 2)

        else:
            raise ValueError(f"Invalid mode: {self.sub_threshold_type}. Choose ['linear', 'exponential', 'gaussian'].")

        # threshold 초과 한 경우는 sub_threshold_fire_prob를 0으로 설정
        sub_threshold_fire_prob = sub_threshold_fire_prob * (sign < 0).float()
        second_spikes = torch.bernoulli(sub_threshold_fire_prob)
        spikes = spikes + second_spikes

        if self.local_running_params['encoders']['ternary']:

            if self.supra_threshold_type == 'linear':
                if self.local_running_params['encoders']['supra_threshold']['linear']['adaptive_margin']:
                    batch_std = activations.std()
                    scale_factor = self.local_running_params['encoders']['supra_threshold']['linear']['scale_factor']
                    margin = batch_std * scale_factor
                else:
                    margin = self.local_running_params['encoders']['supra_threshold']['linear']['margin']  # 고정 margin 사용
                eps = self.local_running_params['encoders']['supra_threshold']['linear']['eps']
                supra_threshold_fire_prob = (distance / (margin + eps)).clamp(0, 1)

            elif self.supra_threshold_type == 'exponential':
                alpha = self.local_running_params['encoders']['supra_threshold']['exponential']['alpha']  # decay rate
                supra_threshold_fire_prob = 1.0 - torch.exp(-alpha * distance)

            elif self.supra_threshold_type == 'gaussian':
                sigma = self.local_running_params['encoders']['supra_threshold']['gaussian']['sigma']
                supra_threshold_fire_prob = 1.0 - torch.exp(-0.5 * (distance / sigma) ** 2)

            else:
                raise ValueError(f"Invalid strong_mode: {self.supra_threshold_type}. Choose 'linear', 'exponential', or 'gaussian'.")

            # threshold 미달한 경우는 supra_threshold_fire_prob를 0으로 설정
            supra_threshold_fire_prob = supra_threshold_fire_prob * (sign > 0).float()

            # 확률적으로 1→2 강화
            ternary_spikes = torch.bernoulli(supra_threshold_fire_prob)

            spikes = spikes + ternary_spikes

            spikes = spikes.clamp(0, 2)  # 스파이크 수를 0, 1, 2로 제한

        return spikes

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
    def __init__(self, num_raw_features: int, local_running_params=None):
        super().__init__(num_raw_features=num_raw_features, local_running_params=local_running_params)

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.repeat(
            tuple([self.num_enc_features] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )  # num_enc_features batch L C
        inputs = inputs.permute(1, 0, 3, 2)  # (B, E, C, L)

        activations = self.norm(inputs)  # (B, E, C, L)

        spikes = self.lif(activations)

        if self.second_chance:
            spikes = self.second_chance_of_firing(activations, spikes)

        return spikes # (B, E, C, L) <-

class DeltaEncoder(BaseEncoder):
    def __init__(self, num_raw_features: int, local_running_params=None):
        super().__init__(num_raw_features=num_raw_features, local_running_params=local_running_params)
        
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
        spikes = self.lif(activations)

        if self.second_chance:
            spikes = self.second_chance_of_firing(activations, spikes)

        if outputs:
            return spikes, activations  # (B, E, C, L), (B, E, C, L)
        else:
            return spikes # (B, E, C, L)

class ConvEncoder(BaseEncoder):
    def __init__(self, num_raw_features: int,
                 kernel_size: int = 3, dilation: int = 1,
                 local_running_params=None):
        super().__init__(num_raw_features=num_raw_features, local_running_params=local_running_params)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.num_enc_features,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
                dilation=dilation,
            ),
        )

    def forward(self, inputs: torch.Tensor, outputs=False):
        # inputs: batch, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # batch, 1, C, L

        activations = self.encoder(inputs)  # batch, num_enc_features, C, L

        activations = self.norm(activations)  # batch, num_enc_features, C, L

        spikes = self.lif(activations)

        if self.second_chance:
            spikes = self.second_chance_of_firing(activations, spikes)

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