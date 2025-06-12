from typing import Dict
import torchinfo
import tqdm, math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
os.environ['WANDB_DIR'] = '/home/hwkang/dev-TSB-AD/TSB-AD/wandb'

from . import Base
from ..utils.torch_utility import EarlyStoppingTorch
from ..utils.dataset import ForecastDataset
from spikingjelly.activation_based import neuron, functional, monitor
import snntorch as snn
from snntorch import utils
from snntorch.functional import probe
from TSB_AD.snn.spikingjelly.encoders import RepeatEncoder, DeltaEncoder, ConvEncoder, ReceptiveEncoder, DynamicReceptiveEncoder
import wandb

class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.mp = torch.nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

class ParallelSNNModel(nn.Module):
    def __init__(self,
                 device='cpu',
                 num_raw_features=-1,
                 local_running_params=None):
                 
        # initialize the super class
        super(ParallelSNNModel, self).__init__()

        self.device = device
        self.local_running_params = local_running_params

        self.num_raw_features = num_raw_features # 원본 시계열 데이터의 채널 수
        self.num_enc_features = self.local_running_params['ParallelSNNModel']['num_enc_features'] # 인코딩으로 생성하는 채널 수
        self.num_out_features = num_raw_features
        self.kernel_size = 3
        self.stride = 1
        self.predict_time_steps = self.local_running_params['model']['predict_time_steps']
        self.dropout_rate = 0.25

        # Block 0: Encoder
        if self.local_running_params['meta']['Encoder_Name'] == 'receptive':
            self.encoder = ReceptiveEncoder(local_running_params, self.num_raw_features, self.num_enc_features)
        elif self.local_running_params['meta']['Encoder_Name'] == 'repeat':
            self.encoder = RepeatEncoder(local_running_params, output_size=self.num_enc_features, neuron_type=self.local_running_params['ParallelSNNModel']['neuron_type'])
        elif self.local_running_params['meta']['Encoder_Name'] == 'delta':
            self.encoder = DeltaEncoder(local_running_params, num_raw_features=num_raw_features, output_size=self.num_enc_features, neuron_type=self.local_running_params['ParallelSNNModel']['neuron_type'])
        elif self.local_running_params['meta']['Encoder_Name'] == 'conv':
            self.encoder = ConvEncoder(local_running_params, output_size=self.num_enc_features, neuron_type=self.local_running_params['ParallelSNNModel']['neuron_type'])
        elif self.local_running_params['meta']['Encoder_Name'] == 'dynamic':
            self.encoder = DynamicReceptiveEncoder(local_running_params, num_raw_features=self.num_raw_features, num_enc_features=self.num_enc_features)
        else:
            raise ValueError(f"Unsupported Encoder_Name: {self.local_running_params['meta']['Encoder_Name']}")

        # Block 1
        self.blk1_conv = nn.Conv1d(self.num_enc_features * 1 if self.local_running_params['meta']['Encoder_Name'] == 'receptive' else self.num_enc_features * self.num_raw_features, 
                                   32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        #self.blk1_bn = nn.BatchNorm1d(32)
        self.blk1_psn = neuron.SlidingPSN(k=3, step_mode='m', backend='conv')
        self.blk1_mp = nn.MaxPool1d(kernel_size=2)

        # Block 2
        self.blk2_conv = nn.Conv1d(32, 40, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        #self.blk2_bn = nn.BatchNorm1d(40)
        self.blk2_psn = neuron.SlidingPSN(k=3, step_mode='m', backend='conv')
        self.blk2_mp = nn.MaxPool1d(kernel_size=2)

        self.fc = nn.Sequential(AdaptiveConcatPool1d(),
                                torch.nn.Flatten(),
                                torch.nn.Linear(2*40, 40),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(self.dropout_rate),
                                torch.nn.Linear(40, self.num_out_features))

    def forward(self, inputs):
        if self.local_running_params['ParallelSNNModel']['neuron_type'] == 'snntorch':
            utils.reset(self.encoder)
        elif self.local_running_params['ParallelSNNModel']['neuron_type'] == 'spikingjelly':    
            functional.reset_net(self.encoder)

        # Block 0: Encoder
        encodings = self.encoder(inputs) # (B, E, C, L) < (B, L, C)

        encodings_shape = encodings.shape

        # Block 1
        activations_1 = self.blk1_conv(encodings.reshape(encodings_shape[0], encodings_shape[1]*encodings_shape[2], encodings_shape[3])) # (B, 32, L) < (B, E*C, L)
        #activations_1 = self.blk1_bn(activations_1) 
        spikes_1 = self.blk1_psn(activations_1)
        spikes_1 = self.blk1_mp(spikes_1)  # (B, 32, L/2) <

        # Block 2
        activations_2 = self.blk2_conv(spikes_1)  # (B, 40, L/2) <
        #activations_2 = self.blk2_bn(activations_2)
        spikes_2 = self.blk2_psn(activations_2)
        spikes_2 = self.blk2_mp(spikes_2)  # (B, 40, L/4) <
        
        outputs = torch.zeros(self.predict_time_steps, spikes_2.shape[0], self.num_out_features).to(self.device)
        for t in range(self.predict_time_steps):
            decoder_input = self.fc(spikes_2)
            outputs[t] = torch.squeeze(decoder_input, dim=-2)
        
        return outputs # (steps, batch, num_out_features)
    
class ParallelSNN(Base.BaseModule):
    def __init__(self, TS_Name, num_raw_features, local_running_params): # Model specific

        assert TS_Name is not None, "TS_Name must be provided"
        assert num_raw_features > 0, "num_raw_features must be greater than 0"
        assert local_running_params is not None, "local_running_params must be provided"
                 
        super().__init__(TS_Name=TS_Name, num_raw_features=num_raw_features, local_running_params=local_running_params)

        self.model = ParallelSNNModel(device=self.device, 
                                      num_raw_features=self.num_raw_features,
                                      local_running_params=local_running_params).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0008)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0.00001)
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=30)

        self.mu = None
        self.sigma = None
        self.eps = 1e-10

        self.loss_dir_path = os.path.join('/home/hwkang/dev-TSB-AD/TSB-AD/results/loss', self.local_running_params['meta']['base_file_name'])
        os.makedirs(self.loss_dir_path, exist_ok=True)
        self.spikerate_dir_path = os.path.join('/home/hwkang/dev-TSB-AD/TSB-AD/results/spikerate', self.local_running_params['meta']['base_file_name'])
        os.makedirs(self.spikerate_dir_path, exist_ok=True)
        self.spike_dir_path = os.path.join('/home/hwkang/dev-TSB-AD/TSB-AD/results/encoding', self.local_running_params['meta']['base_file_name'])
        os.makedirs(self.spike_dir_path, exist_ok=True)
        self.base_name = TS_Name.split('.')[0]

        # Monitor
        if self.local_running_params['ParallelSNNModel']['neuron_type'] == 'spikingjelly':
            pass
            #self.spikerate_monitor = monitor.OutputMonitor(net=self.model, instance=(neuron.LIFNode, neuron.SlidingPSN), function_on_output=lambda x: x.mean())
            #self.spike_monitor = monitor.OutputMonitor(net=self.model, instance=(neuron.LIFNode, neuron.SlidingPSN))
        else:
            self.spikerate_monitor = probe.OutputMonitor(net=self.model, instance=snn.Leaky, function_on_output=lambda x: x.mean())

        self.wnb = self.local_running_params['analysis']['wandb']

    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(ForecastDataset(tsTrain, window_size=self.window_size, pred_len=self.predict_time_steps),
                                  batch_size=self.batch_size,
                                  shuffle=True)
        
        valid_loader = DataLoader(ForecastDataset(tsValid, window_size=self.window_size, pred_len=self.predict_time_steps),
                                  batch_size=self.batch_size,
                                  shuffle=False)
        
        if self.wnb:
            if wandb.run is not None:
                wandb.finish()

            # W&B
            copy_of_local_running_params = self.local_running_params.copy()
            copy_of_local_running_params.pop('data', None)  # Remove 'data' key if it exists
            copy_of_local_running_params.pop('meta', None)  # Remove 'meta' key if it exists
            local_model_params = copy_of_local_running_params
            tokens = self.TS_Name.split('.')[0].split('_')
            ts_name = tokens[0] + '_' + tokens[1]
            encoder_name = self.local_running_params['meta']['Encoder_Name']
            run = wandb.init(
                project=f'MTS-AD-DynamicReceptiveEncoder',
                group=f'ParallelSNN-{encoder_name}',
                name=f'{self.id_code}_{self.postfix}_{ts_name}_{encoder_name}',
                config=local_model_params,
                reinit=True,
            )
            #print(run)
 
        train_loss_rec = []
        valid_loss_rec = []
        if self.wnb:
            wandb.watch(self.model, log='all', log_freq=1)
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            '''self.spikerate_monitor.disable()  # Disable spike monitor for training
            self.spike_monitor.disable()  # Disable spike monitor for training'''
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(x)
                output = output.view(-1, self.num_raw_features*self.predict_time_steps)
                target = target.view(-1, self.num_raw_features*self.predict_time_steps)

                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()
                
                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            train_loss = avg_loss/max(len(train_loader), 1)
            train_loss_rec.append(train_loss)
            
            self.model.eval()
            scores = []
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            '''self.spikerate_monitor.enable()'''
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)

                    '''if idx == 0:
                        self.spike_monitor.enable()
                        x_copy = x.clone() # first batch for spike monitor
                    else:
                        self.spike_monitor.disable()'''
                    
                    output = self.model(x)
                    
                    output = output.view(-1, self.num_raw_features*self.predict_time_steps)
                    target = target.view(-1, self.num_raw_features*self.predict_time_steps)
                    
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
                    
            valid_loss = avg_loss/max(len(valid_loader), 1)
            valid_loss_rec.append(valid_loss)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop or epoch == self.epochs - 1:
                # fitting Gaussian Distribution
                if len(scores) > 0:
                    scores = torch.cat(scores, dim=0)
                    self.mu = torch.mean(scores)
                    self.sigma = torch.var(scores)
                    print(self.mu.size(), self.sigma.size())
                if self.early_stopping.early_stop:
                    print("   Early stopping<<<")
                break

            if self.wnb:
                # W&B logging
                wandb.log({
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                })        

        '''
        train_loss_rec = np.array(train_loss_rec)
        valid_loss_rec = np.array(valid_loss_rec)
        np.save(os.path.join(self.loss_dir_path, f'{self.base_name}_train.npy'), train_loss_rec)
        np.save(os.path.join(self.loss_dir_path, f'{self.base_name}_valid.npy'), valid_loss_rec)

        if self.local_running_params['ParallelSNNModel']['neuron_type'] == 'spikingjelly':
            if self.local_running_params['analysis']['spikerate']:
                for layer_idx, layer in enumerate(self.spikerate_monitor.monitored_layers):
                    # record will be tensor with shape (epochs * batch_size)
                    record = torch.stack(self.spikerate_monitor[layer])
                    record = record.cpu().numpy()
                    # replace '.' with '-' and '_' with '-'
                    layer = layer.replace('.', '-').replace('_', '-')
                    np.save(os.path.join(self.spikerate_dir_path, f'{self.base_name}_spikerate_{layer_idx}_{layer}.npy'), record)
            elif self.local_running_params['analysis']['spike']:
                assert self.local_running_params['model']['batch_size'] <= 16, "Batch size should be less than 16 for spike monitor to work properly."
                if self.local_running_params['meta']['Encoder_Name'] == 'receptive':
                    spike_s_first = self.spike_monitor['encoder.s_lif'][0]
                    spike_f_first = self.spike_monitor['encoder.f_lif'][0]
                    spike_n_first = self.spike_monitor['encoder.n_lif'][0]
                    spike_s_last = self.spike_monitor['encoder.s_lif'][-1]
                    spike_f_last = self.spike_monitor['encoder.f_lif'][-1]
                    spike_n_last = self.spike_monitor['encoder.n_lif'][-1]

                    encoding_first = spike_s_first + spike_f_first + spike_n_first
                    encoding_last = spike_s_last + spike_f_last + spike_n_last

                    encoding_first = encoding_first.permute(1, 2, 0) # (B, C, L)
                    encoding_last = encoding_last.permute(1, 2, 0) # (B, C, L)
                else:
                    encodings_first = self.spike_monitor['encoder.lif'][0]
                    encodings_last = self.spike_monitor['encoder.lif'][-1]

                    encoding_first = encodings_first.sum(dim=0) # (B, C, L)
                    encoding_last = encodings_last.sum(dim=0) # (B, C, L)

                    print(f'encodings_first shape: {encodings_first.shape}, encodings_last shape: {encodings_last.shape}')
                    
                encoding_first = encoding_first.cpu().numpy() # (B, L, C) < (B, C, L)
                encoding_last = encoding_last.cpu().numpy()
                print(f'encoding_first shape: {encoding_first.shape}, encoding_last shape: {encoding_last.shape}')
                np.save(os.path.join(self.spike_dir_path, f'{self.base_name}_encoding_first.npy'), encoding_first)
                np.save(os.path.join(self.spike_dir_path, f'{self.base_name}_encoding_last.npy'), encoding_last)

                raw_data = x_copy.permute(0, 2, 1).cpu().numpy() # (B, L, C)
                np.save(os.path.join(self.spike_dir_path, f'{self.base_name}_raw.npy'), raw_data)
        '''