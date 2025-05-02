from typing import Dict
import torchinfo
import tqdm, math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os

from . import Base
from ..utils.torch_utility import EarlyStoppingTorch
from ..utils.dataset import ForecastDataset
from ..snn.encoders import PoissonEncoder, DeltaEncoder, ConvEncoder, RepeatEncoder
from ..snn.activations import SpikeActivation
from snntorch import utils

class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.mp = torch.nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

class SpikeCNNModel(nn.Module):
    def __init__(self,
                 device='cpu',
                 num_raw_features=-1,
                 predict_time_steps=1,
                 Encoder_Name=None,
                 local_running_params=None):
                 
        # initialize the super class
        super(SpikeCNNModel, self).__init__()

        self.device = device

        # save the default values
        self.num_channel = local_running_params['SpikeCNNModel']['num_channel']
        self.num_raw_features = num_raw_features # 원본 시계열 데이터의 채널 수
        self.num_enc_features = local_running_params['num_enc_features'] # 인코딩으로 생성하는 채널 수
        self.num_in_features = self.num_raw_features * self.num_enc_features # 예측 네트워크에 전달되는 입력 채널 수
        self.num_out_features = num_raw_features
        self.kernel_size = local_running_params['SpikeCNNModel']['kernel_size']
        self.stride = local_running_params['SpikeCNNModel']['stride']
        self.predict_time_steps = predict_time_steps
        self.dropout_rate = local_running_params['SpikeCNNModel']['dropout_rate']
        self.learn_threshold = local_running_params['SpikeCNNModel']['learn_threshold']

        # 인코더 지정
        self.Encoder_Name = Encoder_Name
        self.encoder = None
        if self.Encoder_Name == 'delta':
            self.encoder = DeltaEncoder(num_raw_features=self.num_raw_features, local_running_params=local_running_params)
        elif self.Encoder_Name == 'conv':
            self.encoder = ConvEncoder(num_raw_features=self.num_raw_features, local_running_params=local_running_params)
        elif self.Encoder_Name == 'repeat':
            self.encoder = RepeatEncoder(num_raw_features=self.num_raw_features, local_running_params=local_running_params)
        elif self.Encoder_Name == 'poisson':
            self.encoder = PoissonEncoder(num_enc_features=self.num_enc_features, local_running_params=local_running_params)
        else:
            raise ValueError(f"Encoder_Name: {self.Encoder_Name} is not supported.")

        # initialize encoder and decoder as a sequential
        self.conv_layers = nn.Sequential()
        prev_channels = self.num_in_features

        for idx, out_channels in enumerate(self.num_channel):
            self.conv_layers.add_module("conv" + str(idx),
                                        torch.nn.Conv1d(prev_channels,
                                                        self.num_channel[idx],
                                                        self.kernel_size,
                                                        self.stride))
            #self.conv_layers.add_module(self.hidden_activation + str(idx), self.activation)
            self.conv_layers.add_module("lif" + str(idx), SpikeActivation())
            self.conv_layers.add_module("pool" + str(idx), nn.MaxPool1d(kernel_size=2))
            prev_channels = out_channels

        self.fc = nn.Sequential(AdaptiveConcatPool1d(),
                                torch.nn.Flatten(),
                                torch.nn.Linear(2*self.num_channel[-1], self.num_channel[-1]),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(self.dropout_rate),
                                torch.nn.Linear(self.num_channel[-1], self.num_out_features))

    def forward(self, x, save_encoding=False):
        utils.reset(self.encoder)
        x = self.encoder(x) # (batch, num_enc_features, num_raw_features, length)

        if x.ndim == 4:
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(x.shape[0], -1, x.shape[3]) # [B, E*C, L] <- [B, E, C, L]
        
        # For each layer with a name starting with "lif", call utils.reset on its instance
        for name, module in self.conv_layers.named_children():
            if name.startswith("lif"):
                utils.reset(module)

        x = self.conv_layers(x) # (batch, num_channel[-1], length)
        
        outputs = torch.zeros(self.predict_time_steps, x.shape[0], self.num_out_features).to(self.device)
        for t in range(self.predict_time_steps):
            decoder_input = self.fc(x)
            outputs[t] = torch.squeeze(decoder_input, dim=-2)

        if save_encoding:
            return outputs, None
        else:
            return outputs # (steps, batch, num_out_features)
    
class SpikeCNN(Base.BaseModule):
    def __init__(self,
                 TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                 num_raw_features=-1,
                 local_running_params=None): # Model specific
                 
        super().__init__(TS_Name=TS_Name, AD_Name=AD_Name, Encoder_Name=Encoder_Name,
                         num_raw_features=num_raw_features,
                         local_running_params=local_running_params)

        self.optimizer_type = local_running_params['SpikeCNN']['optimizer']
        self.lr = local_running_params['SpikeCNN']['lr']
        self.scheduler_type = local_running_params['SpikeCNN']['scheduler']
        self.loss_type = local_running_params['SpikeCNN']['loss']
        self.mu = local_running_params['SpikeCNN']['mu']
        self.sigma = local_running_params['SpikeCNN']['sigma']
        self.eps = local_running_params['SpikeCNN']['eps']
        self.save_encoding = local_running_params['save_encoding']
        self.trace_threshold = local_running_params['trace_threshold']
        self.early_stop = local_running_params['early_stop']
    
        self.model = SpikeCNNModel(device=self.device,
                               num_raw_features=self.num_raw_features,
                               predict_time_steps=self.pred_len,
                               Encoder_Name=self.Encoder_Name,
                               local_running_params=local_running_params).to(self.device)
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            pass
        if self.scheduler_type == 'steplr':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        else:
            pass
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            pass

    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]
        
        # 학습/검증 데이터가 최소한 pred_len보다는 길어야 함
        if len(tsTrain) <= self.pred_len or len(tsValid) <= self.pred_len:
            raise ValueError(f"데이터 길이가 pred_len보다 커야 합니다. 학습 데이터 길이: {len(tsTrain)}, 검증 데이터 길이: {len(tsValid)}, pred_len: {self.pred_len}")

        # window_size 조정: 데이터셋이 최소한 하나의 샘플을 생성할 수 있도록 함
        max_possible_window_size = min(len(tsTrain) - self.pred_len, len(tsValid) - self.pred_len)
        self.window_size = min(self.window_size, max_possible_window_size)

        # window_size가 적어도 1 이상이어야 샘플 생성 가능
        if self.window_size < 1:
            raise ValueError(f"window_size가 너무 작아 샘플을 생성할 수 없습니다. 조정된 window_size: {self.window_size}")

        train_loader = DataLoader(ForecastDataset(tsTrain, window_size=self.window_size, pred_len=self.pred_len),
                                  batch_size=self.batch_size,
                                  shuffle=True)
        
        valid_loader = DataLoader(ForecastDataset(tsValid, window_size=self.window_size, pred_len=self.pred_len),
                                  batch_size=self.batch_size,
                                  shuffle=False)

        encoding_results = []
        threshold_traces = []
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                '''
                if idx == 0:
                    output, x_copy = self.model(x, save_encoding=self.save_encoding)
                    encoding_results.append(x_copy) # (B, E, C, L) * Epoch <-
                else:
                    output = self.model(x)
                '''
                
                output = self.model(x)
                '''
                if self.trace_threshold:
                    threshold = self.model.encoder.lif.threshold.detach().cpu()
                    threshold_traces.append(threshold)
                '''
                
                output = output.view(-1, self.num_raw_features*self.pred_len)
                target = target.view(-1, self.num_raw_features*self.pred_len)

                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()
                
                avg_loss += loss.cpu().item()

                if self.local_running_params['verbose']:
                    loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            self.model.eval()
            scores = []
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)

                    output = self.model(x)
                    
                    output = output.view(-1, self.num_raw_features*self.pred_len)
                    target = target.view(-1, self.num_raw_features*self.pred_len)
                    
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()

                    if self.local_running_params['verbose']:
                        loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                        loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
                    
            valid_loss = avg_loss/max(len(valid_loader), 1)
            self.scheduler.step()
            
            if self.local_running_params['early_stop']:
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop or epoch == self.epochs - 1:
                    # fitting Gaussian Distribution
                    if len(scores) > 0:
                        scores = torch.cat(scores, dim=0)
                        self.mu = torch.mean(scores)
                        self.sigma = torch.var(scores)
                        #print(self.mu.size(), self.sigma.size())
                    if self.early_stopping.early_stop:
                        #print("   Early stopping<<<")
                        pass
                    break

        '''
        if self.save_encoding:
            save_dir_path = '/home/hwkang/TSB-AD/TSB_AD/snn/tests/encoding'
            # encoding_results를 numpy로 변환
            x_copy = torch.stack(encoding_results, dim=0) # (Epochs, B, E, C, L)
            x_numpy = x_copy.detach().cpu().numpy()
            file_name = f'{self.Encoder_Name}_{self.encoding_threshold}_{self.learn_threshold}_{self.TS_Name}'
            save_file_path = os.path.join(save_dir_path, f'{file_name}.npy')
            np.save(save_file_path, x_numpy)

        if self.trace_threshold:
            save_dir_path = '/home/hwkang/TSB-AD/TSB_AD/snn/tests/threshold'
            x_copy = torch.stack(threshold_traces, dim=0) # (Epochs)
            x_numpy = x_copy.numpy()
            file_name = f'{self.Encoder_Name}_{self.encoding_threshold}_{self.learn_threshold}_{self.TS_Name}'
            save_file_path = os.path.join(save_dir_path, f'{file_name}.npy')
            np.save(save_file_path, x_numpy)
        '''
        
    def decision_function(self, data):
        test_loader = DataLoader(
            ForecastDataset(data, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        scores = []
        y_hats = []
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)

        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                
                output = self.model(x)
                
                output = output.view(-1, self.num_raw_features*self.pred_len)
                target = target.view(-1, self.num_raw_features*self.pred_len)

                mse = torch.sub(output, target).pow(2)

                y_hats.append(output.cpu())
                scores.append(mse.cpu())
                loop.set_description(f'Testing: ')

        scores = torch.cat(scores, dim=0)
        
        scores = scores.numpy()
        scores = np.mean(scores, axis=1)
        
        y_hats = torch.cat(y_hats, dim=0)
        y_hats = y_hats.numpy()
        
        l, w = y_hats.shape

        assert scores.ndim == 1
        
        if self.local_running_params['verbose']:
            print('scores: ', scores.shape) 
        if scores.shape[0] < len(data):
            padded_decision_scores_ = np.zeros(len(data))
            padded_decision_scores_[: self.window_size+self.pred_len-1] = scores[0]
            padded_decision_scores_[self.window_size+self.pred_len-1 : ] = scores

        self.__anomaly_score = padded_decision_scores_
        return padded_decision_scores_