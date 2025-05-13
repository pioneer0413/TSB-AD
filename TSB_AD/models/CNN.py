from typing import Dict
import torchinfo
import tqdm, math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from . import Base
from ..utils.utility import get_activation_by_name
from ..utils.dataset import ForecastDataset
from ..snn.utils import loss_visualization
from ..snn.params import running_params

class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.mp = torch.nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

class CNNModel(nn.Module):
    def __init__(self,
                 device='cpu',
                 num_channel=[32, 40],
                 num_in_features=-1,
                 kernel_size=3,
                 stride=1,
                 predict_time_steps=1,
                 dropout_rate=0.25,
                 hidden_activation='relu',
                 ):

        # initialize the super class
        super(CNNModel, self).__init__()

        self.device = device

        # save the default values
        self.num_channel = num_channel
        self.num_in_features = num_in_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.predict_time_steps = predict_time_steps
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation

        # get the object for the activations functions
        self.activation = get_activation_by_name(hidden_activation)

        # initialize encoder and decoder as a sequential
        self.conv_layers = nn.Sequential()
        prev_channels = self.num_in_features

        for idx, out_channels in enumerate(self.num_channel):
            self.conv_layers.add_module(
                "conv" + str(idx),
                torch.nn.Conv1d(prev_channels, self.num_channel[idx], 
                self.kernel_size, self.stride))
            self.conv_layers.add_module(self.hidden_activation + str(idx),
                                    self.activation)
            self.conv_layers.add_module("pool" + str(idx), nn.MaxPool1d(kernel_size=2))
            prev_channels = out_channels

        self.fc = nn.Sequential(
            AdaptiveConcatPool1d(),
            torch.nn.Flatten(),
            torch.nn.Linear(2*self.num_channel[-1], self.num_channel[-1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.num_channel[-1], self.num_in_features)
        )

    def forward(self, x):
        b, l, c = x.shape
        x = x.view(b, c, l)
        x = self.conv_layers(x)     # [128, feature, 23]

        outputs = torch.zeros(self.predict_time_steps, b, self.num_in_features).to(self.device)
        for t in range(self.predict_time_steps):
            decoder_input = self.fc(x)
            outputs[t] = torch.squeeze(decoder_input, dim=-2)

        return outputs
    
class CNN(Base.BaseModule):
    def __init__(self,
                 TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                 batch_size=128, epochs=50, validation_size=0.2, # 훈련 루프 하이퍼파라미터
                 window_size=100, num_channel=[32, 40], lr=0.0008, # 학습 하이퍼파라미터
                 num_raw_features=1,
                 pred_len=1,
                 local_running_params=None): # 미지정 하이퍼파라미터
                 
        super().__init__(TS_Name=TS_Name, AD_Name=AD_Name, Encoder_Name=Encoder_Name,
                         batch_size=batch_size, epochs=epochs, validation_size=validation_size,
                         window_size=window_size, num_channel=num_channel, lr=lr,
                         num_raw_features=num_raw_features, pred_len=pred_len,
                         local_running_params=local_running_params)
  
        self.model = CNNModel(device=self.device,
                              num_channel=self.num_channel,
                              num_in_features=self.num_raw_features, 
                              predict_time_steps=self.pred_len).to(self.device)
        
        if self.local_running_params['load']:
            self.model.load_state_dict(torch.load(self.local_running_params['load_file_path']))
            print(f"Model loaded from {self.local_running_params['load_file_path']}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        
        self.mu = None
        self.sigma = None
        self.eps = 1e-10
        
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

        train_loader = DataLoader(
            ForecastDataset(tsTrain, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=True)
        
        valid_loader = DataLoader(
            ForecastDataset(tsValid, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False)
        
        train_loss_rec = []
        valid_loss_rec = []
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)

                # print('x: ', x.shape)       # (bs, win, feat)
                # print('target: ', target.shape)     # # (bs, pred_len, feat)
                # print('len(tsTrain): ', len(tsTrain))
                # print('len(train_loader): ', len(train_loader))

                self.optimizer.zero_grad()
                
                output = self.model(x)
                output = output.view(-1, self.num_raw_features*self.pred_len)
                target = target.view(-1, self.num_raw_features*self.pred_len)

                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()
                
                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))

                train_loss_rec.append(loss.cpu().item())
            
            
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
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
                    
                    valid_loss_rec.append(loss.cpu().item())
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            self.scheduler.step()
            
            try:
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop or epoch == self.epochs - 1:
                    #loss_visualization(train_loss_rec, valid_loss_rec, TS_Name=self.TS_Name, AD_Name='CNN', Encoder_Name=None)
                    # fitting Gaussian Distribution
                    if len(scores) > 0:
                        scores = torch.cat(scores, dim=0)
                        self.mu = torch.mean(scores)
                        self.sigma = torch.var(scores)
                        print(self.mu.size(), self.sigma.size())
                    if self.early_stopping.early_stop:
                        print("   Early stopping<<<")
                    break
            except Exception as e:
                print(f"Early stopping failed: {e}")
                break