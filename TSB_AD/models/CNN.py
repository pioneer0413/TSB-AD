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
from ..utils.utility import get_activation_by_name
from ..utils.torch_utility import EarlyStoppingTorch
from ..utils.dataset import ForecastDataset
import wandb

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
                 num_raw_features=-1,
                 kernel_size=3,
                 stride=1,
                 predict_time_steps=1,
                 dropout_rate=0.25,
                 hidden_activation='relu',
                 local_running_params=None,
                 ):

        # initialize the super class
        super(CNNModel, self).__init__()

        self.device = device
        self.local_running_params = local_running_params

        # save the default values
        self.num_channel = num_channel
        self.num_raw_features = num_raw_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.predict_time_steps = predict_time_steps
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation

        # get the object for the activations functions
        self.activation = get_activation_by_name(hidden_activation)

        # initialize encoder and decoder as a sequential
        self.conv_layers = nn.Sequential()
        prev_channels = self.num_raw_features

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
            torch.nn.Linear(self.num_channel[-1], self.num_raw_features)
        )

    def forward(self, x):
        b, l, c = x.shape
        x = x.view(b, c, l)

        x = self.conv_layers(x)     # [128, feature, 23]

        outputs = torch.zeros(self.predict_time_steps, b, self.num_raw_features).to(self.device)
        for t in range(self.predict_time_steps):
            decoder_input = self.fc(x)
            outputs[t] = torch.squeeze(decoder_input, dim=-2)

        return outputs
    
class CNN(Base.BaseModule):
    def __init__(self,
                 TS_Name=None,
                 num_raw_features=1,
                 local_running_params=None): # 미지정 하이퍼파라미터
        
        assert TS_Name is not None, "TS_Name must be provided"
        assert num_raw_features > 0, "num_raw_features must be greater than 0"
        assert local_running_params is not None, "local_running_params must be provided"

        super().__init__(TS_Name=TS_Name, num_raw_features=num_raw_features, local_running_params=local_running_params)
  
        self.model = CNNModel(device=self.device,
                              num_raw_features=self.num_raw_features,
                              local_running_params=local_running_params).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0008)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=30)
        
        self.mu = None
        self.sigma = None
        self.eps = 1e-10

        self.loss_dir_path = os.path.join('/home/hwkang/dev-TSB-AD/TSB-AD/results/loss', self.local_running_params['meta']['base_file_name'])
        os.makedirs(self.loss_dir_path, exist_ok=True)
        self.base_name = TS_Name.split('.')[0]

        self.wnb = self.local_running_params['analysis']['wandb']
        
    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            ForecastDataset(tsTrain, window_size=self.window_size, pred_len=self.predict_time_steps),
            batch_size=self.batch_size,
            shuffle=True)
        
        valid_loader = DataLoader(
            ForecastDataset(tsValid, window_size=self.window_size, pred_len=self.predict_time_steps),
            batch_size=self.batch_size,
            shuffle=False)
        
        # W&B
        if self.wnb:
            if wandb.run is not None:
                wandb.finish()

            copy_of_local_running_params = self.local_running_params.copy()
            copy_of_local_running_params.pop('data', None)  # Remove 'data' key if it exists
            copy_of_local_running_params.pop('meta', None)  # Remove 'meta' key if it exists
            local_model_params = copy_of_local_running_params
            tokens = self.TS_Name.split('.')[0].split('_')
            ts_name = tokens[0] + '_' + tokens[1]
            encoder_name = self.local_running_params['meta']['Encoder_Name']
            run = wandb.init(
                project=f'MTS-AD-DynamicReceptiveEncoder',
                group=f'CNN-{encoder_name}',
                name=f'{self.id_code}_{self.postfix}_{ts_name}_CNN',
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
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)

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
        
        train_loss_rec = np.array(train_loss_rec)
        valid_loss_rec = np.array(valid_loss_rec)
        np.save(os.path.join(self.loss_dir_path, f'{self.base_name}_train.npy'), train_loss_rec)
        np.save(os.path.join(self.loss_dir_path, f'{self.base_name}_valid.npy'), valid_loss_rec)