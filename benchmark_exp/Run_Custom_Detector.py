# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

from TSB_AD.snn.spikingjelly.encoders import DynamicReceptiveEncoder
from TSB_AD.snn.params import running_params
import matplotlib.pyplot as plt

import torchinfo
import tqdm, math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
from TSB_AD.utils.torch_utility import EarlyStoppingTorch
from TSB_AD.utils.dataset import ForecastDataset

class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.mp = torch.nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)
    
class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.mp = torch.nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

class TargetModel(nn.Module):
    def __init__(self, HP, num_raw_features, num_enc_features):
        super(TargetModel, self).__init__()

        self.HP = HP

        self.encoder = DynamicReceptiveEncoder(
            local_running_params=HP,
            num_raw_features=num_raw_features,
            num_enc_features=HP['ParallelSNNModel']['num_enc_features'],
        )

        self.conv = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=(1,3),
                stride=(1,1),
                padding=(0,1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
        )

        self.fc = torch.nn.Sequential(
            AdaptiveConcatPool1d(),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=HP['ParallelSNNModel']['num_enc_features'] * 2,
                out_features=HP['ParallelSNNModel']['num_enc_features'],
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(
                in_features=HP['ParallelSNNModel']['num_enc_features'],
                out_features=HP['model']['predict_time_steps'],
            ),
        )

    def forward(self, x):
        enc = self.encoder(x)
        enc = torch.mean(enc, dim=1)
        print('enc shape: ', enc.shape)
        outputs = torch.zeros(self.HP['model']['predict_time_steps'], enc.shape[0], num_raw_features, device=enc.device)
        for t in range(self.HP['model']['predict_time_steps']):
            decoder_input = self.fc(enc)
            outputs[t] = torch.squeeze(decoder_input, dim=-2)
        return outputs

class Custom_AD(BaseDetector):

    def __init__(self, HP, normalize=True, num_raw_features=-1):
        super().__init__()
        self.HP = HP
        self.normalize = normalize

        assert num_raw_features > 0, "num_raw_features must be specified for Custom_AD"

        self.num_raw_features = num_raw_features
        self.device_type = self.HP['model']['device_type']
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.device_type == 'cuda' else 'cpu')
        self.batch_size = self.HP['model']['batch_size']
        self.epochs = self.HP['model']['max_epochs']
        self.validation_size = self.HP['model']['validation_size']
        self.window_size = self.HP['model']['window_size']
        self.predict_time_steps = self.HP['model']['predict_time_steps']

        self.model = TargetModel(
            HP=HP,
            num_raw_features=self.num_raw_features,
            num_enc_features=self.HP['ParallelSNNModel']['num_enc_features']
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Print model summary
        #print("Model Summary:")
        #print(torchinfo.summary(self.model, input_size=(1, self.window_size, self.num_raw_features), device=self.device_type))

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_samples, n_features = X.shape
        if self.normalize: X = zscore(X, axis=1, ddof=1)

        # dataloader
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(ForecastDataset(tsTrain, window_size=self.window_size, pred_len=self.predict_time_steps),
                                  batch_size=self.batch_size,
                                  shuffle=True)
        
        valid_loader = DataLoader(ForecastDataset(tsValid, window_size=self.window_size, pred_len=self.predict_time_steps),
                                  batch_size=self.batch_size,
                                  shuffle=False)

        # model preparation
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0008)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0.0001) #cosine annealing scheduler
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=30)

        self.mu = None
        self.sigma = None
        self.eps = 1e-10

        # training loop
        for epoch in range(1, self.epochs+1):
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

        #self.decision_scores_ = np.zeros(n_samples)
        #return self

    def decision_function(self, X):
        test_loader = DataLoader(
            ForecastDataset(X, window_size=self.window_size, pred_len=self.predict_time_steps),
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
                
                output = output.view(-1, self.num_raw_features*self.predict_time_steps)
                target = target.view(-1, self.num_raw_features*self.predict_time_steps)
                
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
        
        print('scores: ', scores.shape) 
        if scores.shape[0] < len(data):
            padded_decision_scores_ = np.zeros(len(data))
            padded_decision_scores_[: self.window_size+self.predict_time_steps-1] = scores[0]
            padded_decision_scores_[self.window_size+self.predict_time_steps-1 : ] = scores

        self.__anomaly_score = padded_decision_scores_
        return padded_decision_scores_
    
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        n_samples, n_features = X.shape
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu').unsqueeze(0)  # Add batch dimension
            enc = self.encoder(X_tensor)  # B, 3, E, L
            enc.squeeze_(0)  # Remove batch dimension
            enc = enc.mean(dim=0)
            enc = enc.sum(dim=0) # L

        enc = enc / enc.max() if enc.max() > 0 else enc  # Avoid division by zero
        #enc = softmax_enc = torch.nn.functional.softmax(enc, dim=0)

        print('normalized_enc: ', enc.shape)
        print(n_samples)

        #decision_scores_ = np.zeros(n_samples)
        decision_scores_ = enc.cpu().numpy()

        return decision_scores_

def run_Custom_AD_Unsupervised(data, HP):
    clf = Custom_AD(HP=HP)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_Custom_AD_Semisupervised(data_train, data_test, num_raw_features, **HP):
    clf = Custom_AD(HP=HP, num_raw_features=num_raw_features)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running Custom_AD')
    parser.add_argument('--filename', type=str, default='002_MSL_id_1_Sensor_tr_500_1st_900.csv')
    parser.add_argument('--data_direc', type=str, default='/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M/')
    parser.add_argument('--AD_Name', type=str, default='Custom_AD')
    parser.add_argument('--id_code', type=int, default=997, help='ID code for the experiment')
    args = parser.parse_args()

    #Custom_AD_HP = {
    #    'HP': ['HP'],
    #}
    Custom_AD_HP = running_params

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    num_raw_features = data.shape[1]

    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    start_time = time.time()

    output = run_Custom_AD_Semisupervised(data_train=data_train, data_test=data, num_raw_features=num_raw_features, **Custom_AD_HP)
    # output = run_Custom_AD_Unsupervised(data, **Custom_AD_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output)+3*np.std(output))

    id_code = args.id_code

    # plot the decision scores
    tokens = args.filename.split('.')[0].split('_')
    ts_name = tokens[0] + '_' + tokens[1]
    plt.figure(figsize=(10, 5))
    plt.plot(output, label='Decision Scores', color='blue')
    plt.plot(pred, label='Predictions', color='green', linestyle='--')
    plt.plot(label, label='Labels', color='red', alpha=0.5)
    plt.savefig(f'/home/hwkang/dev-TSB-AD/TSB-AD/figures/20250605/{id_code}_prediction_{ts_name}.png')

    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)

    record = list(evaluation_result.values())
    record.insert(0, run_time)
    record.insert(0, args.filename)

    col_w = ['file', 'Time'] + list(evaluation_result.keys())
    csv_path = f'/home/hwkang/dev-TSB-AD/TSB-AD/results/full/{id_code}_Custom_AD.csv'

    # append the record to the csv file
    row_df = pd.DataFrame([record], columns=col_w)
    if not os.path.exists(csv_path):
        row_df.to_csv(csv_path, index=False)
    else:
        row_df.to_csv(csv_path, mode='a', header=False, index=False)
        
