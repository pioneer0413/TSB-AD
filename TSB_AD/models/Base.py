import torchinfo
import numpy as np
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
import os
from torch.utils.data import DataLoader
from ..utils.dataset import ForecastDataset
import tqdm
import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

class BaseModule():
    def __init__(self,
                 TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                 batch_size=128, epochs=50, validation_size=0.2, # 훈련 루프 하이퍼파라미터
                 window_size=100, num_channel=[32, 40], lr=0.0008, # 학습 하이퍼파라미터
                 num_raw_features=1,
                 pred_len=1,
                 local_running_params=None): # 미지정 하이퍼파라미터

        self.local_running_params = local_running_params

        self.__anomaly_score = None
        self.y_hats = None

        # GPU/CPU 설정
        cuda = True
        self.cuda = cuda
        self.device = get_gpu(self.cuda) if local_running_params['off_cuda'] == False else torch.device('cpu')

        # 메타 데이터
        self.TS_Name = TS_Name
        self.AD_Name = AD_Name
        self.Encoder_Name = Encoder_Name

        if self.AD_Name == 'SpikeCNN':
            # 훈련 루프 하이퍼파라미터
            self.batch_size = self.local_running_params['batch_size']
            self.epochs = self.local_running_params['epochs']
            self.validation_size = self.local_running_params['validation_size']
            self.window_size = self.local_running_params['window_size'] # 학습 하이퍼파라미터
            self.num_raw_features = num_raw_features # Features 지정 하이퍼파라미터
            self.pred_len = self.local_running_params['predict_time_steps'] # 미지정 하이퍼파라미터
        else:
            self.batch_size = batch_size
            self.epochs = epochs
            self.validation_size = validation_size
            self.window_size = window_size
            self.num_channel = num_channel
            self.lr = lr
            self.num_raw_features = num_raw_features
            self.pred_len = pred_len

        # 적대적 공격
        self.adversarial = self.local_running_params['adversarial']
        
        self.root_dir_path = '/home/hwkang/dev-TSB-AD/TSB-AD/'
        self.save_path = os.path.join(self.root_dir_path, 'weights')
        os.makedirs(self.save_path, exist_ok=True)
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3, filename=local_running_params['save_file_path'])

    def fit(self, data):
        raise NotImplementedError("fit method not implemented")

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

                with torch.enable_grad():
                    if self.adversarial['type'] == 'fgsm':
                        x = fast_gradient_method(self.model, x, self.adversarial['fgsm']['eps'], self.adversarial['fgsm']['norm'])
                    elif self.adversarial['type'] == 'pgd':
                        x = projected_gradient_descent(self.model, x, self.adversarial['pgd']['eps'], self.adversarial['pgd']['eps_iter'], self.adversarial['pgd']['nb_iter'], self.adversarial['pgd']['norm'])

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

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hats
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))

    def clean_cuda(self):
        import torch, gc
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()