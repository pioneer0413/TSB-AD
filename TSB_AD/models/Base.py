import torchinfo
import numpy as np
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
import os

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
        self.device = get_gpu(self.cuda)

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

        self.root_dir_path = '/home/hwkang/dev-TSB-AD/TSB-AD/'
        self.save_path = os.path.join(self.root_dir_path, 'weights')
        os.makedirs(self.save_path, exist_ok=True)
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3, filename=f'{self.AD_Name}_{TS_Name}.pt')

    def fit(self, data):
        raise NotImplementedError("fit method not implemented")

    def decision_function(self, data):
        raise NotImplementedError("decision_function method not implemented")

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