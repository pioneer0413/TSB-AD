import torchinfo
import numpy as np
from ..utils.torch_utility import get_gpu

class BaseModule():
    def __init__(self,
                 TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                 num_raw_features=-1,
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

        # 훈련 루프 하이퍼파라미터
        self.batch_size = self.local_running_params['batch_size']
        self.epochs = self.local_running_params['epochs']
        self.validation_size = self.local_running_params['validation_size']

        # 학습 하이퍼파라미터
        self.window_size = self.local_running_params['window_size']

        # Features 지정 하이퍼파라미터
        self.num_raw_features = num_raw_features

        #self.lr = self.local_running_params['SpikeCNN']['lr']

        # 미지정 하이퍼파라미터
        self.pred_len = self.local_running_params['predict_time_steps']

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