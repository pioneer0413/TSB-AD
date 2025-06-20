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
    def __init__(self, TS_Name, num_raw_features, local_running_params): # 미지정 하이퍼파라미터

        self.local_running_params = local_running_params

        self.__anomaly_score = None
        self.y_hats = None

        # 메타 데이터
        self.TS_Name = TS_Name
        self.AD_Name = self.local_running_params['meta']['AD_Name']
        self.Encoder_Name = self.local_running_params['meta']['Encoder_Name']
        self.postfix = self.local_running_params['meta']['postfix']
        self.id_code = self.local_running_params['meta']['id_code']

        # 모델 공통
        self.num_raw_features = num_raw_features
        self.device_type = self.local_running_params['model']['device_type']
        self.batch_size = self.local_running_params['model']['batch_size']
        self.epochs = self.local_running_params['model']['max_epochs']
        self.validation_size = self.local_running_params['model']['validation_size']
        self.window_size = self.local_running_params['model']['window_size']
        self.predict_time_steps = self.local_running_params['model']['predict_time_steps']

        # GPU/CPU 설정
        self.device = get_gpu(True) if self.device_type == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

        # 모델 저장 및 로드 경로 설정 (Not in use)      
        self.root_dir_path = '/home/hwkang/dev-TSB-AD/TSB-AD/'
        self.save_path = os.path.join(self.root_dir_path, 'weights')
        os.makedirs(self.save_path, exist_ok=True)

    def fit(self, data):
        raise NotImplementedError("fit method not implemented")

    def decision_function(self, data):
        test_loader = DataLoader(
            ForecastDataset(data, window_size=self.window_size, pred_len=self.predict_time_steps),
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

    def measure_time(self, data, min_run_time=100):
        import torch.utils.benchmark as benchmark

        # prepare dataloader
        test_loader = DataLoader(
            ForecastDataset(data, window_size=self.window_size, pred_len=self.predict_time_steps),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.eval()
        loop = list(test_loader)  # materialize dataloader once (torch.utils.benchmark requires it)

        def run_model():
            with torch.no_grad():
                for x, target in loop:
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    output = output.view(-1, self.num_raw_features * self.predict_time_steps)
                    target = target.view(-1, self.num_raw_features * self.predict_time_steps)
                    _ = torch.sub(output, target).pow(2)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        # warmup
        run_model()

        # benchmarking
        t = benchmark.Timer(
            stmt="run_model()",
            globals={"run_model": run_model},
            num_threads=1
        )

        result = t.blocked_autorange(min_run_time=min_run_time)
        return result