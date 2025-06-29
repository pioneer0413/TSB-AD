import numpy as np
import math
from .utils.slidingWindows import find_length_rank
import logging

Unsupervise_AD_Pool = ['FFT', 'SR', 'NORMA', 'Series2Graph', 'Sub_IForest', 'IForest', 'LOF', 'Sub_LOF', 'POLY', 'MatrixProfile', 'Sub_PCA', 'PCA', 'HBOS', 
                        'Sub_HBOS', 'KNN', 'Sub_KNN','KMeansAD', 'KMeansAD_U', 'KShapeAD', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA', 'Lag_Llama', 'TimesFM', 'Chronos', 'MOMENT_ZS']
Semisupervise_AD_Pool = ['Left_STAMPi', 'SAND', 'MCD', 'Sub_MCD', 'OCSVM', 'Sub_OCSVM', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly', 
                        'AnomalyTransformer', 'TimesNet', 'FITS', 'Donut', 'OFA', 'MOMENT_FT', 'M2N2', 'ParallelSNN']

def run_Unsupervise_AD(model_name, data, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message

def run_Semisupervise_AD(data_train=None, data_test=None, # 데이터
                         TS_Name=None, local_running_params=None, # 하이퍼파라미터
                         **kwargs):

    try:
        AD_Name = local_running_params['meta']['AD_Name']
        function_name = f'run_{AD_Name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data_train, data_test, 
                                   TS_Name=TS_Name, local_running_params=local_running_params, **kwargs)
        return results
    except KeyError as e:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_call = tb[-1]
            error_message = f"Error in {last_call.filename} at line {last_call.lineno}: {str(e)}"
        else:
            error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message

def run_CNN(data_train, data_test, TS_Name=None, local_running_params=None, **kwargs): # 하이퍼파라미터
    from .models.CNN import CNN
    clf = CNN(TS_Name=TS_Name, num_raw_features=data_test.shape[1], local_running_params=local_running_params) 
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    clf.clean_cuda()
    return score.ravel()

def run_ParallelSNN(data_train, data_test, TS_Name=None, local_running_params=None, **kwargs): # 하이퍼파라미터
    from .models.ParallelSNN import ParallelSNN
    clf = ParallelSNN(TS_Name=TS_Name, num_raw_features=data_test.shape[1], local_running_params=local_running_params)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    clf.clean_cuda()
    return score.ravel()

def run_FFT(data, ifft_parameters=5, local_neighbor_window=21, local_outlier_threshold=0.6, max_region_size=50, max_sign_change_distance=10):
    from .models.FFT import FFT
    clf = FFT(ifft_parameters=ifft_parameters, local_neighbor_window=local_neighbor_window, local_outlier_threshold=local_outlier_threshold, max_region_size=max_region_size, max_sign_change_distance=max_sign_change_distance)
    clf.fit(data)  
    score = clf.decision_scores_ 
    return score.ravel()

def run_Sub_IForest(data, periodicity=1, n_estimators=100, max_features=1, n_jobs=1):
    from .models.IForest import IForest
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_IForest(data, slidingWindow=100, n_estimators=100, max_features=1, n_jobs=1):
    from .models.IForest import IForest
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_LOF(data, periodicity=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from .models.LOF import LOF
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_LOF(data, slidingWindow=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from .models.LOF import LOF
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_POLY(data, periodicity=1, power=3, n_jobs=1):
    from .models.POLY import POLY
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = POLY(power=power, window = slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MatrixProfile(data, periodicity=1, n_jobs=1):
    from .models.MatrixProfile import MatrixProfile
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = MatrixProfile(window=slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Left_STAMPi(data_train, data, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None, # 하이퍼파라미터
                   ):
    from .models.Left_STAMPi import Left_STAMPi
    clf = Left_STAMPi(n_init_train=len(data_train), window_size=100)
    clf.fit(data)
    score = clf.decision_function(data)
    return score.ravel()

def run_SAND(data_train, data_test, periodicity=1, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None, # 하이퍼파라미터
                   ):
    from .models.SAND import SAND
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = SAND(pattern_length=slidingWindow, subsequence_length=4*(slidingWindow))
    clf.fit(data_test.squeeze(), online=True, overlaping_rate=int(1.5*slidingWindow), init_length=len(data_train), alpha=0.5, batch_size=max(5*(slidingWindow), int(0.1*len(data_test))))
    score = clf.decision_scores_
    return score.ravel()

def run_KShapeAD(data, periodicity=1):
    from .models.SAND import SAND
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = SAND(pattern_length=slidingWindow, subsequence_length=4*(slidingWindow))
    clf.fit(data.squeeze(), overlaping_rate=int(1.5*slidingWindow))
    score = clf.decision_scores_
    return score.ravel()

def run_Series2Graph(data, periodicity=1):
    from .models.Series2Graph import Series2Graph
    slidingWindow = find_length_rank(data, rank=periodicity)

    data = data.squeeze()
    s2g = Series2Graph(pattern_length=slidingWindow)
    s2g.fit(data)
    query_length = 2*slidingWindow
    s2g.score(query_length=query_length,dataset=data)

    score = s2g.decision_scores_
    score = np.array([score[0]]*math.ceil(query_length//2) + list(score) + [score[-1]]*(query_length//2))
    return score.ravel()

def run_Sub_PCA(data, periodicity=1, n_components=None, n_jobs=1):
    from .models.PCA import PCA
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_PCA(data, slidingWindow=100, n_components=None, n_jobs=1):
    from .models.PCA import PCA
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_NORMA(data, periodicity=1, clustering='hierarchical', n_jobs=1):
    from .models.NormA import NORMA
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = NORMA(pattern_length=slidingWindow, nm_size=3*slidingWindow, clustering=clustering)
    clf.fit(data)
    score = clf.decision_scores_
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    if len(score) > len(data):
        start = len(score) - len(data)
        score = score[start:]
    return score.ravel()

def run_Sub_HBOS(data, periodicity=1, n_bins=10, tol=0.5, n_jobs=1):
    from .models.HBOS import HBOS
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_HBOS(data, slidingWindow=1, n_bins=10, tol=0.5, n_jobs=1):
    from .models.HBOS import HBOS
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, periodicity=1, n_jobs=1, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.OCSVM import OCSVM
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, slidingWindow=1, n_jobs=1, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.OCSVM import OCSVM
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Sub_MCD(data_train, data_test, support_fraction=None, periodicity=1, n_jobs=1, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.MCD import MCD
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_MCD(data_train, data_test, support_fraction=None, slidingWindow=1, n_jobs=1, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터
    from .models.MCD import MCD
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Sub_KNN(data, n_neighbors=10, method='largest', periodicity=1, n_jobs=1):
    from .models.KNN import KNN
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors,method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KNN(data, slidingWindow=1, n_neighbors=10, method='largest', n_jobs=1):
    from .models.KNN import KNN
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors, method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KMeansAD(data, n_clusters=20, window_size=20, n_jobs=1):
    from .models.KMeansAD import KMeansAD
    clf = KMeansAD(k=n_clusters, window_size=window_size, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()

def run_KMeansAD_U(data, n_clusters=20, periodicity=1,n_jobs=1):
    from .models.KMeansAD import KMeansAD
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KMeansAD(k=n_clusters, window_size=slidingWindow, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()

def run_COPOD(data, n_jobs=1):
    from .models.COPOD import COPOD
    clf = COPOD(n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_CBLOF(data, n_clusters=8, alpha=0.9, n_jobs=1):
    from .models.CBLOF import CBLOF
    clf = CBLOF(n_clusters=n_clusters, alpha=alpha, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_COF(data, n_neighbors=30):
    from .models.COF import COF
    clf = COF(n_neighbors=n_neighbors)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_EIF(data, n_trees=100):
    from .models.EIF import EIF
    clf = EIF(n_trees=n_trees)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_RobustPCA(data, max_iter=1000):
    from .models.RobustPCA import RobustPCA
    clf = RobustPCA(max_iter=max_iter)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_SR(data, periodicity=1):
    from .models.SR import SR
    slidingWindow = find_length_rank(data, rank=periodicity)
    return SR(data, window_size=slidingWindow)

def run_AutoEncoder(data_train, data_test, window_size=100, hidden_neurons=[64, 32], n_jobs=1, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.AE import AutoEncoder
    clf = AutoEncoder(slidingWindow=window_size, hidden_neurons=hidden_neurons, batch_size=128, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_LSTMAD(data_train, data_test, window_size=100, lr=0.0008, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.LSTMAD import LSTMAD
    clf = LSTMAD(window_size=window_size, pred_len=1, lr=lr, feats=data_test.shape[1], batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_TranAD(data_train, data_test, win_size=10, lr=1e-3, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.TranAD import TranAD
    clf = TranAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_AnomalyTransformer(data_train, data_test, win_size=100, lr=1e-4, batch_size=128, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.AnomalyTransformer import AnomalyTransformer
    clf = AnomalyTransformer(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OmniAnomaly(data_train, data_test, win_size=100, lr=0.002, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.OmniAnomaly import OmniAnomaly
    clf = OmniAnomaly(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_USAD(data_train, data_test, win_size=5, lr=1e-4, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.USAD import USAD
    clf = USAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Donut(data_train, data_test, win_size=120, lr=1e-4, batch_size=128, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.Donut import Donut
    clf = Donut(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_TimesNet(data_train, data_test, win_size=96, lr=1e-4, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.TimesNet import TimesNet
    clf = TimesNet(win_size=win_size, enc_in=data_test.shape[1], lr=lr, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_FITS(data_train, data_test, win_size=100, lr=1e-3, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.FITS import FITS
    clf = FITS(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OFA(data_train, data_test, win_size=100, batch_size = 64, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.OFA import OFA
    clf = OFA(win_size=win_size, enc_in=data_test.shape[1], epochs=10, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Lag_Llama(data, win_size=96, batch_size=64):
    from .models.Lag_Llama import Lag_Llama
    clf = Lag_Llama(win_size=win_size, input_c=data.shape[1], batch_size=batch_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Chronos(data, win_size=50, batch_size=64):
    from .models.Chronos import Chronos
    clf = Chronos(win_size=win_size, prediction_length=1, input_c=data.shape[1], model_size='base', batch_size=batch_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_TimesFM(data, win_size=96):
    from .models.TimesFM import TimesFM
    clf = TimesFM(win_size=win_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MOMENT_ZS(data, win_size=256):
    from .models.MOMENT import MOMENT
    clf = MOMENT(win_size=win_size, input_c=data.shape[1])

    # Zero shot
    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MOMENT_FT(data_train, data_test, win_size=256, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.MOMENT import MOMENT
    clf = MOMENT(win_size=win_size, input_c=data_test.shape[1])

    # Finetune
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_M2N2(data_train, data_test, epochs=10, win_size=12, lr=1e-3, batch_size=128, TS_Name=None, AD_Name=None, Encoder_Name=None, # 메타 데이터
                   local_running_params=None): # 하이퍼파라미터):
    from .models.M2N2 import M2N2
    clf = M2N2(win_size=win_size, num_channels=data_test.shape[1], lr=lr, batch_size=batch_size, epochs=epochs)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()