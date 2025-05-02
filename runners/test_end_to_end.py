# Regular Test
''' 튜닝
python benchmark_exp/HP_Tuning_M.py \
    --dataset_dir Datasets/TSB-AD-M/ \
    --file_list Datasets/File_List/TSB-AD-M-Tuning.csv \
    --save_dir eval/tuning/multi/ \
    --AD_Name Proposal \
    --Encoder_Name conv
'''

''' 훈련&평가용 - CNN
python benchmark_exp/Run_Detector_M.py \
    --dataset_dir Datasets/TSB-AD-M/ \
    --file_list Datasets/File_List/TSB-AD-M-Tiny-Eva.csv \
    --AD_Name CNN \
    --score_dir eval/score/multi/ \
    --save_dir eval/metrics/multi/ \
    --time_energy_dir eval/time_energy/multi/ \
    --save True
'''

''' 훈련&평가용 - SpikeCNN
python benchmark_exp/Run_Detector_M.py \
    --dataset_dir Datasets/TSB-AD-M/ \
    --file_list Datasets/File_List/TSB-AD-M-Tiny-Eva.csv \
    --score_dir eval/score/multi/ \
    --save_dir eval/metrics/multi/ \
    --time_energy_dir eval/time_energy/multi \
    --save \
    --AD_Name SpikeCNN \
    --Encoder_Name conv
'''


# Quick Test
''' Quick Tuning
python benchmark_exp/HP_Tuning_M.py \
    --dataset_dir Datasets/TSB-AD-M/ \
    --file_list Datasets/File_List/TSB-AD-M-Quick-Tuning.csv \
    --save_dir eval/tuning/multi/ \
    --AD_Name Proposal \
    --Encoder_Name conv
'''

''' Quick Evaluation
python benchmark_exp/Run_Detector_M.py \
    --dataset_dir Datasets/TSB-AD-M/ \
    --file_list Datasets/File_List/TSB-AD-M-Quick-Eva.csv \
    --score_dir eval/score/multi/ \
    --save_dir eval/metrics/multi/ \
    --time_energy_dir eval/time_energy/multi/ \
    --save \
    --AD_Name Proposal \
    --Encoder_Name proposed
'''