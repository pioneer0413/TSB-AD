####
# Test
####
param_set_test = [
    # CNN
    #['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Tiny-Eva.csv', '--save', '--AD_Name', 'CNN', '--postfix', 'test'],
    # Baseline(no learn, no ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Tiny-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'test'],
]

####
# S1: Baselines
####
# AB: All Baselines
param_set_AB = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'OCSVM', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'MCD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'KNN', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'KMeansAD', '--postfix', 'baseline', '--overwrite'],
]
# Baseline: all TSB-AD-M algorithms
param_set_1_1 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'IForest', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'LOF', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'PCA', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'HBOS', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'OCSVM', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'MCD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'KNN', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'KMeansAD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'COPOD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'CBLOF', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'EIF', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'RobustPCA', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'AutoEncoder', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'LSTMAD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'TranAD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'AnomalyTransformer', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'OmniAnomaly', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'USAD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'Donut', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'TimesNet', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'FITS', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'OFA', '--postfix', 'baseline', '--overwrite'],
]
# Baseline: CNN, conv, delta, repeat
param_set_1_2 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'CNN', '--postfix', 'baseline', '--overwrite'], # CNN-basline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'baseline', '--overwrite'], # conv-baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'delta', '--postfix', 'baseline', '--overwrite'], # delta-basline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'repeat', '--postfix', 'baseline', '--overwrite'], # repeat-baseline
]
param_set_1_2_1 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'baseline', '--overwrite'], # conv-baseline
]
# Baseline: all cases of N-LT and SCoF
param_set_1_3 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'OXX', '--overwrite', '--learn_threshold'], # conv-OXX
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'OOX', '--overwrite', '--learn_threshold', '--second_chance'], # conv-OOX
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'OOO', '--overwrite', '--learn_threshold', '--second_chance', '--ternary'], # conv-OOO
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'XOX', '--overwrite', '--second_chance'], # conv-XOX
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'XOO', '--overwrite', '--second_chance', '--ternary'], # conv-XOO

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'OXX', '--overwrite', '--learn_threshold'], # delta-OXX
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'OOX', '--overwrite', '--learn_threshold', '--second_chance'], # delta-OOX
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'OOO', '--overwrite', '--learn_threshold', '--second_chance', '--ternary'], # delta-OOO
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'XOX', '--overwrite', '--second_chance'], # delta-XOX
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'XOO', '--overwrite', '--second_chance', '--ternary'], # delta-XOO

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'OXX', '--overwrite', '--learn_threshold'], # repeat-OXX
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'OOX', '--overwrite', '--learn_threshold', '--second_chance'], # repeat-OOX
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'OOO', '--overwrite', '--learn_threshold', '--second_chance', '--ternary'], # repeat-OOO
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'XOX', '--overwrite', '--second_chance'], # repeat-XOX
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'XOO', '--overwrite', '--second_chance', '--ternary'], # repeat-XOO
]
# Baseline: CNN, conv
param_set_1_4 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'CNN', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'baseline', '--overwrite'],
]

####
# S2: SCoF vs. Neuron-wise Learnable Threshold
####
# 2-1: Neuron-wise Learnable Threshold for conv
param_set_2_1 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_all1s', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'all-1s'], # conv-neuron_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_all0s', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'all-0s'], # conv-neuron_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_random', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # conv-neuron_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_he', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'he'], # conv-neuron_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_all1s', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'all-1s'], # conv-channel_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_all0s', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'all-0s'], # conv-channel_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_random', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'random'], # conv-channel_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_he', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'he'], # conv-channel_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'scalar', '--overwrite', 
     '--learn_threshold', '--threshold_init', 'scalar'], # conv-scalar
]
# 2-2: Second Chance of Firing for conv
param_set_2_2 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'l_l', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'linear', '--ternary', '--supra_threshold_type', 'linear', '--early_stop_off'], # conv-linear_linear
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'l_e', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'linear', '--ternary', '--supra_threshold_type', 'exponential', '--early_stop_off'], # conv-linear_exponential
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'l_g', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'linear', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off'], # conv-linear_gaussian

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'e_l', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'linear', '--early_stop_off'], # conv-exponential_linear
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'e_e', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'exponential', '--early_stop_off'], # conv-exponential_exponential
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'e_g', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off'], # conv-exponential_gaussian

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'g_l', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'gaussian', '--ternary', '--supra_threshold_type', 'linear', '--early_stop_off'], # conv-gaussian_linear
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'g_e', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'gaussian', '--ternary', '--supra_threshold_type', 'exponential', '--early_stop_off'], # conv-gaussian_exponential
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'g_g', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'gaussian', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off'], # conv-gaussian_gaussian

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'linear', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'linear', '--early_stop_off'], # conv-linear
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'exponential', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'exponential', '--early_stop_off'], # conv-exponential
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'gaussian', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'gaussian', '--early_stop_off'], # conv-gaussian
]
# 2-3: Second Chance of Firing for conv (exponential-gaussian)
param_set_2_3 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'e_g', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # conv-exponential_gaussian
]

####
# S3: SCoF + Neuron-wise Learnable Threshold
####
param_set_3_1 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'nlt_eg', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off',
        '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # conv-exponential_gaussian
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'nlt_eg', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off',
        '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # delta-exponential_gaussian
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'nlt_eg', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off',
        '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # repeat-exponential_gaussian
]

####
# S4: SCoFs and LTs to repeat and delta (running on dgx1)
####

####
# S5: Adversarial Attack
####
# Scenario 5-1: Adversarial Attack pt.1
param_set_5_1 = [
    # CNN
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'CNN', '--postfix', 'adv_base', '--overwrite'], # CNN-adversarial
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'CNN', '--postfix', 'adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm'], # CNN-adversarial
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'CNN', '--postfix', 'adv_pgd', '--overwrite', '--adversarial_type', 'pgd'], # CNN-adversarial

    # conv
    # conv-adv_base
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'adv_base', '--overwrite'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm'], # fgsm
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'adv_pgd', '--overwrite', '--adversarial_type', 'pgd'], # pgd

    # conv-adv_N-LT
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_adv_base', '--overwrite', 
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm',
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # fgsm
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_adv_pgd', '--overwrite', '--adversarial_type', 'pgd',
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # pgd
    
    # conv-adv_SCoF
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scof_adv_base', '--overwrite',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scof_adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # fgsm
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scof_adv_pgd', '--overwrite', '--adversarial_type', 'pgd',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # pgd
]
# Scenario 5-2: Adversarial Attack pt.2
param_set_5_2 = [
    # delta
     # delta-adv_base
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'delta', '--postfix', 'adv_base', '--overwrite'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'delta', '--postfix', 'adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm'], # fgsm
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'delta', '--postfix', 'adv_pgd', '--overwrite', '--adversarial_type', 'pgd'], # pgd

    # delta-adv_N-LT
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'delta', '--postfix', 'nlt_adv_base', '--overwrite', 
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'delta', '--postfix', 'nlt_adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm',
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # fgsm
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'delta', '--postfix', 'nlt_adv_pgd', '--overwrite', '--adversarial_type', 'pgd',
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # pgd
    
    # delta-adv_SCoF
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'delta', '--postfix', 'scof_adv_base', '--overwrite',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'delta', '--postfix', 'scof_adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # fgsm
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'delta', '--postfix', 'scof_adv_pgd', '--overwrite', '--adversarial_type', 'pgd',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # pgd

    # repeat
    # repeat-adv_base
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'repeat', '--postfix', 'adv_base', '--overwrite'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'repeat', '--postfix', 'adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm'], # fgsm
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'repeat', '--postfix', 'adv_pgd', '--overwrite', '--adversarial_type', 'pgd'], # pgd

    # repeat-adv_N-LT
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'repeat', '--postfix', 'nlt_adv_base', '--overwrite', 
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'repeat', '--postfix', 'nlt_adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm',
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # fgsm
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'repeat', '--postfix', 'nlt_adv_pgd', '--overwrite', '--adversarial_type', 'pgd',
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # pgd
    
    # repeat-adv_SCoF
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'repeat', '--postfix', 'scof_adv_base', '--overwrite',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'repeat', '--postfix', 'scof_adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # fgsm
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'repeat', '--postfix', 'scof_adv_pgd', '--overwrite', '--adversarial_type', 'pgd',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # pgd
]
# Scenario 5-3: Single layer, conv[baseline, N-LT, SCoF, deterministic, rate] baseline, FGSM
param_set_5_3 = [
    
    # conv-baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'adv_base', '--overwrite'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm'], # fgsm
    
    # conv-N-LT
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_adv_base', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # fgsm
    
    # conv-SCoF
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scof_adv_base', '--overwrite',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scof_adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # fgsm
    
    # conv-deterministic
    # only ternary
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'det_ternary_adv_base', '--overwrite'],
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'det_ternary_adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm'], # fgsm
    # nlt ternary
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_ternary_adv_base', '--overwrite', 
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_ternary_adv_fgsm', '--overwrite', 
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random', '--adversarial_type', 'fgsm'], # fgsm

    # conv-rate
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'rate_adv_base', '--overwrite', '--rate_encode'], # baseline
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'rate_adv_fgsm', '--overwrite', '--rate_encode', '--adversarial_type', 'fgsm'] # fgsm
]
# Scenario 5-4: Single layer, conv[baseline, N-LT, SCoF, deterministic, rate] PGD
param_set_5_4 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'adv_pgd', '--overwrite', '--adversarial_type', 'pgd'], # pgd

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_adv_pgd', '--overwrite', '--adversarial_type', 'pgd', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # pgd

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scof_adv_pgd', '--overwrite', '--adversarial_type', 'pgd',
     '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian'], # pgd

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'det_ternary_adv_pgd', '--overwrite', '--adversarial_type', 'pgd'], # pgd

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_ternary_adv_pgd', '--overwrite', 
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random', '--adversarial_type', 'pgd'], # pgd

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'rate_adv_pgd', '--overwrite', '--rate_encode', '--adversarial_type', 'pgd'] # pgd
]
# Scenario 5-5: All layers, conv[baseline, N-LT, SCoF, deterministic, rate] baseline, FGSM
param_set_5_5 = [
    
    # SCoF
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv',
     '--postfix', 'scof_all_adv_base', '--overwrite', 
     '--second_chance_all', '--second_chance', '--ternary'], # baseline
    
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv',
     '--postfix', 'scof_all_adv_fgsm', '--overwrite', '--adversarial_type', 'fgsm',
     '--second_chance_all', '--second_chance', '--ternary'], # fgsm

]
# Scenario 5-6: All layers, conv[baseline, N-LT, SCoF, deterministic, rate] PGD
param_set_5_6 = [

    # SCoF
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv',
     '--postfix', 'scof_all_adv_pgd', '--overwrite', '--adversarial_type', 'pgd',
     '--second_chance_all', '--second_chance', '--ternary'], # pgd
]

####
# S6: Neuron-wise Learnable Threshold + Ternary Spikes
####
# 
param_set_6_1 = [
    # N-LT: False; Only Ternary Spikes
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_ternary', '--overwrite'],
    # N-LT: True; N-LT + Ternary Spikes
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_ternary', '--overwrite', 
    '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # baseline
]

####
# S7: Neuron-wise Learnable Threshold to entire layer
####

####
# S8: Adversarial Attack on ParaLIF-SpikeCNN
####

####
# S9: Difference from Rate Coding with timestep=2
#### 
param_set_9_1 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'rate', '--overwrite', '--rate_encode']
]

####
# S10: First layer only vs. All layers
####
# 10-1: First layer only
param_set_10_1 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_all1s', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'all-1s'], # conv-neuron_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_all0s', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'all-0s'], # conv-neuron_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_random', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # conv-neuron_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_he', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'he'], # conv-neuron_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_all1s', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'all-1s'], # conv-channel_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_all0s', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'all-0s'], # conv-channel_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_random', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'random'], # conv-channel_all1s
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_he', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'he'], # conv-channel_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scalar', '--overwrite', 
     '--learn_threshold', '--threshold_init', 'scalar'], # conv-scalar
]
# 10-2: All layers
param_set_10_2 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_all1s', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'all-1s',
     '--learn_threshold_all', '--granularity_all', 'neuron', '--threshold_init_all', 'all-1s'], # conv-neuron_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_all0s', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'all-0s',
    '--learn_threshold_all', '--granularity_all', 'neuron', '--threshold_init_all', 'all-0s'], # conv-neuron_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_random', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random',
     '--learn_threshold_all', '--granularity_all', 'neuron', '--threshold_init_all', 'random'], # conv-neuron_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_he', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'he',
     '--learn_threshold_all', '--granularity_all', 'neuron', '--threshold_init_all', 'he'], # conv-neuron_all1s


    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_all1s', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'all-1s',
     '--learn_threshold_all', '--granularity_all', 'channel', '--threshold_init_all', 'all-1s'], # conv-channel_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_all0s', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'all-0s',
     '--learn_threshold_all', '--granularity_all', 'channel', '--threshold_init_all', 'all-1s'], # conv-channel_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_random', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'random',
     '--learn_threshold_all', '--granularity_all', 'channel', '--threshold_init_all', 'all-1s'], # conv-channel_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_he', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'he',
     '--learn_threshold_all', '--granularity_all', 'channel', '--threshold_init_all', 'all-1s'], # conv-channel_all1s


    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'scalar', '--overwrite', 
     '--learn_threshold', '--threshold_init', 'scalar',
     '--learn_threshold_all', '--threshold_init_all', 'scalar'], # conv-scalar
]
# 10-3: First layer only + Ternary
param_set_10_3 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_all1s', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'all-1s', '--deterministic'], # conv-neuron_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_all0s', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'all-0s', '--deterministic'], # conv-neuron_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_random', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random', '--deterministic'], # conv-neuron_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'neuron_he', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'he', '--deterministic'], # conv-neuron_all1s


    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_all1s', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'all-1s', '--deterministic'], # conv-channel_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_all0s', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'all-0s', '--deterministic'], # conv-channel_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_random', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'random', '--deterministic'], # conv-channel_all1s

    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'channel_he', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'he', '--deterministic'], # conv-channel_all1s


    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scalar', '--overwrite', 
     '--learn_threshold', '--threshold_init', 'scalar', '--deterministic'], # conv-scalar
]
# 10-4: All layers + SCoF
param_set_10_4 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv',
     '--postfix', 'scof_all', '--overwrite',
     '--second_chance_all', '--second_chance', '--ternary']
]

####
# S12: Leaderboard Top-3
####
param_set_11_MINI =[
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scalar_ternaryF_learnT', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'scalar'], # scalar
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scof_all', '--overwrite', '--second_chance_all', '--second_chance', '--ternary'], # scof_all
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_random', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # N-LT_random
]
param_set_11_FULL =[
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scalar_ternaryF_learnT', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'scalar'], # scalar
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'scof_all', '--overwrite', '--second_chance_all', '--second_chance', '--ternary'], # scof_all
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'nlt_random', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # N-LT_random
]

####
# S13: BNTT
####
param_set_13_1 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'bntt_only', '--overwrite', '--bntt'], # BNTT Only
    
]
param_set_13_2 = [
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'bntt_only', '--overwrite', '--bntt', '--second_chance_all', '--second_chance'], # BNTT + SCoF(All)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 'bntt_only', '--overwrite', '--bntt', '--learn_threshold_all', '--granularity_all', 'neuron', '--threshold_init_all', 'random',
     '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # BNTT + N-LT(Random)(All)
]

####
# S14: Information Loss
####
param_set_14_1 = [
    # Baseline(no learn, no ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Tiny-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_baseline', '--overwrite', '--save_encoding', '--early_stop_off'], # baseline

    # SCoF(no ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Tiny-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_scof_no_ternary', '--overwrite',
     '--second_chance', '--save_encoding', '--early_stop_off'], # SCoF(no ternary)

    # SCoF(ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Tiny-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_scof_ternary', '--overwrite',
     '--second_chance', '--ternary', '--save_encoding', '--early_stop_off'], # SCoF(ternary)

    # N-LT(no ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Tiny-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_nlt_no_ternary', '--overwrite',
     '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random', '--save_encoding', '--early_stop_off'], # N-LT(no ternary)

    # N-LT(ternary-Det)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Tiny-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_nlt_ternary_det', '--overwrite',
     '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random', '--deterministic', '--save_encoding', '--early_stop_off'], # N-LT(ternary-Det)

    # Scalar(no ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Tiny-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_scalar_no_ternary', '--overwrite',
     '--learn_threshold', '--threshold_init', 'scalar', '--save_encoding', '--early_stop_off'], # Scalar(no ternary)

    # Scalar(ternary-Det)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Tiny-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_scalar_ternary_det', '--overwrite',
     '--learn_threshold', '--threshold_init', 'scalar', '--deterministic', '--save_encoding', '--early_stop_off'], # Scalar(ternary-Det)

    # Rate(ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Tiny-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_rate_ternary', '--overwrite',
     '--rate_encode', '--save_encoding', '--early_stop_off'], # Rate(ternary)
]
param_set_14_2 = [ # Acc. performance
    # Baseline(no learn, no ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_baseline_acc', '--overwrite',], # baseline

    # SCoF(no ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_scof_no_ternary_acc', '--overwrite',
     '--second_chance', ], # SCoF(no ternary)

    # SCoF(ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_scof_ternary_acc', '--overwrite',
     '--second_chance', '--ternary', ], # SCoF(ternary)

    # N-LT(no ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_nlt_no_ternary_acc', '--overwrite',
     '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random', ], # N-LT(no ternary)

    # N-LT(ternary-Det)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_nlt_ternary_det_acc', '--overwrite',
     '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random', '--deterministic', ], # N-LT(ternary-Det)

    # Scalar(no ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_scalar_no_ternary_acc', '--overwrite',
     '--learn_threshold', '--threshold_init', 'scalar', ], # Scalar(no ternary)

    # Scalar(ternary-Det)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_scalar_ternary_det_acc', '--overwrite',
     '--learn_threshold', '--threshold_init', 'scalar', '--deterministic', ], # Scalar(ternary-Det)

    # Rate(ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's14_rate_ternary_acc', '--overwrite',
     '--rate_encode', ], # Rate(ternary)
]

####
# S15: Running on CPU
####
param_set_15 = [
    # PCA
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'PCA', '--postfix', 's15_baseline'],
    # CNN
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'CNN', '--postfix', 's15_baseline', '--off_cuda'],
    # Baseline(no learn, no ternary)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's15_baseline', '--off_cuda'],
]

####
# S16: TernarySpikeActivation with off_cuda
####
param_set_16_1 = [
    # TernarySpikeActivation (CPU)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's16_ternary_runtime_cpu', '--off_cuda', '--activation_type', 'ternary'],
    # TernarySpikeActivation (GPU)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's16_ternary_runtime_gpu', '--activation_type', 'ternary'],
]
param_set_16_2 = [
    # CNN (GPU)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'CNN', '--postfix', 's15_baseline_gpu'],
    # Baseline(no learn, no ternary) (GPU)
    ['--file_list', '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Mini-Eva.csv', '--save', '--AD_Name', 'SpikeCNN', '--Encoder_Name', 'conv', '--postfix', 's16_baseline_gpu', '--activation_type', 'binary'],
]