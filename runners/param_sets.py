# Baseline on all TSB-AD-M

param_set9 = [
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'OCSVM', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'MCD', '--postfix', 'baseline', '--overwrite'],
]

param_set0 = [
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'IForest', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'LOF', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'PCA', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'HBOS', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'OCSVM', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'MCD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'KNN', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'KMeansAD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'COPOD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'CBLOF', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'EIF', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'RobustPCA', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'AutoEncoder', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'LSTMAD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'TranAD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'AnomalyTransformer', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'OmniAnomaly', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'USAD', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'Donut', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'TimesNet', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'FITS', '--postfix', 'baseline', '--overwrite'],
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'OFA', '--postfix', 'baseline', '--overwrite'],
]

# Baseline
param_set1 = [
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--AD_Name', 'CNN', '--postfix', 'baseline', '--overwrite'], # CNN-basline
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'baseline', '--overwrite'], # conv-baseline
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'baseline', '--overwrite'], # delta-basline
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'baseline', '--overwrite'], # repeat-baseline
]

# Scenario 1: Default L/SC/T to all encoder types
param_set2 = [
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'OXX', '--overwrite', '--learn_threshold'], # conv-OXX
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'OOX', '--overwrite', '--learn_threshold', '--second_chance'], # conv-OOX
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'OOO', '--overwrite', '--learn_threshold', '--second_chance', '--ternary'], # conv-OOO
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'XOX', '--overwrite', '--second_chance'], # conv-XOX
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'XOO', '--overwrite', '--second_chance', '--ternary'], # conv-XOO

    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'OXX', '--overwrite', '--learn_threshold'], # delta-OXX
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'OOX', '--overwrite', '--learn_threshold', '--second_chance'], # delta-OOX
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'OOO', '--overwrite', '--learn_threshold', '--second_chance', '--ternary'], # delta-OOO
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'XOX', '--overwrite', '--second_chance'], # delta-XOX
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'XOO', '--overwrite', '--second_chance', '--ternary'], # delta-XOO

    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'OXX', '--overwrite', '--learn_threshold'], # repeat-OXX
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'OOX', '--overwrite', '--learn_threshold', '--second_chance'], # repeat-OOX
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'OOO', '--overwrite', '--learn_threshold', '--second_chance', '--ternary'], # repeat-OOO
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'XOX', '--overwrite', '--second_chance'], # repeat-XOX
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'XOO', '--overwrite', '--second_chance', '--ternary'], # repeat-XOO
]

# Scenario 2: SCoF vs. Neuron-wise Learnable Threshold
# 2-1 Neuron-wise Learnable Threshold for conv
param_set3 = [
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'neuron_all1s', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'all-1s', '--early_stop_off'], # conv-neuron_all1s
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'neuron_all0s', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'all-0s', '--early_stop_off'], # conv-neuron_all1s
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'neuron_random', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random', '--early_stop_off'], # conv-neuron_all1s
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'neuron_he', '--overwrite', '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'he', '--early_stop_off'], # conv-neuron_all1s

    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'channel_all1s', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'all-1s', '--early_stop_off'], # conv-channel_all1s
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'channel_all0s', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'all-0s', '--early_stop_off'], # conv-channel_all1s
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'channel_random', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'random', '--early_stop_off'], # conv-channel_all1s
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'channel_he', '--overwrite', '--learn_threshold', '--granularity', 'channel', '--threshold_init', 'he', '--early_stop_off'], # conv-channel_all1s

    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'scalar', '--overwrite', 
     '--learn_threshold', '--threshold_init', 'scalar', '--early_stop_off'], # conv-scalar
]

# 2-2 Seconf Chance of Firing
param_set4 = [
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'l_l', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'linear', '--ternary', '--supra_threshold_type', 'linear', '--early_stop_off'], # conv-linear_linear
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'l_e', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'linear', '--ternary', '--supra_threshold_type', 'exponential', '--early_stop_off'], # conv-linear_exponential
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'l_g', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'linear', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off'], # conv-linear_gaussian

    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'e_l', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'linear', '--early_stop_off'], # conv-exponential_linear
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'e_e', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'exponential', '--early_stop_off'], # conv-exponential_exponential
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'e_g', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off'], # conv-exponential_gaussian

    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'g_l', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'gaussian', '--ternary', '--supra_threshold_type', 'linear', '--early_stop_off'], # conv-gaussian_linear
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'g_e', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'gaussian', '--ternary', '--supra_threshold_type', 'exponential', '--early_stop_off'], # conv-gaussian_exponential
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'g_g', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'gaussian', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off'], # conv-gaussian_gaussian

    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'linear', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'linear', '--early_stop_off'], # conv-linear
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'exponential', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'exponential', '--early_stop_off'], # conv-exponential
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'gaussian', '--overwrite', 
     '--second_chance', '--sub_threshold_type', 'gaussian', '--early_stop_off'], # conv-gaussian
]

# Scenario 3: SCoF + Learnable Threshold
param_set5 = [
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'conv', '--postfix', 'nlt_eg', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off',
        '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # conv-exponential_gaussian
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'delta', '--postfix', 'nlt_eg', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off',
        '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # delta-exponential_gaussian
    ['--file_list', '/home/hwkang/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv', '--save', '--Encoder_Name', 'repeat', '--postfix', 'nlt_eg', '--overwrite',
        '--second_chance', '--sub_threshold_type', 'exponential', '--ternary', '--supra_threshold_type', 'gaussian', '--early_stop_off',
        '--learn_threshold', '--granularity', 'neuron', '--threshold_init', 'random'], # repeat-exponential_gaussian
]