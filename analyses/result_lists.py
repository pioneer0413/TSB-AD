

####
# S5: Adversarial Attack on SpikeCNN
####
file_paths = [
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/CNN_004_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/CNN_003_adv_fgsm.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/CNN_002_adv_pgd.csv',

    # conv-baseline
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_001_conv_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_005_conv_adv_fgsm.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_007_conv_adv_pgd.csv',
    # conv-N-LT
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_006_conv_nlt_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_000_conv_nlt_adv_fgsm.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_016_conv_nlt_adv_pgd.csv',
    # conv-SCoF
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_017_conv_scof_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_018_conv_scof_adv_fgsm.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_020_conv_scof_adv_pgd.csv',

    # delta-baseline
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_013_delta_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_008_delta_adv_fgsm.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_010_delta_adv_pgd.csv',
    # delta-N-LT
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_014_delta_nlt_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_009_delta_nlt_adv_fgsm.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_015_delta_nlt_adv_pgd.csv',
    # delta-SCoF
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_011_delta_scof_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_012_delta_scof_adv_fgsm.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_019_delta_scof_adv_pgd.csv',

    # repeat-baseline
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_021_repeat_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_022_repeat_adv_fgsm.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_023_repeat_adv_pgd.csv',
    # repeat-N-LT
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_024_repeat_nlt_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_025_repeat_nlt_adv_fgsm.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_026_repeat_nlt_adv_pgd.csv',
    # repeat-SCoF
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_032_repeat_scof_adv_base.csv'
]
model_names = [
    'CNN(B)', 'CNN(FGSM)', 'CNN(PGD)',
    'conv(B)', 'conv(FGSM)', 'conv(PGD)',
    'conv(N-LT)(B)', 'conv(N-LT)(FGSM)', 'conv(N-LT)(PGD)',
    'conv(SCoF)(B)', 'conv(SCoF)(FGSM)', 'conv(SCoF)(PGD)',
    'delta(B)', 'delta(FGSM)', 'delta(PGD)',
    'delta(N-LT)(B)', 'delta(N-LT)(FGSM)', 'delta(N-LT)(PGD)',
    'delta(SCoF)(B)', 'delta(SCoF)(FGSM)', 'delta(SCoF)(PGD)',
    'repeat(B)', 'repeat(FGSM)', 'repeat(PGD)',
    'repeat(N-LT)(B)', 'repeat(N-LT)(FGSM)', 'repeat(N-LT)(PGD)',
    'repeat(SCoF)(B)',
]

####
# S9: Difference from Rate Coding with timestep=2
#### 
file_paths = [
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/CNN_030_baseline.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/CNN_038_baseline.csv', # redundant
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_029_conv_baseline.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_039_conv_baseline.csv', # redundant
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_037_conv_rate.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_040_conv_e_g.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_006_conv_nlt_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_033_conv_nlt_ternary.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_034_conv_nlt_ternary.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_031_delta_baseline.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_028_repeat_baseline.csv',
]
model_names = [
    'CNN(B)', 'CNN(B,R)', # redundant
    'conv(B)', 'conv(B,R)', # redundant
    'conv(rate)', 'conv(e-g)', 'conv(N-LT)', 'conv(Det.T)', 'conv(N-LT+Det.T)',
    'delta(B)', 'repeat(B)',
]

####
# Leaderboard
####
file_paths = [
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/CNN_038_baseline.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/IForest_075_baseline_legacy.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/PCA_077_baseline_legacy.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_070_conv_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_079_conv_scof_all_adv_base.csv', #
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_082_conv_scalar.csv', #
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_085_conv_nlt_eg_legacy.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_006_conv_nlt_adv_base.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_012_conv_XOX_legacy.csv',
    '/home/hwkang/dev-TSB-AD/TSB-AD/eval/metrics/multi/SpikeCNN_061_conv_neuron_random.csv',
]
model_names = [
    'CNN', 'IForest', 'PCA', 'SpikeCNN(B)', 
    'SpikeCNN(SCoF)', 'SpikeCNN(Scalar)', 'SpikeCNN(N-LT+e-g)',
    'SpikeCNN(Adv-N-LT)', 'SpikeCNN(XOX)', 'SpikeCNN(Random)', 
]