###
# Scenario 1: Defaults to all
file_paths = [
    '/home/hwkang/TSB-AD/eval/metrics/multi/CNN_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_002_conv_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_009_conv_OXX.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_007_conv_OOX.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_028_conv_OOO.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_012_conv_XOX.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_027_conv_XOO.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_008_delta_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_019_delta_OXX.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_022_delta_OOX.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_025_delta_OOO.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_026_delta_XOX.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_024_delta_XOO.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_010_repeat_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_020_repeat_OXX.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_023_repeat_OOX.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_021_repeat_OOO.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_029_repeat_XOX.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_030_repeat_XOO.csv',
]

# Set custom model names (optional)
model_names = ['CNN(B)', 'conv(B)', 'conv(OXX)', 'conv(OOX)', 'conv(OOO)', 'conv(XOX)', 'conv(XOO)', 'delta(B)', 'delta(OXX)', 'delta(OOX)', 'delta(OOO)', 'delta(XOX)', 'delta(XOO)', 'repeat(B)', 'repeat(OXX)', 'repeat(OOX)', 'repeat(OOO)', 'repeat(XOX)', 'repeat(XOO)']

###
# Scenario 2: Learnable Threshold
file_paths = [
    '/home/hwkang/TSB-AD/eval/metrics/multi/CNN_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_008_delta_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_010_repeat_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_002_conv_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_043_conv_channel_all1s.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_042_conv_channel_all0s.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_040_conv_channel_random.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_036_conv_channel_he.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_039_conv_neuron_all1s.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_041_conv_neuron_all0s.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_037_conv_neuron_random.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_038_conv_neuron_he.csv',
]

# Set custom model names (optional)
model_names = ['CNN(B)', 'delta(B)', 'repeat(B)', 'conv(B)', 'all1s(C)', 'all0s(C)', 'random(C)', 'he(C)', 'all1s(N)', 'all0s(N)', 'random(N)', 'he(N)']

###
# Scenario 3: SCoF vs. Learnable Threshold
file_paths = [
    #'/home/hwkang/TSB-AD/eval/metrics/multi/CNN_baseline.csv',
    #'/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_002_conv_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_043_conv_channel_all1s.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_042_conv_channel_all0s.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_040_conv_channel_random.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_036_conv_channel_he.csv',
    #'/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_079_conv_scalar.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_039_conv_neuron_all1s.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_041_conv_neuron_all0s.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_037_conv_neuron_random.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_038_conv_neuron_he.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_049_conv_l_l.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_046_conv_l_e.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_044_conv_l_g.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_047_conv_e_l.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_050_conv_e_e.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_045_conv_e_g.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_048_conv_g_l.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_051_conv_g_e.csv',
    #'/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_080_conv_g_g.csv',
    #'/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_081_conv_linear.csv',
    #'/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_082_conv_exponential.csv',
    #'/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_083_conv_gaussian.csv',
]

# Set custom model names (optional)
#model_names = ['CNN(B)', 'conv(B)', 'all1s(N)', 'all0s(N)', 'random(N)', 'he(N)', 'l_l', 'l_e', 'l_g', 'e_l', 'e_e', 'e_g', 'g_l', 'g_e']
#model_names = ['C(all1s)', 'C(all0s)', 'C(random)', 'C(he)', '(scalar)', 'N(all1s)', 'N(all0s)', 'N(random)', 'N(he)', 'l_l', 'l_e', 'l_g', 'e_l', 'e_e', 'e_g', 'g_l', 'g_e', 'g_g', 'linear', 'exponential', 'gaussian']
model_names = [
    'C(all1s)', 'C(all0s)', 'C(random)', 'C(he)', 'N(all1s)', 'N(all0s)', 'N(random)', 'N(he)',
    'l_l', 'l_e', 'l_g', 'e_l', 'e_e', 'e_g', 'g_l', 'g_e'
]

###
# Scenario 4: SCoF vs. Baselines
file_paths = [
    '/home/hwkang/TSB-AD/eval/metrics/multi/CNN_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_002_conv_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_008_delta_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_010_repeat_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_049_conv_l_l.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_046_conv_l_e.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_044_conv_l_g.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_047_conv_e_l.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_050_conv_e_e.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_045_conv_e_g.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_048_conv_g_l.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_051_conv_g_e.csv',
]

model_names = ['CNN(B)', 'conv(B)', 'delta(B)', 'repeat(B)', 'l_l', 'l_e', 'l_g', 'e_l', 'e_e', 'e_g', 'g_l', 'g_e']

###
# Scenario 5: Baselines
file_paths = [
    '/home/hwkang/TSB-AD/eval/metrics/multi/IForest_075_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/LOF_076_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/CNN_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_002_conv_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_008_delta_baseline.csv',
    '/home/hwkang/TSB-AD/eval/metrics/multi/SpikeCNN_010_repeat_baseline.csv',
]
model_names = ['IForest', 'LOF', 'CNN(B)', 'conv(B)', 'delta(B)', 'repeat(B)']