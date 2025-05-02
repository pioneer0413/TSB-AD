running_params = {

    'verbose': False,         # Visualization
    'save_encoding': False,     # Visualization
    'trace_threshold': False,   # Visualization
    'early_stop': True,     # Execution time
    
    'batch_size': 128,          # CV
    'epochs': 50,               # CV
    'validation_size': 0.2,     # CV

    'num_enc_features': 8,      # CV
    'window_size': 50,          # CV
    'predict_time_steps': 1,    # CV
    
    'SpikeCNN': {
        'optimizer': 'adam',    # CV
        'lr': 0.0008,           # CV
        'scheduler': 'steplr',  # CV
        'loss': 'mse',          # CV
        'mu': None,             # CV
        'sigma': None,          # CV
        'eps': 1e-10,           # CV
    },

    'SpikeCNNModel': {
        'num_channel': [32, 40],# CV
        'kernel_size': 3,       # CV
        'stride': 1,            # CV
        'dropout_rate': 0.25,   # CV
        'learn_threshold': False# CV
    },

    'encoders': {
        'kernel_size': 3,       # CV
        'stride': 1,            # CV
        'normalization_layer': {    # Independent Variable
            'type': 'bn',       # Independent Variable ['bn', 'gn', 'ln']
            'gn': {
                'num_groups': 4
            }
        },
        'learn_threshold': False,   # Independent Variable ['True', 'False']
        'granularity': 'neuron',    # Independent Variable ['neuron', 'channel']
        'threshold_init': 'all-1s', # Independent Variable ['all-1s', 'all-0s', 'random', 'he', 'scalar']
        'second_chance': False,     # Independent Variable ['True', 'False']
        'sub_threshold': {
            'type': 'exponential',  # Independent Variable ['linear', 'exponential', 'gaussian']
            'linear': {
                'adaptive_margin': False,    # Independent Variable ['True', 'False']
                'scale_factor': 0.5,         # Independent Variable
                'margin': 0.5,               # Independent Variable
                'eps': 1e-10,                # Independent Variable
            },
            'exponential': {
                'alpha': 5.0,               # Independent Variable
            },
            'gaussian': {
                'sigma': 0.5                # Independent Variable
            },
        },
        'ternary': False,                   # Independent Variable ['True', 'False']
        'supra_threshold': {
            'type': 'linear',               # Independent Variable ['linear', 'exponential', 'gaussian']
            'linear': {
                'adaptive_margin': False,       # Independent Variable ['True', 'False']
                'scale_factor': 0.5,            # Independent Variable
                'margin': 0.5,                  # Independent Variable
                'eps': 1e-10,                   # Independent Variable
            },
            'exponential': {
                'alpha': 5.0,                # Independent Variable
            },
            'gaussian': {
                'sigma': 0.5                # Independent Variable
            }
        },
    }
}