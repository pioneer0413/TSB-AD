import numpy as np

running_params = {

    'save': True,
    'save_file_path': None,

    'load': False,
    'load_file_path': None,

    'off_cuda': False,

    'AD_Name': None,
    'Encoder_Name': None,
    'postfix': None,
    'id_code': 0,

    'verbose': False,         # Visualization
    'analysis': False,
    'early_stop': True,     
    
    'batch_size': 128,          
    'epochs': 10,               
    'validation_size': 0.2,     

    'num_enc_features': 8,      
    'window_size': 50,          
    'predict_time_steps': 1,    

    'zero_pruning': False,

    'adversarial':{
        'type': 'fgsm',
        'fgsm': {
            'eps': 0.1,          
            'norm': np.inf,
        },
        'pgd': {
            'eps': 0.1,           
            'eps_iter': 0.01,     
            'nb_iter': 40,       
            'norm': np.inf
        }
    },

    'CNN': {
        'encoding': False,
    },
    
    'SpikeCNN': {
        'optimizer': 'adam',    
        'lr': 0.0008,           
        'scheduler': 'steplr',  
        'loss': 'mse',          
        'mu': None,             
        'sigma': None,          
        'eps': 1e-10,           
    },

    'SpikeCNNModel': {
        'num_channel': [32, 40],
        'kernel_size': 3,       
        'stride': 1,            
        'dropout_rate': 0.25,
    },

    'normalization_layer': {    
        'type': 'bn',       # [bn, ln, gn, bntt, None]
        'gn': {
            'num_groups': 4
        }
    },

    'activations': {
        'activation': 'ternary',
        'common': {
            'num_steps': 10,
            'beta': 0.99,           
        },
        'dynamic_receptive': {
            'adaptation': False,
            'sensitization': False,
            'delta_activation': False,
            'burst': False,
            'learn_threshold': False,
            'granularity': 'scalar',
            'learn_beta': False,
            'reset_mechanism': 'subtract',
            'integration': 'concat',
        },
        'binary': {
            'learn_threshold': False,
            'granularity': 'neuron',    
            'threshold_init': 'all-1s', 
            'second_chance': False,     
            'deterministic': False,     
            'rate_encode': False,
            'adaptive': False,
        },
        'ternary': {
            'pos_threshold': 1.0,
            'neg_threshold': -1.0,
            'adaptive': False,
        },
    },

    'encoders': {
        'type': 'conv',
        'poisson': {

        },
        'repeat': {

        },
        'delta': {

        },
        'conv':{
            'kernel_size': 3,       
            'stride': 1,  
            'dilation': 1,
        },          
    },
}