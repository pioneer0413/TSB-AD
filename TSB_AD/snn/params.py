running_params = {

    'data': {
        'dataset_dir': '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M',
        'file_list': '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv',
        'score_dir': '/home/hwkang/dev-TSB-AD/TSB-AD/scores',
        'result_dir': '/home/hwkang/dev-TSB-AD/TSB-AD/results/full/',
        'swap': False,
        'shuffle': False,
    },

    'meta': {
        'root_dir_path': '/home/hwkang/dev-TSB-AD/TSB-AD/',
        'AD_Name': None,
        'Encoder_Name': None,
        'postfix': None,
        'id_code': 0,
        'base_file_name': None,
    },

    'analysis': {
        'wandb': False,
        'spikerate': False,
        'spike': False,
    },

    'model': {
        'device_type': 'cuda',
        'batch_size': 256,
        'max_epochs': 1000,
        'validation_size': 0.2,
        'window_size': 50,
        'predict_time_steps': 1,
    },

    'ParallelSNNModel': {
        'num_enc_features': 32,
        'norm_type': 'ln',
        'neuron_type': 'spikingjelly',
        'step_mode': 'm',
        'encoding_kernel': [7, 3, 5],
    },
}