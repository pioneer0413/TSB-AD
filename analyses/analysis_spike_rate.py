import pandas as pd
from TSB_AD.model_wrapper import run_Semisupervise_AD
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.snn.params import running_params

if __name__ == "__main__":
    src_dir_path = '/home/hwkang/dev-TSB-AD/TSB-AD/Datasets/TSB-AD-M'
    TS_Name = "019_MITDB_id_1_Medical_tr_37500_1st_103211.csv"
    AD_Name = "SpikeCNN"
    Encoder_Name = "conv"

    df = pd.read_csv(src_dir_path + '/' + TS_Name)
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    feats = data.shape[1]
    slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
    train_index = TS_Name.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    output = run_Semisupervise_AD(
        data_train=data_train,
        data_test=data,
        TS_Name=TS_Name,
        AD_Name=AD_Name,
        Encoder_Name=Encoder_Name,
        local_running_params=running_params,
    )
