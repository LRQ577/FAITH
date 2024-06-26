from data_provider.data_loader import Dataset_Covid, Dataset_Custom, Dataset_Pred, Dataset_Custom_
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_Custom_,#Dataset_ETT_hour,
    'ETTm1': Dataset_Custom_,
    'traffic': Dataset_Custom,
    'electricity': Dataset_Custom_,
    'exchange': Dataset_Custom_,
    'weather': Dataset_Custom_,
    'covid': Dataset_Covid,
    'ECG': Dataset_Custom_,
    'metr': Dataset_Custom_,
    'covid': Dataset_Covid,
    'ETTh2': Dataset_Custom_,
    'ETTm2': Dataset_Custom_,
    'exchange_rate': Dataset_Custom_,
    'coils':Dataset_Custom_,
    'solar':Dataset_Custom_,
    'wiki': Dataset_Custom_,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
