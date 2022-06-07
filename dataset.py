import sys
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
import category_encoders as ce
from sklearn.preprocessing import KBinsDiscretizer
import pickle
import augmentation as A


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def get_transform(conf_augmentation):
    def get_object(trans):
        if trans.type in {'Compose', 'OneOf'}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(A, trans.type)(augs_tmp, **trans.params)

        if hasattr(A, trans.type):
            return getattr(A, trans.type)(**trans.params)
        else:
            return eval(trans.type)(**trans.params)
    augs = []
    if conf_augmentation is not None:
        augs += [get_object(aug) for aug in conf_augmentation]

    return A.Compose(augs)


def add_target_encoding(tr_df, val_df, test_df, n_bins=500,
                        strategy='uniform', feature='u_in',
                        smoothing=1, skip_test=True):
    if smoothing == 1:
        sm = ''
    else:
        sm = '_' + str(smoothing)
        
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    tr_df[f'{feature}_{n_bins}bins{sm}'] = kbd.fit_transform(tr_df[feature].values.reshape(-1, 1))
    val_df[f'{feature}_{n_bins}bins{sm}'] = kbd.transform(val_df[feature].values.reshape(-1, 1))

    test_df[f'{feature}_{n_bins}bins{sm}'] = kbd.transform(test_df[feature].values.reshape(-1, 1))

    tr_df[f'{feature}_{n_bins}bins{sm}']  = tr_df[f'{feature}_{n_bins}bins{sm}'].astype('category')
    val_df[f'{feature}_{n_bins}bins{sm}']  = val_df[f'{feature}_{n_bins}bins{sm}'].astype('category')

    test_df[f'{feature}_{n_bins}bins{sm}']  = test_df[f'{feature}_{n_bins}bins{sm}'].astype('category')

    feature_name = f'target_encode_{feature}_{n_bins}bins{sm}'
    te = ce.target_encoder.TargetEncoder(verbose=1, smoothing=smoothing)
    tr_df[feature_name] = te.fit_transform(tr_df[f'{feature}_{n_bins}bins{sm}'], tr_df['pressure']) 
    val_df[feature_name] = te.transform(val_df[f'{feature}_{n_bins}bins{sm}']) 
    test_df[feature_name] = te.transform(test_df[f'{feature}_{n_bins}bins{sm}'])

    te_col_name = f'target_encode_{feature}_{n_bins}bins{sm}'

    # if te_col_name not in features:
    #     features.append(te_col_name)

    return tr_df, val_df, test_df, feature_name


def add_target_cencoding(tr_df, val_df, test_df, feature='RC', smoothing=1,):

    if smoothing == 1:
        sm = ''
    else:
        sm = '_' + str(smoothing)

    feature_name = f'target_encode_{feature}'
    te = ce.target_encoder.TargetEncoder(verbose=1, smoothing=smoothing)
    tr_df[feature_name] = te.fit_transform(tr_df[feature], tr_df['pressure']) 
    val_df[feature_name] = te.transform(val_df[feature]) 
    test_df[feature_name] = te.transform(test_df[feature])

    return tr_df, val_df, test_df, feature_name



class VentilatorDatasetV1(Dataset):
    def __init__(self, df):
        if "pressure" not in df.columns:
            df['pressure'] = 0

        self.df = df.groupby('breath_id').agg(list).reset_index()
        
        self.prepare_data()
                
    def __len__(self):
        return self.df.shape[0]
    
    def prepare_data(self):
        self.pressures = np.array(self.df['pressure'].values.tolist())
        
        rs = np.array(self.df['R'].values.tolist())
        cs = np.array(self.df['C'].values.tolist())
        u_ins = np.array(self.df['u_in'].values.tolist())
        
        self.u_outs = np.array(self.df['u_out'].values.tolist())
        
        self.inputs = np.concatenate([
            rs[:, None], 
            cs[:, None], 
            u_ins[:, None], 
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None]
        ], 1).transpose(0, 2, 1)

    def __getitem__(self, idx):
        data = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }
        return data


class VentilatorDatasetV2(Dataset):
    def __init__(self, df, config, mode, transform=None):
        self.df = df
        self.groups = df.groupby('breath_id').groups
        self.keys = list(self.groups.keys())
        self.config = config
        self.mode = mode
        self.transform = transform
        
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indexes = self.groups[self.keys[idx]]
        df = self.df.iloc[indexes]

        if self.transform is not None:
            x = df[self.config.aug_cols].values
            df[self.config.aug_cols] = np.squeeze(self.transform(x[np.newaxis, :, :]))

        cate_seq_x = torch.LongTensor(df[self.config.cate_seq_cols].values)
        cont_seq_x = torch.FloatTensor(df[self.config.cont_seq_cols].values)
        u_out = torch.LongTensor(df['u_out'].values) # original



        if self.mode != "test":
            label = torch.FloatTensor(df['pressure'].values)
            return {
                "cate_seq_x": cate_seq_x,
                "cont_seq_x": cont_seq_x,
                "u_out": u_out,
                "targets": label,
                # "ids": ids
            }
        else:
            return {
                "cate_seq_x": cate_seq_x,
                "cont_seq_x": cont_seq_x,
                "u_out": u_out,
                # "ids": ids
            }


class DataModule(pl.LightningDataModule):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = "./data/"

        self.test_df = pd.read_csv(os.path.join(self.data_dir, f"{self.config.globals.test_df}"))
        df = pd.read_csv(os.path.join(self.data_dir, f"{self.config.globals.df}"))

        if self.config.globals.fold_num is not None:
            fold = int(self.config.globals.fold_num)
            self.train_df = df[df["kfold"] != fold].reset_index()
            self.valid_df = df[df["kfold"] == fold].reset_index()
        else:
            self.train_df = df
            self.valid_df = df[df["kfold"] == 0].reset_index()

        print("train_df.shape: ", self.train_df.shape)
        print("valid_df.shape: ", self.valid_df.shape)



        if config.Dataset.target_encoding:
            print("target encoding")
            self.train_df, self.valid_df, self.test_df, feature_name = add_target_encoding(self.train_df, self.valid_df, self.test_df, n_bins=300, feature="u_in", smoothing=1)
            self.train_df, self.valid_df, self.test_df, feature_name = add_target_encoding(self.train_df, self.valid_df, self.test_df, n_bins=600, feature="u_in", smoothing=1)
            self.train_df, self.valid_df, self.test_df, feature_name = add_target_encoding(self.train_df, self.valid_df, self.test_df, n_bins=1000, feature="u_in", smoothing=1)
            self.train_df, self.valid_df, self.test_df, feature_name = add_target_encoding(self.train_df, self.valid_df, self.test_df, n_bins=100, feature="u_in", smoothing=1)
            if "u_in_round2" in self.train_df.columns:
                self.train_df, self.valid_df, self.test_df, feature_name = add_target_encoding(self.train_df, self.valid_df, self.test_df, n_bins=500, feature="u_in_round2", smoothing=10)


            # self.train_df, self.valid_df, self.test_df, feature_name = add_target_cencoding(self.train_df, self.valid_df, self.test_df, feature='RC', smoothing=1,)
            # self.train_df, self.valid_df, self.test_df, feature_name = add_target_cencoding(self.train_df, self.valid_df, self.test_df, feature='breath_id', smoothing=1,)
            print("finish!!")

        self.batch_size = self.config.Dataset.batch_size

        print(self.config.Dataset.cont_seq_cols)  
        
        if config.globals.task == "classification":
            target_dic = load_dict("./data/target_dic.pkl")
            
            print("reg  ------------>  cls")
            self.train_df["pressure"] = self.train_df["pressure"].map(target_dic)
            self.valid_df["pressure"] = self.valid_df["pressure"].map(target_dic)


    def get_datagrame(self, mode):
        assert mode in {"train", "valid", "test"}

        if mode == "train":
            return self.train_df
        elif mode == "valid":
            return self.valid_df
        elif mode == "test":
            return self.test_df

    def get_ds(self, mode):
        assert mode in {"train", "valid", "test"}

        df = self.get_datagrame(mode=mode)

        train_transforms = A.Compose([
                                    A.OneOf([
                                        A.Jitter(p=0.8),
                                        A.Scaling(p=0.8),
                                    ]),
                                    A.OneOf([        
                                        # A.Permutation(p=0.3),
                                        A.Magnitude_warp(p=0.7),
                                        A.Time_warp(p=0.7),
                                        A.Window_slice(p=0.7),
                                        A.Window_warp(p=0.7)
                                        ])
                                    ])

        # train_transforms = None
        valid_transforms = None
     
        if mode != "test":
            ds = globals()[self.config.Dataset.train.type](
                df=df,
                config=self.config.Dataset,
                mode="train",
                transform = train_transforms if mode == "train" else valid_transforms
            )
        else:
            ds = globals()[self.config.Dataset.val.type](
                df=df,
                config=self.config.Dataset,
                mode = "test",
                transform = valid_transforms
            )
        return ds

    def get_loader(self, mode):
        dataset = self.get_ds(mode=mode)
        # sampler = BalanceClassSampler(labels=dataset.get_labels(), mode="downsampling") if mode == "train" else SequentialSampler(dataset)
        # sampler = RandomSampler if mode == "train" else SequentialSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset,
            # sampler=sampler,
            batch_size=self.batch_size if mode == "train" else 128,
            shuffle=True if mode == "train" else False,
            num_workers=4,
            drop_last=True if mode == "train" else False
        )

    def train_dataloader(self):
        return self.get_loader(mode="train")

    def val_dataloader(self):
        return self.get_loader(mode="valid")

    def test_dataloader(self):
        return self.get_loader(mode="test")


if __name__ == "__main__":
    import time
    from pprint import pprint

    import matplotlib.pyplot as plt
    import yaml
    from addict import Dict

    with open(f"configs/{sys.argv[1]}.yml", "r")  as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    cfg = Dict(yml)

    pprint(cfg)

    datamodule=DataModule(cfg)
    print(len(datamodule.val_dataloader()))
    batch = iter(datamodule.train_dataloader()).next()

    print(batch["cate_seq_x"].shape)
    print(batch["cont_seq_x"].shape)
    print(batch["targets"].shape)
    print(batch["u_out"].shape)

    print(batch["targets"])
    print(batch["u_out"].min(), batch["u_out"].max())

    # batch = iter(datamodule.test_dataloader()).next()