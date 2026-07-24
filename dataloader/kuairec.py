import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader

class KUAIRECDataset(Dataset):
    def __init__(self, dataset_name, df, description):
        super(KUAIRECDataset, self).__init__()
        self.dataset_name = dataset_name
        self.length = len(df)

        self.name2array = {name: torch.from_numpy(np.array(list(df[name])).reshape([self.length, -1]))
                           for name in df.columns}
        self.format(description)
        self.features = [name for name, size, type in description if type != 'label']
        self.label = 'play_time'

    def format(self, description):
        for name, size, type in description:
            if type == 'spr' or type == 'seq' or type == 'aux':
                self.name2array[name] = self.name2array[name].to(torch.long)
            elif type == 'ctn' or type == 'seqm' or type == 'other':
                self.name2array[name] = self.name2array[name].to(torch.float32)
            elif type == 'label':
                self.name2array[name] = self.name2array[name].to(torch.float32)
            else:
                raise ValueError('unkwon type {}'.format(type))

    def __getitem__(self, index):
        return {name: self.name2array[name][index] for name in self.features}, \
                self.name2array[self.label][index].squeeze()

    def __len__(self):
        return self.length

class KUAIRECDataLoader(object):

    def __init__(self, dataset_name, dataset_path, device, bsz=32):
        assert os.path.exists(dataset_path), '{} does not exist'.format(dataset_path)
        with open(dataset_path, 'rb+') as f:
            data = pickle.load(f)
        self.dataset_name = dataset_name
        self.dataloaders = {}
        self.description = data['description']
        self.play_duration_max = float(data.get('play_duration_max', 999639.0))
        self.video_duration_max = float(data.get('video_duration_max', 600.0))

        def _make_collate(dev):
            def collate_fn(batch):
                features_list, labels_list = zip(*batch)
                features_batch = {
                    k: torch.stack([f[k] for f in features_list]).to(dev)
                    for k in features_list[0]
                }
                labels_batch = torch.stack(labels_list).to(dev)
                return features_batch, labels_batch
            return collate_fn

        _non_df_keys = {'description', 'play_duration_max', 'video_duration_max'}
        for key, df in data.items():
            if key in _non_df_keys or not isinstance(df, pd.DataFrame):
                continue
            dataset = KUAIRECDataset(dataset_name, df, self.description)
            self.dataloaders[key] = DataLoader(
                dataset,
                batch_size=bsz,
                shuffle=(key == 'train'),
                collate_fn=_make_collate(device),
            )
        self.keys = list(self.dataloaders.keys())

    def __getitem__(self, name):
        assert name in self.keys, '{} not in keys of datasets'.format(name)
        return self.dataloaders[name]
