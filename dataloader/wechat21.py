from dataloader.kuairec import KUAIRECDataset

import os
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader


class WeChat21DataLoader(object):
    """
    WeChat21 数据集加载器。
    pkl 格式与 KuaiRec 完全相同：dict，keys 为
    'train', 'test', 'description', 'play_duration_max', 'video_duration_max'。
    数据在 CPU 上预加载，每个 batch 通过 collate_fn 懒惰移动到 device。
    """

    def __init__(self, dataset_name, dataset_path, device, bsz=32):
        assert os.path.exists(dataset_path), '{} does not exist'.format(dataset_path)
        with open(dataset_path, 'rb+') as f:
            data = pickle.load(f)
        self.dataset_name       = dataset_name
        self.dataloaders        = {}
        self.description        = data['description']
        self.play_duration_max  = float(data.get('play_duration_max',  1200000.0))
        self.video_duration_max = float(data.get('video_duration_max',  600000.0))

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
