# Dataset Setup

Download datasets from their official sources:

- KuaiRec: https://kuairec.com/
- WeChat21: https://algo.weixin.qq.com/

Place the downloaded files in the corresponding `dataset/<name>/raw_data/` directory, then run:

```bash
python dataset/kuairec/kuairec_process.py
```

or:

```bash
python dataset/wechat21/wechat21_process.py
```

The preprocessing scripts create the files consumed by the training entry point. Dataset artifacts are excluded from version control.
