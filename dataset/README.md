# Dataset Setup

## KuaiRec

Place the following raw data files under `dataset/kuairec/raw_data/`:

| File | Description |
|------|-------------|
| `big_matrix.csv` | Main interaction matrix (user-video interactions) |
| `user_features.csv` | User profile features |
| `item_daily_features.csv` | Item daily features (video type, music, tags) |
| `item_categories.csv` | Item category information |
| `kuairec_caption_category.csv` | Item caption and category labels |

Download from: https://kuairec.com/

After placing files, run:
```bash
cd dataset/kuairec
python kuairec_process.py
```

Output: `kuairec_data.pkl` (10% sample), `kuairec_data_full.pkl` (full data)

---

## WeChat21

Place the following raw data files under `dataset/wechat21/raw_data/`:

| File | Description |
|------|-------------|
| `user_action.csv` | User interaction records (userid, feedid, play, date, device, ...) |
| `feed_info.csv` | Video/feed metadata (feedid, authorid, videoplayseconds, keywords, ...) |

Download from: https://algo.weixin.qq.com/ (WeChatBigData Challenge 2021)

After placing files, run:
```bash
cd dataset/wechat21
python wechat21_process.py
```

Output: `wechat21_data.pkl` (10% sample), `wechat21_data_full.pkl` (full data)
