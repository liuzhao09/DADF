# DADF：面向观看时长回归的分布感知纠偏框架

[![RecSys 2026](https://img.shields.io/badge/RecSys-2026-blue)](https://recsys.acm.org/recsys26/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)

**DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems**（RecSys 2026）官方代码实现。

## 背景与动机

观看时长预测是短视频推荐系统中的核心回归任务。在生产系统中，观看时长标签呈严重长尾分布，且预测偏差在不同时长区间上存在系统性差异：模型往往高估短播样本、低估长播样本。由于海量的短播误差与稀少的长播误差相互抵消，全局校准指标（如 PCOC）可能接近 1.0，制造出"全局已校准"的假象——我们称之为**伪平衡现象**。

**DADF** 通过在已部署的第一阶段预测器之上进行二阶段乘性残差纠偏来解决这一问题。框架不替换基础模型，而是学习一个轻量级的乘性纠偏因子：

```
y_hat = y_hat_base × b_hat
```

其中 `b_hat` 由排序特征、基础模型预测值以及辅助信号共同估计得出。

## 核心组件

1. **Box-Cox 变换** — 将纠偏目标变换到近似高斯空间，使回归更稳定，并支持方差感知的正态化约束。

2. **时长感知分桶专家** — 多个专家头分别针对不同视频时长区间建模偏差，通过可学习的软路由（Soft Routing）实现差异化纠偏。

3. **正态化正则损失** — 强制各时长桶内的 Box-Cox 变换分布满足近似高斯条件（均值→0，方差→1，偏度→0，峰度→3），提供分布层面的训练约束。

4. **辅助观看信号多任务** — 以短播率、完播率、长播率等辅助信号进行多任务学习，为纠偏因子估计提供额外的侧信息。

## 两阶段训练流程

- **阶段一（Warmup）**：仅训练基础模型，使其充分收敛。
- **阶段二（联合训练）**：联合优化基础模型与 DADF 纠偏网络，损失由基础 loss、Box-Cox MSE loss、绝对时长 Huber loss 和正态化正则 loss 组合而成。

## 支持的基础模型

| 模型 | 描述 |
|------|------|
| `wlr` | 加权逻辑回归（Wide&Deep + WBCE） |
| `vr`  | 普通回归（Wide&Deep + MSE） |
| `egmn`| 指数-高斯混合网络 |
| `tpm` | 树概率模型 |
| `d2q` | 时长到分位数回归 |
| `cread`| 有序回归解码框架 |
| `d2co`| 基于 GMM 映射的时长条件输出 |

## 项目结构

```
DADF/
├── model/
│   ├── __init__.py
│   ├── wd.py                   # Wide&Deep 基础模型
│   ├── egmn.py                 # EGMN 模型
│   ├── cread.py                # CREAD 模型
│   ├── d2q.py                  # D2Q 模型
│   ├── tpm.py                  # TPM 模型
│   ├── layers.py               # 共享网络层
│   ├── framework_utils.py      # 共享工具函数
│   └── v2_debias/
│       ├── __init__.py
│       ├── train.py            # DADF 主训练脚本
│       ├── network.py          # DebiasNetV2 架构定义
│       ├── adapter.py          # 基础模型适配层
│       ├── transforms.py       # Box-Cox 变换工具
│       └── losses.py           # DADF 损失函数
├── dataloader/
│   ├── __init__.py
│   ├── kuairec.py              # KuaiRec 数据加载器
│   └── wechat21.py             # WeChat21 数据加载器
├── dataset/
│   ├── kuairec/
│   │   ├── kuairec_process.py  # KuaiRec 数据预处理脚本
│   │   └── raw_data/           # 原始数据放置目录
│   └── wechat21/
│       ├── wechat21_process.py # WeChat21 数据预处理脚本
│       └── raw_data/           # 原始数据放置目录
├── utils.py                    # 评估指标
├── logger.py                   # 日志工具
├── run_DADF_wlr.sh             # 在 WLR 骨干上运行 DADF
└── README.md
```

## 环境安装

```bash
pip install torch torchvision numpy pandas scikit-learn
```

需要 Python 3.8+，PyTorch 1.12+。

## 数据集准备

### KuaiRec

KuaiRec 是来自快手短视频平台的全观测推荐数据集。

1. 从官方地址下载：[KuaiRec](https://kuairec.com/)
2. 将以下文件放置到 `dataset/kuairec/raw_data/` 目录：
   - `big_matrix.csv` — 主交互记录
   - `user_features.csv` — 用户特征
   - `item_daily_features.csv` — 视频特征
   - `item_categories.csv` — 视频类别特征
   - `kuairec_caption_category.csv` — 视频标题分类
3. 运行预处理脚本：

```bash
cd dataset/kuairec
python kuairec_process.py
```

将在 `dataset/kuairec/` 下生成 `kuairec_data.pkl`（10% 采样）和 `kuairec_data_full.pkl`（全量）。

### WeChat21

WeChat21 来自 2021 年微信大数据挑战赛。

1. 从以下地址下载：[WeChatBigData Challenge 2021](https://algo.weixin.qq.com/)
2. 将以下文件放置到 `dataset/wechat21/raw_data/` 目录：
   - `user_action.csv` — 用户交互记录
   - `feed_info.csv` — 视频元数据
3. 运行预处理脚本：

```bash
cd dataset/wechat21
python wechat21_process.py
```

将在 `dataset/wechat21/` 下生成 `wechat21_data.pkl`（10% 采样）和 `wechat21_data_full.pkl`（全量）。

**预处理说明：**
- 词表基于全量数据构建，归一化统计量（最大值、时长分桶）仅从训练集计算，不存在数据穿越。
- 按时间顺序进行 80% / 10% / 10% 的训练/验证/测试划分。

## 运行 DADF

### 快速开始：WLR 骨干上的 DADF

```bash
# 同时在 KuaiRec 和 WeChat21 上运行（并行）
bash run_DADF_wlr.sh

# 仅在 KuaiRec 上运行
DATASET=kuairec bash run_DADF_wlr.sh

# 仅在 WeChat21 上运行
DATASET=wechat21 bash run_DADF_wlr.sh

# 指定 GPU
DEVICE=cuda:1 bash run_DADF_wlr.sh

# 串行模式（便于调试）
SEQUENTIAL=1 DATASET=kuairec bash run_DADF_wlr.sh
```

### 手动训练

```bash
# DADF + WLR 在 KuaiRec 上（K=4 等频时长分桶）
python model/v2_debias/train.py \
    --base_model wlr \
    --dataset_name kuairec \
    --dataset_path dataset \
    --full-data \
    --two_stage_debias \
    --debias_bucket_num 4 \
    --duration_thresh_mode quantile \
    --epoch 25 \
    --warmup_epoch 3 \
    --patience 6 \
    --base_lr 0.1 \
    --debias_lr 0.02 \
    --weight_decay 1e-6 \
    --abs_time_weight 0.8 \
    --nr_weight 0.05 \
    --use_aux_targets \
    --aux_targets svr,fpr,evr,lvr,evr_p60,lvr_p80,lvr_p90 \
    --aux_target_weight 0.10 \
    --device cuda:0

# DADF + WLR 在 WeChat21 上（K=3 等频时长分桶）
python model/v2_debias/train.py \
    --base_model wlr \
    --dataset_name wechat21 \
    --dataset_path dataset \
    --full-data \
    --two_stage_debias \
    --debias_bucket_num 3 \
    --duration_thresh_mode quantile \
    --epoch 30 \
    --warmup_epoch 3 \
    --patience 6 \
    --base_lr 0.1 \
    --debias_lr 0.01 \
    --weight_decay 1e-6 \
    --abs_time_weight 0.8 \
    --nr_weight 0.05 \
    --use_aux_targets \
    --aux_targets svr,fpr,evr,lvr,evr_p60,lvr_p80,lvr_p90 \
    --aux_target_weight 0.10 \
    --device cuda:0
```

### 使用其他基础模型

将 `--base_model wlr` 替换为 `vr`、`egmn`、`tpm`、`d2q`、`cread` 中的任意一个即可。

## 评估指标

| 指标 | 说明 |
|------|------|
| **MAE**（秒） | 观看时长预测的平均绝对误差 |
| **XAUC** | 连续标签的排序质量指标（AUC 的推广形式） |

## 关键超参数

| 参数 | 说明 | KuaiRec | WeChat21 |
|------|------|---------|---------|
| `--debias_bucket_num` | 时长专家桶数量 | 4 | 3 |
| `--duration_thresh_mode` | 分桶方式（`quantile`=等频，`physical`=物理阈值） | quantile | quantile |
| `--warmup_epoch` | Warmup 轮数（仅训练基础模型） | 3 | 3 |
| `--epoch` | 联合训练轮数 | 25 | 30 |
| `--debias_lr` | DADF 模块学习率 | 0.02 | 0.01 |
| `--nr_weight` | 正态化正则损失权重 | 0.05 | 0.05 |
| `--abs_time_weight` | 绝对时长 Huber 损失权重 | 0.8 | 0.8 |
| `--aux_target_weight` | 辅助任务损失权重 | 0.10 | 0.10 |

## 引用

如果本工作对您有帮助，请引用：

```bibtex
@inproceedings{yang2026dadf,
  title     = {DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems},
  author    = {Yiqing Yang and Xinlong Zhao and Zhao Liu and Xiao Lv and Ruiming Tang},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems (RecSys)},
  year      = {2026},
  address   = {Minneapolis, MN, USA}
}
```

## 许可证

本项目使用 MIT 许可证开源。
