# DADF：面向观看时长回归的分布感知纠偏框架

[![论文稿件](https://img.shields.io/badge/status-manuscript-blue)](#代码范围与复现说明)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)

**DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems** 的公开参考实现。

## 代码范围与复现说明

本仓库提供与论文公开数据集实验对应的 DADF 结构参考实现，**不是完整的线上生产代码**。线上专有特征流程、第一阶段多任务塔、服务基础设施和系统优化均未开源。公开代码使用公共数据集可获得的特征与辅助标签重建核心结构，主要用于理解和评估论文方法。

未显式传入实验参数时，公开实现与论文方法保持一致。纠偏训练阶段冻结第一阶段预测器，时长分组采用 hard routing，Box-Cox 参数是组级参数，不进行 per-sample 自适应，也不包含额外的 user-level 或 video-level correction stage。默认目标仅包含变换空间拟合、绝对时长、矩正则和辅助任务四项。预测侧正则、lambda 平滑、峰度正则、分桶重加权、backbone 自动调参和窄区间推理裁剪默认均关闭。

## 背景与动机

观看时长预测是短视频推荐系统中的核心回归任务。在生产系统中，观看时长标签呈严重长尾分布，且预测偏差在不同时长区间上存在系统性差异：模型往往高估短播样本、低估长播样本。由于海量的短播误差与稀少的长播误差相互抵消，全局校准指标（如 PCOC）可能接近 1.0，制造出"全局已校准"的假象——我们称之为**伪平衡现象**。

**DADF** 通过在已部署的第一阶段预测器之上进行二阶段乘性残差纠偏来解决这一问题。框架不替换基础模型，而是学习一个轻量级的乘性纠偏因子：

```
y_hat = y_hat_base × b_hat
```

其中 `b_hat` 由排序特征、基础模型预测值以及辅助信号共同估计得出。

## 核心组件

1. **动态分布感知变换** — 对乘性纠偏目标使用分组 Box-Cox 变换，并结合变换空间拟合与矩正则，使长尾目标更稳定。

2. **纠偏因子感知模块** — 按视频时长将样本分组，并通过 hard one-hot routing 选择对应专家，与论文定义保持一致。

3. **多标签感知表征** — 对辅助 logit 进行固定非线性投影，并与辅助 tower 表征和共享纠偏上下文拼接，再通过 MLP 估计纠偏因子。

## 两阶段训练流程

- **阶段一（Warmup）**：仅训练基础模型，使其充分收敛。
- **阶段二（纠偏训练）**：冻结第一阶段预测器，优化 DADF 纠偏网络；损失包括变换空间拟合、绝对时长、矩正则和辅助任务损失。

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
- 各数据划分共享特征词表。
- 训练/验证/测试样本按固定随机种子进行 80% / 10% / 10% 随机划分。
- 标签相关的归一化统计量（最大值、时长分桶）仅从训练集计算。

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
| **XAUC** | 所有无序样本对上的严格顺序一致率；标签或预测相同时均计 0 |

## 主要结果

以下数值直接与当前论文表格对齐，确保仓库和论文使用同一报告口径。MAE 单位为秒，越低越好；XAUC 越高越好。

| Backbone | 方法 | KuaiRec MAE | KuaiRec XAUC | WeChat21 MAE | WeChat21 XAUC |
|---|---|---:|---:|---:|---:|
| VR | Base | 4.584 | 0.5578 | 18.681 | 0.6766 |
| VR | w/ TranSUN | 4.478 | 0.5693 | 18.571 | 0.6787 |
| VR | w/ DADF | **4.235** | **0.6125** | **17.912** | **0.6902** |
| WLR | Base | 4.414 | 0.5941 | 18.215 | 0.6861 |
| WLR | w/ TranSUN | 4.364 | 0.5965 | 18.133 | 0.6876 |
| WLR | w/ DADF | **4.172** | **0.6227** | **17.838** | **0.6934** |
| TPM | Base | 4.459 | 0.5495 | 19.545 | 0.6570 |
| TPM | w/ TranSUN | 4.361 | 0.5971 | 18.529 | 0.6814 |
| TPM | w/ DADF | **4.166** | **0.6233** | **18.109** | **0.6898** |
| D2Q | Base | 4.123 | 0.6319 | 17.544 | 0.6935 |
| D2Q | w/ TranSUN | 4.323 | 0.6082 | 17.855 | 0.6925 |
| D2Q | w/ DADF | **4.106** | **0.6345** | **17.534** | **0.6946** |
| CREAD | Base | 4.346 | 0.5927 | 19.128 | 0.6679 |
| CREAD | w/ TranSUN | 4.395 | 0.5958 | 18.515 | 0.6824 |
| CREAD | w/ DADF | **4.189** | **0.6211** | **18.164** | **0.6903** |
| D²CO | Base | 4.613 | 0.5687 | 18.558 | 0.6861 |
| D²CO | w/ TranSUN | 4.300 | 0.6097 | 18.080 | 0.6868 |
| D²CO | w/ DADF | **4.168** | **0.6233** | **17.683** | **0.6952** |
| EGMN | Base | 4.081 | 0.6245 | 18.330 | 0.6896 |
| EGMN | w/ TranSUN | 4.255 | 0.6120 | 18.099 | 0.6892 |
| EGMN | w/ DADF | **4.002** | **0.6257** | **17.955** | **0.6911** |

研究中使用匹配随机种子重复离线比较以检查稳定性。为保持表格紧凑，这里保留论文中的点估计而不展开逐次方差；复现实验应让配对方法使用同一组种子，并根据生成的日志计算不确定性。

### Baseline 优化与公平性说明

我们不会通过刻意弱化第一阶段 baseline 来放大二阶段纠偏方法的相对收益。在接入 TranSUN 或 DADF 之前，我们首先基于验证集优化每个复现的 baseline，使其达到具有竞争力的工作点。所有 backbone 使用一致的特征预处理和数据划分、16 维稀疏特征 embedding，以及规模可比的 MLP（隐藏层维度统一为 256、128、64）。各方法特有的学习率、损失权重、离散化粒度和混合分量数量等参数均依据验证集效果选择。对于同一 backbone，Base、w/ TranSUN 和 w/ DADF 共用完全相同且已经冻结的第一阶段预测器，因此组内差异反映的是纠偏方法本身，而不是来自更弱的基础模型。

作为外部合理性校验，原始 [RecSys 2025 EGMN 论文](https://arxiv.org/pdf/2508.12665) 报告的 EGMN 结果为：KuaiRec 上 MAE 4.204 / XAUC 0.6093，WeChat 上 MAE 18.88 / XAUC 0.6692。我们调优后的 EGMN baseline 分别达到 4.081 / 0.6245 和 18.330 / 0.6896，在两套数据的两个指标上均更优。因此，DADF 的对比起点并不是一个被劣化的 EGMN 复现，而是一个已经充分优化的强 baseline。考虑到不同仓库间的数据预处理细节和评估划分可能并不完全一致，这一跨论文比较应视为外部合理性校验，而不是严格受控的 head-to-head 实验。

### WLR 消融实验

| 变体 | KuaiRec MAE | KuaiRec XAUC | WeChat21 MAE | WeChat21 XAUC |
|---|---:|---:|---:|---:|
| Full DADF | 4.1723 | 0.6227 | 17.8376 | 0.6934 |
| w/o 动态分布感知变换 | 4.1901 | 0.6210 | 17.8748 | 0.6930 |
| w/o 纠偏因子感知模块 | 4.1823 | 0.6212 | 17.8454 | 0.6931 |
| w/o 多标签感知表征 | 4.1865 | 0.6204 | 17.9137 | 0.6920 |

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

Hard duration routing、等频分桶、辅助 heads 和冻结第一阶段预测器均为默认行为。`--soft_routing`、`--joint_finetune_base`、`--nr_pred_weight`、`--lambda_smooth_weight`、`--kurtosis_weight`、`--bucket_reweighting` 与 `--backbone_autotune` 都是显式实验选项，不影响与论文对齐的默认路径。

## 引用

如果本工作对您有帮助，请引用：

```bibtex
@misc{yang2026dadf,
  title     = {DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems},
  author    = {Yiqing Yang and Xinlong Zhao and Zhao Liu and Xiao Lv and Ruiming Tang},
  year      = {2026},
  note      = {Manuscript}
}
```

## 许可证

本项目使用 MIT 许可证开源。
