# DADF：面向观看时长回归的分布感知纠偏框架

论文 **DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems** 的参考实现。

[English README](README.md)

## 方法概览

DADF 是一个轻量级的二阶段观看时长纠偏框架。它冻结已经训练完成的第一阶段预测器，仅学习其中可由推理时信号预测的条件残差。修正后的输出保持原有标量接口：

```text
修正后观看时长 = 第一阶段观看时长预测 * 纠偏因子
```

本文中的“纠偏”特指修正可预测的条件残差，而不是对第一阶段模型进行全局重校准。视频时长用于索引不同的分布区间，但不被视为预测误差的唯一原因。

## 核心模块

代码包含三个核心组件：

1. **分组目标变换（Regime-Specific Target Transformation）**：对乘性纠偏目标应用可学习的组级 Box-Cox 变换。
2. **时长索引专家路由（Duration-Indexed Expert Routing）**：按视频时长确定所属区间，并通过 hard routing 仅执行一个纠偏专家。
3. **辅助行为表征（Auxiliary Behavioral Representation）**：对辅助任务 logit 进行固定非线性展开，再与辅助 tower 表征和共享纠偏上下文融合。

纠偏训练目标包含四项：

- 变换空间 MSE；
- 原始观看时长空间 Huber loss；
- 分组统计矩正则；
- 辅助任务 BCE loss。

构造纠偏目标时会对第一阶段预测执行 stop-gradient。辅助标签只提供训练监督；推理阶段使用预测得到的辅助 logit、tower 表征以及其他推理时可用特征。

## 公开实现范围

本仓库提供用于 KuaiRec 和 WeChat21 公共数据集实验的 DADF 参考实现，包括数据预处理、七种第一阶段预测器、DADF 纠偏模块、训练流程，以及 MAE/XAUC 评估。

仓库聚焦可复现的公开研究路径，不分发数据集文件和部署相关基础设施。公开版本使用两套公共数据中可构造的标签训练辅助 heads，并采用相同的辅助表征设计。

## 实验设置

离线实验统一使用以下第一阶段模型和纠偏训练设置：

- 所有 backbone 使用相同的特征预处理和 80%/10%/10% 数据划分；
- 稀疏特征 embedding 维度统一为 16；
- 适用的 backbone MLP 隐藏层宽度统一为 256、128 和 64；
- 每个第一阶段 backbone 均先在验证集上选择 checkpoint，再进行纠偏训练；
- 同一 backbone 下的 Base 与 DADF 从完全相同的第一阶段 checkpoint 出发；
- DADF 训练期间冻结第一阶段 checkpoint；
- 配对实验使用相同的数据划分和随机种子。

这些设置保证同一 backbone 组内不同方法使用一致的模型容量和数据处理方式。

## 支持的第一阶段预测器

| 名称 | 第一阶段建模方式 |
|---|---|
| `vr` | 直接数值回归 |
| `wlr` | 加权逻辑回归 |
| `tpm` | 基于树结构的有序观看时长建模 |
| `d2q` | 时长感知分位数回归 |
| `cread` | 误差自适应离散化与数值恢复 |
| `d2co` | 时长相关成分修正 |
| `egmn` | 指数-高斯混合分布建模 |

## 默认配置

| 配置 | KuaiRec | WeChat21 |
|---|---:|---:|
| 时长区间数量 | 4 | 3 |
| 区间构造方式 | 等频 | 等频 |
| Batch size | 2048 | 2048 |
| 纠偏隐层维度 | 64 | 64 |
| 最大纠偏训练轮数 | 25 | 30 |
| 早停 patience | 6 | 6 |

两套数据共同使用以下默认设置：

- hard duration routing；
- 每个区间独立的可学习 Box-Cox 参数；
- 各区间均匀加权；
- 冻结第一阶段预测器；
- 七个观看时长辅助目标；
- 基于验证集 XAUC 早停；
- 损失权重 `(变换空间, 原始空间, 正则, 辅助任务) = (1.0, 0.8, 0.05, 0.10)`。

逆变换时会对变换空间预测和恢复后的纠偏因子进行数值裁剪。

## 环境安装

- Python 3.8+
- PyTorch 1.12+

```bash
pip install -r requirements.txt
```

## 数据准备

从官方渠道下载数据：

- [KuaiRec](https://kuairec.com/)
- [WeChatBigData Challenge 2021](https://algo.weixin.qq.com/)

按照 [`dataset/README.md`](dataset/README.md) 放置并预处理原始文件。原始数据和处理后的数据文件均不纳入版本控制。

## 运行 DADF

在 KuaiRec 上运行标准 WLR 配置：

```bash
DATASET=kuairec DEVICE=cuda:0 bash run_DADF_wlr.sh
```

运行 WeChat21 或同时运行两套数据：

```bash
DATASET=wechat21 DEVICE=cuda:0 bash run_DADF_wlr.sh
DATASET=all DEVICE=cuda:0 bash run_DADF_wlr.sh
```

直接训练入口支持所有第一阶段预测器：

```bash
python model/dadf/train.py --help
python model/dadf/train.py --base_model egmn --dataset_name kuairec --full-data --device cuda:0
```

## 评估与论文结果摘要

运行程序报告：

- **MAE**：越低越好；
- **XAUC**：严格样本对排序一致率，越高越好。

在 14 个 backbone-数据集组合中，相比对应的冻结 backbone，DADF 平均降低 MAE **4.33%**，平均提升 XAUC **4.01%**。每次运行的随机种子和指标会写入本地日志目录。

### EGMN 参考结果

原始 [EGMN 论文](https://arxiv.org/pdf/2508.12665) 的结果可作为本仓库 EGMN backbone 复现的参考：

| 来源 | KuaiRec MAE | KuaiRec XAUC | WeChat MAE | WeChat XAUC |
|---|---:|---:|---:|---:|
| 原始 EGMN | 4.204 | 0.6093 | 18.880 | 0.6692 |
| 本仓库 EGMN baseline | 4.081 | 0.6245 | 18.330 | 0.6896 |

本仓库复现的 EGMN 在两套数据上均达到有竞争力的工作点。由于不同仓库的数据预处理和评估划分可能存在差异，这组数值用于结果参考，不作为严格的横向对照。

WLR backbone 上包含三个核心组件的消融实验：

| 变体 | 移除内容 |
|---|---|
| `w/o Dist.` | 分组目标变换 |
| `w/o Factor` | 时长索引专家路由，改为共享纠偏映射 |
| `w/o Aux.` | 辅助行为表征 |

移除任一组件都会使两套公共数据上的 MAE/XAUC 下降。代码通过 `--shared_correction` 提供路由消融，通过 `--no_aux_targets` 提供辅助表征消融。

## 目录结构

```text
model/dadf/        DADF 网络、变换、损失、适配器与训练入口
model/             第一阶段预测器和共享网络层
dataloader/        数据加载器
dataset/           公共数据集预处理
tests/             论文方法契约测试
run_DADF_wlr.sh    WLR 实验入口
```

## 引用

```bibtex
@misc{yang2026dadf,
  title  = {DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems},
  author = {Yiqing Yang and Xinlong Zhao and Zhao Liu and Xiao Lv and Ruiming Tang and Kun Gai},
  year   = {2026}
}
```

## 许可证

MIT
