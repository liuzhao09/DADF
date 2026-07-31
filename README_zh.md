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

在 KuaiRec 上运行标准 WLR+DADF 配置：

```bash
BASE_MODEL=wlr MODE=dadf DATASET=kuairec DEVICE=cuda:0 bash run_DADF.sh
```

通过 `BASE_MODEL` 选择任意受支持的 backbone：

```bash
BASE_MODEL=egmn MODE=dadf DATASET=kuairec DEVICE=cuda:0 bash run_DADF.sh
BASE_MODEL=cread MODE=dadf DATASET=wechat21 DEVICE=cuda:0 bash run_DADF.sh
```

只训练 backbone，完全跳过 DADF 构建与训练：

```bash
BASE_MODEL=wlr MODE=base DATASET=kuairec DEVICE=cuda:0 bash run_DADF.sh
```

同时在后台启动全部七个 backbone：

```bash
bash run_all_backbone.sh
```

该入口固定使用 `MODE=base`，不会构建或训练 DADF；默认保持原始
`256 128 64` MLP 宽度，并在两张 GPU 上轮询分配。默认情况下，
`vr/tpm/cread/egmn` 使用 `cuda:0`，`wlr/d2q/d2co` 使用 `cuda:1`。
七个任务会立即并发启动，而不是排队执行。每个任务最多训练 100 个
epoch；若验证集 XAUC 连续 6 个 epoch 未提升，则提前停止并恢复最佳
checkpoint。

日志和 PID 分别保存在：

```text
logs/all_backbones_<时间戳>/base_earlystop_<backbone>.log
logs/all_backbones_<时间戳>/backbone_pids.txt
```

无需修改脚本即可覆盖 GPU、最大训练轮数和 patience：

```bash
DEVICES="cuda:0 cuda:1" BASE_EPOCH=100 PATIENCE=6 \
  bash run_all_backbone.sh
```

同时启动七个参数量匹配的纯 backbone 消融实验：

```bash
CAPACITY_MATCHED=1 bash run_all_backbone.sh
```

该模式仍然完全跳过 DADF，但会为每个模型自动选择匹配
Backbone+DADF dense 参数量的 MLP：VR、WLR、D2CO、EGMN 使用
`354 128 64`，TPM、D2Q、CREAD 使用 `342 128 64`。日志独立保存在
`logs/all_backbones_capacity_matched_<时间戳>/capacity_matched_<backbone>.log`。

查看最新一次并发实验：

```bash
RUN_DIR=$(ls -dt logs/all_backbones_* | head -1)
cat "${RUN_DIR}/backbone_pids.txt"
tail -f "${RUN_DIR}/base_earlystop_wlr.log"
watch -n 2 nvidia-smi
```

在全量数据模式下，每个进程都会在 CPU 内存中持有一份处理后的数据。
使用并发入口前应确认主机内存充足；内存不足时应改用 `run_DADF.sh`
逐个运行。

通过空格分隔的维度列表扩展 backbone MLP：

```bash
BASE_MODEL=wlr MODE=base BASE_MLP_DIMS="354 128 64" \
  DATASET=kuairec DEVICE=cuda:0 bash run_DADF.sh
```

直接训练入口提供相同控制：

```bash
python model/dadf/train.py --help
python model/dadf/train.py --base_model egmn --base_mlp_dims 256 128 64 \
  --dataset_name kuairec --full-data --device cuda:0
python model/dadf/train.py --base_model egmn --base_only --base_epoch 30 \
  --base_mlp_dims 354 128 64 --dataset_name kuairec --full-data --device cuda:0
```

### Dense 参数量对照

训练入口会输出去重后的总参数量与 dense 参数量，其中 dense 参数不包含
embedding table。在 KuaiRec 默认特征和 DADF 配置下，仅扩展 backbone 第一层
即可在 0.1% 误差内匹配对应 Backbone+DADF 的 dense 参数量：

| Backbone | Backbone dense | Backbone+DADF dense | 匹配的 `BASE_MLP_DIMS` | 匹配后 dense |
|---|---:|---:|---|---:|
| VR | 185,730 | 253,649 | `354 128 64` | 253,448 |
| WLR | 185,730 | 253,649 | `354 128 64` | 253,448 |
| TPM | 187,936 | 247,663 | `342 128 64` | 247,448 |
| D2Q | 185,986 | 245,713 | `342 128 64` | 245,498 |
| CREAD | 294,771 | 354,498 | `342 128 64` | 354,283 |
| D2CO | 185,730 | 253,649 | `354 128 64` | 253,448 |
| EGMN | 188,001 | 255,920 | `354 128 64` | 255,817 |

若预处理删除了常量特征，应以实际运行日志打印的参数量为准。容量对照实验应
使用相同的数据划分、随机种子、训练预算和验证集选择协议。

#### 为什么需要控制 backbone 参数量？

DADF 的纠偏模块会引入额外的 dense 参数，因此需要排除效果仅来自模型容量
增加的可能性。为此，我们完全移除 DADF，仅扩展 backbone MLP，使其 dense
参数量与对应的 Backbone+DADF 在 0.1% 误差内匹配。

更重要的是，默认 backbone、参数量匹配 backbone 与 DADF 使用完全相同的
预处理样本、训练/验证/测试划分以及原始输入特征字段。DADF 不引入外部数据集
或额外的线上特征字段；其辅助目标均由同一批训练样本中已有的标签确定性构造。
因此，该实验同时控制了信息来源与模型容量：参数量匹配 backbone 与 DADF
使用相同的数据和特征，并将 dense 参数量控制在 0.1% 误差内。两者剩余的
核心差异是 DADF 如何通过分布感知结构和训练目标组织、监督这些参数。

对于每个随机种子，使用相同的数据划分、优化预算和基于 validation XAUC 的
checkpoint 选择协议，分别运行默认容量与参数量匹配版本：

```bash
# 默认容量 backbone
SEED=42 CAPACITY_MATCHED=0 DEVICES="cuda:0 cuda:1" \
  BASE_EPOCH=100 PATIENCE=6 bash run_all_backbone.sh

# 参数量匹配 backbone；应在上面一组任务全部结束后启动
SEED=42 CAPACITY_MATCHED=1 DEVICES="cuda:0 cuda:1" \
  BASE_EPOCH=100 PATIENCE=6 bash run_all_backbone.sh
```

我们使用 10 个随机种子重复了上述成对实验。由于 `run_all_backbone.sh`
会启动七个后台任务后立即返回，因此每组任务全部结束后才启动下一个 seed。

#### 10-seed 参数量对照结果

下表报告 10 个随机种子的测试集 XAUC 均值；每次运行均恢复 validation XAUC
最优的 checkpoint 后进行测试。`Delta XAUC` 定义为“参数量匹配版减去默认
容量版”，因此正值表示扩充参数后更好。

| Backbone | 默认 Dense 参数 | 匹配 Dense 参数 | 参数增幅 | 默认 XAUC | 匹配 XAUC | Delta XAUC |
|---|---:|---:|---:|---:|---:|---:|
| VR | 185,730 | 253,448 | +36.46% | **0.5612** | 0.5583 | -0.0029 |
| WLR | 185,730 | 253,448 | +36.46% | **0.5947** | 0.5807 | -0.0140 |
| TPM | 187,936 | 247,448 | +31.67% | 0.5517 | **0.5534** | +0.0017 |
| D2Q | 185,986 | 245,498 | +32.00% | **0.6317** | 0.6316 | -0.0001 |
| CREAD | 294,771 | 354,283 | +20.19% | 0.5954 | **0.6013** | +0.0059 |
| D2CO | 185,730 | 253,448 | +36.46% | **0.5708** | 0.5706 | -0.0002 |
| EGMN | 188,001 | 255,817 | +36.07% | 0.6263 | **0.6268** | +0.0005 |
| **Average** | **201,983** | **266,199** | **+31.79%** | **0.5903** | **0.5890** | **-0.0013** |

平均增加 31.79% 的 dense 参数并未带来稳定收益：平均 XAUC 下降 0.0013，
且七个 backbone 中有四个没有提升。参数量匹配 backbone 与 DADF 使用相同
的训练样本和原始特征，并具有近乎一致的 dense 参数量，但仅将这些参数加入
backbone 并不能复现 DADF 的增益。因此，DADF 的稳定增益不能归因于额外
数据、额外特征或参数量本身，而是来自其对这些参数进行分布感知组织与训练的
方式。

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
run_DADF.sh        通用 backbone 与 DADF 实验入口
run_all_backbone.sh 七个纯 backbone 并发训练入口
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
