# DADF：面向观看时长回归的分布感知纠偏框架

论文 **DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems** 的参考实现。

## 方法

DADF 在冻结的第一阶段观看时长预测器上进行二阶段乘性修正：

\[
\hat{Y}=\hat{Y}_0\hat{B}.
\]

代码实现与论文一致的三个组件：

1. **分组目标变换**：对纠偏目标应用可学习的组级 Box–Cox 变换。
2. **时长索引专家路由**：通过确定性 hard routing 为每个样本选择一个纠偏专家。
3. **辅助行为表征**：将辅助任务 logit、tower 表征与共享纠偏上下文融合。

训练目标由变换空间 MSE、原始空间 Huber、变换矩正则和辅助 BCE 组成。辅助标签仅用于训练监督，推理只使用模型输出和推理时可用特征。

## 环境

- Python 3.8+
- PyTorch 1.12+

```bash
pip install -r requirements.txt
```

## 数据

从官方渠道下载 KuaiRec 或 WeChat21，并按照 [`dataset/README.md`](dataset/README.md) 准备数据。原始数据和处理后的数据文件不纳入版本控制。

## 运行

WLR 标准实验：

```bash
DATASET=kuairec DEVICE=cuda:0 bash run_DADF_wlr.sh
```

将 `DATASET` 设置为 `wechat21` 可运行 WeChat21，设置为 `all` 可运行两套数据。直接入口为：

```bash
python model/dadf/train.py --help
```

与论文一致的默认设置包括：纠偏训练时冻结第一阶段预测器、等频时长分组、组级变换参数、hard routing、辅助行为表征，以及基于验证集 XAUC 的早停。

## 支持的第一阶段预测器

`vr`、`wlr`、`tpm`、`d2q`、`cread`、`d2co` 和 `egmn`。

## 评估

运行程序按照论文定义报告 MAE 和 XAUC，并在本地日志目录记录随机种子与每次运行的指标。

## 引用

```bibtex
@misc{yang2026dadf,
  title  = {DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems},
  author = {Yiqing Yang and Xinlong Zhao and Zhao Liu and Xiao Lv and Ruiming Tang},
  year   = {2026},
  note   = {Manuscript}
}
```

## 许可证

MIT
