# AI牙冠生成项目：当前数据的训练效果验证说明

本 README 用于指导当前目录下数据的第一轮实验验证。它不是正式论文阶段的完整 SOP，而是基于 `AI牙冠生成研究实验方案.pdf` 和现有 7 例病例数据，先完成一次“小样本可行性验证”，确认数据链路、训练链路和基础指标是否跑通。

当前仓库已经补入一套可直接运行的轻量验证代码，代码目录如下：

```text
crown_validate/
configs/default_validation.json
configs/smoke_test.json
```

说明：

1. 这套代码是“小样本验证框架”，重点是先跑通预处理、交叉验证训练、指标统计和结果落盘。
2. 它不是论文方案中完整的 DMC 原始实现，也没有包含 CNC 物理验证和专家盲评模块。
3. 对当前这 7 例数据，已经可以直接执行 end-to-end 的训练验证。

## 1. 当前目标

根据方案文档，项目的正式目标是基于 DMC 架构完成牙冠生成模型优化，并通过虚拟评价、物理验证和专家盲评形成完整研究闭环。

但目前本目录下只有少量病例数据，因此当前阶段建议先完成以下 3 个目标：

1. 验证数据是否能够稳定组成 `预备体/工作模 + 对颌牙 + 真值牙冠` 的训练样本。
2. 验证预处理、训练、推理、评价流程是否能完整跑通。
3. 先用小样本比较“基线 DMC”和“改进损失”的趋势，判断是否值得进入下一阶段的大样本整理与正式训练。

## 2. 当前数据概况

当前目录中的病例数据位于 `患者数据/病例/`，共 7 例。

| 病例 | 预备体或工作模 | 对颌/对合 | 真值牙冠 |
|---|---|---|---|
| 赵威 | `赵威.stl` | `赵威对颌牙.stl` | `赵威33992-36-crown_cad.stl` |
| 徐红 | `徐红.stl` | `徐红对颌牙.stl` | `徐红71319-36-coping_cad.stl` |
| 刘新华 | `刘新华工作模.stl` | `刘新华对合7.stl` | `刘新华牙冠7.stl` |
| 王 | `王15039.stl` | `王对合dentalCAD.stl` | `王牙冠.stl` |
| 徐梦恬 | `徐梦恬34063.stl` | `徐梦恬对合.stl` | `徐梦恬牙冠.stl` |
| 周翠霞 | `周翠霞工作模.stl` | `周翠霞对合1.stl` | `周翠霞牙冠1.stl` |
| 吴丰荷 | `009_李灿政_吴丰荷-LowerJaw.stl` | `009_李灿政_吴丰荷-UpperJaw.stl` | `34803吴丰荷W-45-crown_cad.stl` |

说明：

1. `吴丰荷` 这一例需要人工再次确认 `LowerJaw` 和 `UpperJaw` 中哪一个是预备体所在颌、哪一个是对颌。
2. 正式方案要求最终有效样本 `n>=100`，当前 7 例只适合做流程验证、过拟合检查和初步趋势判断，不适合做论文级统计推断。

## 3. 建议的验证路径

当前最稳妥的做法不是直接追求“最终论文结果”，而是分两步走：

### 第一步：完成小样本可行性验证

验证以下事项：

1. STL 文件是否都能正常读取、显示和配准。
2. 每例是否都能明确区分输入和真值。
3. 模型能否在极小数据集上收敛。
4. 改进损失在测试病例上是否比基线更稳定，至少不能明显退化。

### 第二步：再扩展到正式研究数据

当第一步跑通后，再按 PDF 方案补齐：

1. 扩展到 100+ 有效病例。
2. 增加边缘线标注。
3. 做区域特异性评价。
4. 做物理验证和专家盲评。

## 4. 实验前的数据整理

在开始训练前，先把 7 个病例统一整理成一致的命名和结构。建议整理为：

```text
患者数据/
  病例标准化/
    case_001/
      prep.stl
      opposing.stl
      crown_gt.stl
      meta.json
    case_002/
      prep.stl
      opposing.stl
      crown_gt.stl
      meta.json
```

`meta.json` 建议至少记录以下字段：

```json
{
  "case_id": "case_001",
  "patient_name": "赵威",
  "tooth_position": "36",
  "prep_file": "prep.stl",
  "opposing_file": "opposing.stl",
  "crown_gt_file": "crown_gt.stl",
  "needs_manual_confirmation": false,
  "notes": ""
}
```

如果暂时不想移动原始文件，也至少先做一个清单表，例如 `cases_manifest.csv`，用于明确每一例的对应关系。

## 4.1 直接可运行的命令

建议使用已有的 Conda 环境 `ensemble` 运行：

### 方式 A：一键跑完整小样本验证

```bash
conda run -n ensemble python -m crown_validate.cli full-pipeline \
  --raw-dir '患者数据/病例' \
  --work-dir runs/full_validation \
  --config configs/default_validation.json \
  --variants baseline,improved \
  --include-unconfirmed
```

输出内容会写入：

```text
runs/full_validation/
  cases_manifest.csv
  processed_cases/
  experiments/
```

### 方式 B：分步骤执行

1. 生成病例清单

```bash
conda run -n ensemble python -m crown_validate.cli build-manifest \
  --raw-dir '患者数据/病例' \
  --output runs/manual/cases_manifest.csv
```

2. 预处理 STL

```bash
conda run -n ensemble python -m crown_validate.cli preprocess \
  --manifest runs/manual/cases_manifest.csv \
  --output-dir runs/manual/processed_cases \
  --config configs/default_validation.json \
  --include-unconfirmed
```

3. 跑交叉验证

```bash
conda run -n ensemble python -m crown_validate.cli run-cv \
  --manifest runs/manual/cases_manifest.csv \
  --processed-dir runs/manual/processed_cases \
  --output-dir runs/manual/experiments \
  --config configs/default_validation.json \
  --variants baseline,improved \
  --include-unconfirmed
```

### 方式 C：先做 smoke test

如果你只想先确认链路能不能跑通：

```bash
conda run -n ensemble python -m crown_validate.cli full-pipeline \
  --raw-dir '患者数据/病例' \
  --work-dir runs/smoke_test \
  --config configs/smoke_test.json \
  --variants baseline,improved \
  --include-unconfirmed
```

## 5. 预处理步骤

按照 PDF 方案，正式预处理包括分割、边缘线标注、下采样和归一化。考虑到目前是小样本验证，建议先做“轻量版预处理”。

### 5.1 文件质量检查

逐例在 MeshLab、CloudCompare 或 Open3D 中检查：

1. STL 是否能正常打开。
2. 网格是否闭合，是否存在明显破洞、翻面、重复面。
3. 牙冠真值是否与预备体大致对应。
4. 对颌牙是否方向正确。

如果某例存在明显破损、坐标尺度异常或方向颠倒，先剔除，不要直接送入训练。

### 5.2 统一坐标和尺度

对每个样本执行：

1. 读入 `prep.stl`、`opposing.stl`、`crown_gt.stl`。
2. 确认单位统一为毫米。
3. 以预备体为参考进行居中。
4. 对输入和真值应用同一套平移/缩放参数。
5. 保存标准化后的中间结果。

建议输出：

```text
data_processed/
  case_001/
    prep_aligned.stl
    opposing_aligned.stl
    crown_gt_aligned.stl
```

### 5.3 点云采样

可先按 PDF 方案的主设置执行：

1. 输入上下文点云采样到 `10,240` 点。
2. 真值牙冠点云采样到 `8,192` 点。
3. 保存为 `npy`、`npz` 或 `pt` 格式，方便训练直接读取。

如果现阶段显存或代码不稳定，可先降为：

1. 输入 `4,096` 或 `8,192` 点。
2. 输出 `2,048` 或 `4,096` 点。

先保证实验跑通，再恢复到正式设置。

### 5.4 边缘线处理

PDF 中的改进损失依赖边缘线信息。当前阶段建议分两种情况：

1. 如果还没有边缘线标注，先跑不依赖边缘线的基线实验。
2. 如果你们能先手工标 3 到 7 例的边缘线，则可以同步验证 `MW-CD` 是否有帮助。

当前阶段不建议一开始就做复杂自动分割，先用人工少量标注确认方法有效性。

## 6. 当前阶段推荐的数据划分

因为目前只有 7 例，不能按正式方案的 `70/15/15` 生硬切分。建议使用以下两种方式中的一种。

### 方案 A：7 折留一验证（推荐）

每一轮：

1. 5 例作为训练集。
2. 1 例作为验证集。
3. 1 例作为测试集。

轮换 7 次，使每个病例都至少做 1 次测试病例。最终报告 7 轮均值和标准差。

### 方案 B：固定划分

适合快速试跑：

1. 训练集：5 例
2. 验证集：1 例
3. 测试集：1 例

如果只是想先确认代码能不能跑，可以先用固定划分；如果要汇报初步结果，建议至少做方案 A。

## 7. 推荐的训练对比实验

根据 PDF，正式消融实验很多。当前数据量太小，第一轮只建议先做下面 3 组。

### 实验 E0：基线复现

目标：

1. 验证 DMC 训练流程能跑通。
2. 得到最基本的对照结果。

配置：

1. 损失函数：`CD + MSE_indicator`
2. 不使用边缘线权重
3. 不使用课程学习

### 实验 E1：加入边缘加权

前提：已经有人手工标出边缘线。

配置：

1. 损失函数：`MW-CD + MSE_indicator`
2. 先不加曲率损失
3. 用于观察边缘区域是否有改善趋势

### 实验 E2：轻量完整版

配置：

1. 损失函数：`MW-CD + CPL + MSE_indicator + L_normal`
2. 可以先不开课程学习，避免小样本时训练不稳定
3. 若 E2 比 E0 持续更差，优先检查预处理和边缘线质量，而不是直接否定方法

## 8. 训练参数建议

PDF 中正式配置为 `AdamW / lr=5e-4 / batch=16 / 400 epochs / A100`。  
考虑当前是 7 例小样本验证，建议先用更保守的设置：

1. 优化器：`AdamW`
2. 学习率：`5e-4`
3. 批大小：`1-4`
4. 训练轮数：`150-300 epochs`
5. 早停：`patience=20-30`
6. 保存最佳验证集 checkpoint

如果发现训练集 loss 持续下降而测试集指标明显波动，说明已经出现过拟合，这在当前样本量下是正常现象。

## 9. 训练效果怎么判断

当前阶段不看“绝对论文结论”，重点看以下 4 件事。

### 9.1 是否收敛

至少需要满足：

1. 训练 loss 明显下降。
2. 验证 loss 不发散。
3. 推理能够稳定生成牙冠网格，而不是塌陷点云或异常网格。

### 9.2 是否优于基线

优先比较以下指标：

1. 全局 `Chamfer Distance`
2. `HD95`
3. `F-score@0.3mm`
4. 若已有边缘线，则比较边缘区误差

判断标准：

1. 改进模型在大多数折次中优于或不差于基线。
2. 改进模型在测试病例上没有明显更多的失败样本。

### 9.3 是否存在明显过拟合

如果出现下面现象，需要先停下来排查：

1. 训练集很好，验证集和测试集很差。
2. 个别病例特别好，换一个测试病例就明显失效。
3. 生成牙冠形态看似光滑，但咬合关系明显不合理。

### 9.4 是否值得进入正式数据阶段

若满足以下条件，就可以进入下一步的大样本整理：

1. 数据链路稳定，无明显配对错误。
2. 基线模型能稳定跑通。
3. 改进损失至少在部分指标上显示出正向趋势。
4. 结果热力图符合临床直觉，边缘区没有系统性恶化。

## 10. 当前阶段最少要出的结果

第一次验证结束后，建议至少整理出以下内容：

1. 每一折的训练日志。
2. 每一折测试病例的预测 STL。
3. 基线与改进模型的指标对比表。
4. 典型病例的三维偏差热力图。
5. 一页结论摘要，回答“数据能不能用、流程通不通、改进是否有趋势”。

建议输出目录：

```text
outputs/
  fold_01/
    checkpoint_best.pt
    pred_crown.stl
    metrics.json
    vis/
  fold_02/
  summary/
    cv_results.csv
    comparison.xlsx
    qualitative_cases.md
```

## 11. 当前阶段不建议立即做的事情

基于目前目录内容，以下工作建议暂时不要作为第一优先级：

1. 不要直接做论文中的完整消融矩阵 `A0-A5、B1-B2、C1-C2、D1-D4`。
2. 不要直接做 `TOST` 等效性检验，因为样本量远远不够。
3. 不要直接做物理冠加工验证，因为目前重点是先证明数字流程能跑通。
4. 不要一开始就追求非常复杂的自动区域分割，先把基础训练和虚拟评价跑起来。

## 12. 建议的实际执行顺序

建议按下面顺序推进：

1. 先确认 7 例数据的输入/真值对应关系，尤其是 `吴丰荷` 这一例。
2. 完成病例清单表和标准化命名。
3. 跑通 STL 读取、配准、采样和归一化。
4. 先训练基线模型 `E0`。
5. 如果边缘线已准备好，再训练 `E1/E2`。
6. 做 7 折或至少多轮交叉验证。
7. 汇总指标、热力图和失败案例。
8. 决定是否扩充病例并进入正式研究阶段。

## 13. 一句话结论

当前这批数据最适合做“训练流程和方法方向是否成立”的预验证，不适合直接作为正式研究结论。只要先把基线模型跑通，并看到改进损失在小样本测试中有稳定的正向趋势，这一轮实验就是成功的。

## 14. 当前运行实验的模型说明

当前仓库中实际运行的模型不是论文方案里的完整 DMC 原始实现，而是一个为了“先完成小样本验证”而实现的轻量点云牙冠生成模型，模型代码在 `crown_validate/model.py`。

模型名称可以理解为：

1. `CrownDeformationNet`
2. 输入为 `prep_points` 和 `opposing_points`
3. 输出为预测牙冠点集 `pred_points`、法向量 `pred_normals` 和边缘点 `pred_margin`

模型结构如下：

1. `PointEncoder`
   对预备体点云和对颌点云分别做 MLP 编码，再通过 max-pooling 得到全局特征。
2. 特征融合
   将 `prep` 特征、`opposing` 特征、二者差值和逐元素乘积拼接后再映射到统一潜变量。
3. 模板形变解码
   以一个预先构造的开放冠体模板网格为基础，让网络预测每个模板顶点的三维偏移量，从而生成冠体表面。
4. 法向量重建
   根据模板三角面重新计算预测网格的顶点法向量。

当前实验实际上跑了两个版本：

### baseline

使用基础损失：

1. `Chamfer Distance`
2. 不加边缘加权
3. 不加曲率项
4. 不加法向量一致性项

### improved

使用改进损失：

1. 边缘加权 `Chamfer Distance`
2. 曲率近似惩罚项
3. 法向量一致性项

因此，README、配置文件和结果表中的 `baseline` 与 `improved`，本质上表示：

1. 同一个轻量冠体生成网络
2. 使用不同的损失函数配置进行训练和对比

如果后续你们要完全对齐 PDF 中的正式研究方案，可以再把这里的轻量模型替换为完整 DMC 复现版，但当前这套实现已经足够完成预处理、训练、交叉验证和预测 STL 导出。

## 15. 最终实验结果文件里有什么

正式实验输出目录以 `runs/formal_validation/` 为例，主要包含以下内容：

```text
runs/formal_validation/
  cases_manifest.csv
  processed_cases/
  experiments/
    variant_comparison.csv
    baseline/
    improved/
```

### 15.1 病例与预处理文件

`cases_manifest.csv`

作用：

1. 记录每个病例的编号
2. 记录原始 STL 路径
3. 指定 `prep / opposing / crown_gt` 的对应关系
4. 标记是否需要人工复核

`processed_cases/case_xxx/processed_case.npz`

内容：

1. `prep_points`
2. `opposing_points`
3. `crown_points`
4. `crown_normals`
5. `margin_points`
6. `center`
7. `scale`

这些是训练真正读取的数据，已经完成采样和归一化。

`processed_cases/case_xxx/metadata.json`

内容：

1. 病例编号和患者名
2. 原始 STL 路径
3. 备注说明
4. `margin_source`

其中 `margin_source` 用来说明边缘区是怎么得到的。

### 15.2 总体结果文件

`experiments/variant_comparison.csv`

这是最先应该看的总表，内容是不同实验版本的总体平均指标，例如：

1. `mean_chamfer_l2_mm2`
2. `mean_hd95_mm`
3. `mean_fscore`
4. `mean_margin_chamfer_l2_mm2`

用它可以快速比较 `baseline` 和 `improved` 谁更好。

`experiments/baseline/aggregate_metrics.json`

`experiments/improved/aggregate_metrics.json`

这两个文件保存每个版本在 7 折交叉验证上的汇总统计，通常包含：

1. 每个指标的 `mean`
2. 每个指标的 `std`

例如：

1. `chamfer_l2_mm2`
2. `hd95_mm`
3. `fscore`
4. `margin_chamfer_l2_mm2`
5. `margin_hd95_mm`
6. `margin_mean_mm`
7. `best_val_loss`

### 15.3 每折结果文件

每个版本下都有：

```text
experiments/baseline/fold_01/
experiments/baseline/fold_02/
...
experiments/improved/fold_01/
```

每一折目录中主要文件如下：

`checkpoint_best.pt`

作用：

1. 当前折验证集表现最好的模型参数
2. 后续如需复现该折推理，可直接加载这个文件

`history.csv`

内容：

1. `epoch`
2. `train_loss`
3. `val_loss`

作用：

1. 查看训练是否收敛
2. 观察是否过拟合

`summary.json`

内容：

1. 当前折测试病例的主要指标
2. 最优验证损失 `best_val_loss`

适合快速看某一折的结论。

`val_metrics.csv`

`test_metrics.csv`

内容通常包括：

1. `case_id`
2. `mean_pred_to_gt_mm`
3. `mean_gt_to_pred_mm`
4. `chamfer_l2_mm2`
5. `hd95_mm`
6. `fscore`
7. `margin_chamfer_l2_mm2`
8. `margin_hd95_mm`
9. `margin_mean_mm`
10. `loss_total`
11. `loss_chamfer`
12. `loss_curvature`
13. `loss_normal`

其中：

1. `val_metrics.csv` 是验证病例的结果
2. `test_metrics.csv` 是测试病例的结果

### 15.4 生成出来的 STL 在哪里

每一折都会自动导出验证病例和测试病例的预测 STL。

测试病例 STL 路径格式为：

```text
experiments/<variant>/fold_xx/test_predictions/case_xxx/pred_crown.stl
```

例如：

```text
runs/formal_validation/experiments/improved/fold_01/test_predictions/case_001/pred_crown.stl
```

同目录下还有：

1. `pred_points_mm.npy`
   预测牙冠表面点，单位为毫米坐标
2. `pred_margin_mm.npy`
   预测边缘区点，单位为毫米坐标

验证病例的预测结果则在：

```text
experiments/<variant>/fold_xx/val_predictions/case_xxx/
```

### 15.5 应该先看哪些文件

如果只想快速理解最终实验结果，建议按这个顺序看：

1. `experiments/variant_comparison.csv`
2. `experiments/improved/aggregate_metrics.json`
3. `experiments/improved/cv_results.csv`
4. 某一折的 `history.csv`
5. 某一折测试病例的 `pred_crown.stl`

这样就能同时看到：

1. 总体对比结论
2. 各折是否稳定
3. 训练是否正常收敛
4. 生成出来的牙冠几何形态长什么样

## 16. 运行完整实验的命令

如果需要从原始病例数据开始，完整执行一轮正式实验，推荐直接运行下面的命令：

```bash
conda run -n ensemble python -m crown_validate.cli full-pipeline \
  --raw-dir '患者数据/病例' \
  --work-dir runs/formal_validation \
  --config configs/default_validation.json \
  --variants baseline,improved \
  --include-unconfirmed
```

该命令会自动完成：

1. 生成 `cases_manifest.csv`
2. 预处理 STL 并写入 `processed_cases/`
3. 运行 `baseline` 和 `improved` 两组 7 折交叉验证
4. 导出每折验证病例和测试病例的预测 STL
5. 汇总总体指标到 `variant_comparison.csv` 和 `aggregate_metrics.json`

如果只想单独跑某个步骤，也可以按下面分步执行。

### 16.1 只生成病例清单

```bash
conda run -n ensemble python -m crown_validate.cli build-manifest \
  --raw-dir '患者数据/病例' \
  --output runs/manual/cases_manifest.csv
```

### 16.2 只做预处理

```bash
conda run -n ensemble python -m crown_validate.cli preprocess \
  --manifest runs/manual/cases_manifest.csv \
  --output-dir runs/manual/processed_cases \
  --config configs/default_validation.json \
  --include-unconfirmed
```

### 16.3 只跑交叉验证训练

```bash
conda run -n ensemble python -m crown_validate.cli run-cv \
  --manifest runs/manual/cases_manifest.csv \
  --processed-dir runs/manual/processed_cases \
  --output-dir runs/manual/experiments \
  --config configs/default_validation.json \
  --variants baseline,improved \
  --include-unconfirmed
```

### 16.4 只跑改进版

如果只想跑 `improved`，可以把 `--variants` 改成：

```bash
conda run -n ensemble python -m crown_validate.cli full-pipeline \
  --raw-dir '患者数据/病例' \
  --work-dir runs/formal_validation_improved_only \
  --config configs/default_validation.json \
  --variants improved \
  --include-unconfirmed
```

### 16.5 单病例过拟合

```bash
conda run -n ensemble python -m crown_validate.cli overfit-one \
  --raw-dir '患者数据/病例' \
  --work-dir runs/overfit_case001 \
  --case-id case_001 \
  --variant improved \
  --config configs/default_validation.json \
  --include-unconfirmed
```


## 17. 当前效果不够像牙时的提升优先级

如果发现 `improved` 的实验指标已经优于 `baseline`，但导出的 `pred_crown.stl` 形态仍然“不像真实牙冠”，建议按下面的优先级逐步改进。

### 17.1 先做单病例过拟合测试

第一步不要急着继续加数据或改很多损失，先验证当前模型本身是否有足够表达能力。

建议做法：

1. 只选 1 个病例训练
2. 训练更多轮次
3. 检查模型能否几乎复现该病例的真值牙冠

判断意义：

1. 如果单病例都学不像，说明问题更可能在模型结构、模板表达能力或预处理
2. 如果单病例能学得很像，说明模型有基本能力，当前主要瓶颈是数据量和条件信息不足

### 17.2 升级模型，而不是只继续调损失函数

当前实现是一个轻量模板形变模型，虽然适合做小样本验证，但表达能力有限。

优先升级方向：

1. 复现更接近 DMC 的条件生成结构
2. 使用更强的 point cloud encoder-decoder
3. 尝试加入 Transformer 几何建模模块

核心原因：

1. 牙冠的牙尖、沟裂、邻接面和轴面外形都很复杂
2. 单纯依赖固定模板加顶点偏移，容易生成“像壳、不像牙”的结果

### 17.3 把输入做成真正的牙冠设计上下文

现在的输入主要是 `prep` 和 `opposing`，条件还不够完整。

后续建议补充：

1. 邻牙信息
2. 更准确的边缘线标注
3. 插入方向或局部坐标系
4. 更明确的局部牙位上下文

这样做的目的，是让模型不仅学会“盖住预备体”，还学会：

1. 邻接关系
2. 咬合关系
3. 自然的轴面外形
4. 更真实的牙冠解剖特征

### 17.4 优先改进初始模板

如果暂时还不打算换成完整 DMC，可以先优化模板本身。

建议方向：

1. 提高模板分辨率
2. 增加模板顶点数
3. 使用更接近真实牙冠的初始模板
4. 针对前牙、前磨牙、磨牙分别设计不同模板

这样可以明显缓解：

1. 结果过于圆滑
2. 像帽子或壳体
3. 咬合面细节不足

### 17.5 增加更贴近牙冠任务的临床约束

当前 `improved` 已经加入了边缘加权、曲率项和法向量一致性，但还可以继续增强任务相关约束。

建议后续增加：

1. 与预备体的间隙约束
2. 与对颌牙的咬合接触约束
3. 邻接区接触约束
4. 更合理的表面平滑正则

这类约束的目标不是只让误差变小，而是让生成结果更符合真实修复体设计逻辑。

### 17.6 扩大数据量

这是最终一定要做的事情。

当前 7 例数据足够验证：

1. 流程能不能跑通
2. `improved` 是否优于 `baseline`
3. 哪些模块值得继续做

但远远不足以支撑高质量牙冠形态学习。

建议目标：

1. 短期先扩到 30 例以上
2. 中期扩到 100 例左右
3. 再进入更正式的统计分析和论文结论阶段

## 18. 当前阶段最推荐的三件事

如果现在只优先做最有价值的 3 件事，建议顺序如下：

1. 做单病例过拟合测试，确认当前模型到底有没有学会能力
2. 升级到更接近 DMC 的模型结构，而不是只继续调 loss
3. 补充邻牙、边缘线和更完整的上下文输入

一句话总结：

当前 `improved` 方向是对的，但“看起来不像牙”主要不是因为改进损失无效，而是因为当前仍受限于轻量模型、数据量太少和输入条件不完整。
