# AI 后牙全冠生成方案

本项目面向真实临床 CAD/CAM 后牙单冠病例，目标是建立一套 AI 后牙全冠外表面生成、临床风险分区评价与失败风险提示流程。模型主要生成全解剖外冠面（full-contour external crown）；实体冠加工阶段由 CAD 软件根据预备体、margin line 和统一 cement gap/spacer 参数生成内表面（intaglio），再与 AI 外冠面合并。

## 研究目标

1. 构建真实后牙 CAD/CAM 配对数据集，包含患者上下颌模型、预备体、技师最终设计冠、margin line 和牙位信息。
2. 在生成模型中引入 margin-line-anchored ring-wise 生成机制，使模型从边缘线出发沿冠方逐层生成外冠面。
3. 建立 R1-R5 后牙全冠临床风险分区评价体系。
4. 构建 Clinical Risk-weighted Crown Score（CRCS），用于综合评价 AI 冠的临床风险。
5. 使用咬合接触图和邻接接触图评价 AI 冠与技师冠的功能接触差异。
6. 通过几何误差、区域化评价、专家盲评和小样本实体冠验证，评估 AI 生成冠的临床可用性。

## 数据范围

纳入后牙单颗天然牙全冠病例，牙位包括：

`14, 15, 16, 17, 24, 25, 26, 27, 34, 35, 36, 37, 44, 45, 46, 47`

每例病例至少包含：

- 患者上颌模型
- 患者下颌模型
- 预备体所在工作模
- 对合牙模型
- 技师最终设计冠
- margin line 坐标
- 牙位号
- 病例编号
- 上下颌信息

排除前牙、第三磨牙、种植体支持冠、固定桥、联冠、多单位修复、margin line 明显错误、上下颌或技师冠坐标错位、严重扫描质量问题、邻牙或对合牙无法评价的病例。

## 推荐样本量

目标样本量为 150-200 例后牙单冠病例。

推荐划分：

- 训练集：70%，约 105-140 例
- 验证集：15%，约 22-30 例
- 测试集：15%，约 22-30 例

数据划分按患者层面执行，避免同一患者的多颗牙冠跨训练集和测试集造成信息泄漏。同时按前磨牙/磨牙、上颌/下颌进行分层，保证牙位分布尽量均衡。

主要结局指标建议设为：完整模型 M3 相较 DMC baseline M0 的 R1 边缘区 RMS 误差。

## 数据目录结构

每例病例按三级结构整理：

```text
case_id_姓名_牙位/
  00_raw_original/
  processed/
    upper_local.stl
    lower_local.stl
    crown_{tooth_id}.stl
    margin_local.xyz
    margin_local.ply
    constructionInfo
  train/
    prep_points.npy
    antagonist_points.npy
    crown_points.npy
    margin_points.npy
    metadata.json
    bbox_scale.json
    roi20_fps_preview.ply
```

若同一患者贡献多颗牙冠，可按患者为主文件夹、牙位为子文件夹整理。

角色命名规则：

- `prep_points.npy`：目标预备体所在颌的局部点云，不固定为 upper 或 lower。
- `antagonist_points.npy`：对颌局部点云。
- `crown_points.npy`：技师最终设计冠外表面点云。
- `margin_points.npy`：CAD 导出的有序闭合 margin line。
- 预备体所在颌由 margin line 到 `upper_local.stl` 和 `lower_local.stl` 的平均最近距离自动判定。
- FDI 牙位预期颌位只作为复核字段，不作为唯一判定依据。

## 训练输入

当前训练包建议使用：

- `prep_points.npy`: `8192 x 6`，坐标和法向量
- `antagonist_points.npy`: `8192 x 6`，坐标和法向量
- `crown_points.npy`: `16384 x 6`，技师冠外表面监督标签
- `margin_points.npy`: `1024 x 3`，有序闭合边缘线
- `metadata.json`: 病例、牙位、颌位、QC 和文件语义信息

所有点云坐标保留真实毫米单位，训练时只做局部平移，不做尺度缩放。

局部坐标定义：

1. 以 `processed/margin_local.xyz` 的中心点作为 ROI 中心和局部原点。
2. 裁取 `20 mm x 20 mm x 20 mm` 固定物理 ROI。
3. 对 prep、antagonist、crown 和 margin 统一减去该中心点。
4. `scale_applied=False`，保留真实 mm 尺度。
5. `center_xyz_mm` 写入 `metadata.json` 和 `bbox_scale.json`，用于结果恢复到原始坐标。

## 数据预处理与质控

预处理流程：

1. 检查 `upper_local.stl`、`lower_local.stl`、`crown_{tooth_id}.stl` 和 `margin_local.xyz` 是否存在并可读取。
2. 检查 mesh 是否为空、破损、重复异常或存在极端坐标。
3. 检查 margin line 是否为有效闭合边缘线。
4. 根据 margin line 到上下颌局部模型的平均最近距离判定 prep 和 antagonist。
5. 对 prep 和 antagonist 裁取 20 mm ROI。
6. 使用 FPS 采样生成训练点云。
7. 将 margin line 重采样为 1024 点有序闭合曲线。
8. 生成批量 QC 汇总表，记录每例病例的几何检查结果。

建议 QC 阈值：

- `margin_to_prep_mean < 0.2 mm` 为理想状态
- `margin_to_crown_mean < 0.1 mm` 为理想状态
- `margin_to_prep_mean >= 0.5 mm` 时应暂停训练并复核原始导出
- `shape_ok=True` 且 `finite_ok=True` 才允许进入训练

## 模型框架

模型名称：`MLA-CrownNet`

全称：`Margin-Line-Anchored Crown Generation Network`

模型输入：

- 预备体所在颌 20 mm ROI 点云
- 对颌 20 mm ROI 点云
- margin line 闭合曲线
- FDI 牙位编码
- prep/antagonist 颌位编码

模型输出：

- AI 生成后牙全冠外表面点云
- AI 外冠面 mesh
- R1-R5 区域标注或区域预测
- 失败风险评分或风险热图

咬合接触图和邻接接触图在核心实验中作为评价指标计算，不作为生成器的直接输出，也不纳入核心训练损失。后续扩展实验可将其加入损失函数。

## 核心模块

### 上下文编码模块

编码预备体、邻牙上下文和对合牙空间信息。可选网络包括：

- DGCNN
- Point Transformer
- PointNet++
- DMC encoder

当前数据结构不单独分割近远中邻牙，邻牙和局部牙列关系通过 20 mm ROI 中的空间上下文隐式提供。

### 牙位编码模块

使用 tooth position embedding 表达不同后牙牙位的形态先验。

### 边缘线锚定模块

这是方案的核心机制改进。与仅把 margin line 当作普通点云输入不同，本方案将 margin line 作为牙冠生成的空间锚点。

基本思路：

1. 以 margin line 中心建立局部坐标原点。
2. 根据 margin line 和预备体几何估计局部冠向方向。
3. 将冠外表面按照距 margin line 的冠方距离划分为多个 ring。
4. 模型从 margin line 出发逐层生成颈部轴面、邻接区、轴-𬌗过渡区和咬合面。
5. 强化边缘区、颈部轴面和过渡区之间的结构连续性。

建议 ring 定义：

- Ring 0：margin line 本身
- Ring 1：距 margin line 冠方 0-0.5 mm
- Ring 2：距 margin line 冠方 0.5-1.0 mm
- Ring 3：距 margin line 冠方 1.0-2.0 mm
- Ring 4：颈部轴面区
- Ring 5：轴-𬌗过渡区
- Ring 6：咬合面区

实际实现时需要明确冠向方向估计方法和每个点归属 ring 的计算规则，以保证训练、评价和复现实验一致。

### 全冠生成解码模块

根据上下文特征、牙位特征和边缘线锚定特征生成完整后牙全冠外表面。输出可为 16384 点点云，必要时通过 Poisson reconstruction、DPSR 或其他 mesh reconstruction 方法生成外冠面网格。

### 失败风险提示模块

识别需要技师重点复核或修改的区域。

风险来源包括：

- R1 边缘区误差高
- margin line 附近短缺或悬突
- 咬合穿透深度过大
- 邻接过紧或无接触
- HD95 异常升高
- 多次生成结果不稳定

输出包括：

- risk score
- risk heatmap
- high-risk region label

## 损失函数

核心实验总损失：

```text
L_total = L_global
        + lambda_1 * L_margin_anchor
        + lambda_2 * L_margin_risk
        + lambda_5 * L_normal
        + lambda_6 * L_region_risk
```

其中：

- `L_global`: 整体 Chamfer Distance
- `L_margin_anchor`: 边缘线锚定损失
- `L_margin_risk`: 边缘区临床风险加权损失
- `L_normal`: 法向量一致性损失
- `L_region_risk`: 区域临床风险损失

边缘区风险加权函数建议：

```text
w(p) = 1 + alpha * exp(-d(p, M)^2 / (2 * sigma^2))
```

建议参数：

- `alpha = 2-4`
- `sigma = 1.0 mm`

咬合接触损失和邻接接触损失不纳入核心训练，仅作为后续扩展项。

## 实验分组

核心实验分为四组：

| 组别 | 名称 | 输入/改动 | 目的 |
| --- | --- | --- | --- |
| M0 | DMC baseline | 预备体 + 对颌/局部上下文 + 牙位编码 | 基础对照 |
| M1 | DMC + margin line input | M0 + margin line | 验证单纯 margin line 输入价值 |
| M2 | MLA-DMC | M1 + margin-line-anchored module + ring-wise strategy | 验证边缘线锚定机制价值 |
| M3 | M2 + risk-weighted loss | M2 + `L_margin_risk` | 验证边缘区风险加权损失价值 |

推荐主要比较：

- `M1 vs M0`: 验证 margin line 输入价值
- `M2 vs M1`: 验证边缘线锚定框架价值
- `M3 vs M2`: 验证边缘区风险加权损失价值
- `M3 vs M0`: 验证完整模型整体提升

## 评价指标

### 整体几何评价

- Chamfer Distance
- RMS error
- P2P distance
- HD95
- F-score@0.3 mm
- normal cosine similarity

### R1-R5 区域化评价

R1：边缘区，margin line 冠方 1.0 mm 范围。

- R1 RMS
- R1 CD
- R1 HD95
- 边缘短缺比例
- 边缘悬突比例
- margin-to-crown deviation

R2：颈部轴面区。

- 轴面突度误差
- 颈部外形误差
- R2 RMS
- R2 HD95

R3：邻接区。

- 近中最小距离
- 远中最小距离
- 近中接触面积
- 远中接触面积
- 无接触比例
- 穿透比例
- 邻接接触图一致性

R4：咬合区。

- 咬合区 RMS
- 最大穿透深度
- 平均穿透深度
- 穿透点比例
- 接触点数量
- 接触面积
- 咬合接触图一致性

R5：轴-𬌗过渡区。

- R5 RMS
- R5 HD95
- 曲面连续性
- 形态自然度

## CRCS 临床风险评分

Clinical Risk-weighted Crown Score（CRCS）用于综合评估 AI 生成冠的临床风险。

建议公式：

```text
CRCS = w1 * R1_error
     + w2 * R3_contact_error
     + w3 * R4_occlusal_error
     + w4 * R2_error
     + w5 * R5_error
```

推荐初始权重：

- R1 边缘区：0.30
- R3 邻接区：0.25
- R4 咬合区：0.25
- R2 颈部轴面区：0.10
- R5 轴-𬌗过渡区：0.10

CRCS 越低，表示临床风险越低。

权重应作为基于临床风险共识的预设权重，而不是根据测试结果事后调参。测试集中可使用 Spearman 相关分析验证 CRCS 与专家整体临床可接受性评分或临床可接受率之间的关系。

风险等级可按测试集三分位划分：

- 低风险：CRCS 位于测试集最低 1/3
- 中风险：CRCS 位于测试集中间 1/3
- 高风险：CRCS 位于测试集最高 1/3，或存在严重边缘、咬合、邻接或专家评分问题

## 接触图评价

咬合接触图由生成冠或技师冠与对合牙之间的距离场计算。

建议分类：

- `< 0 mm`: 穿透或干涉
- `0-100 um`: 强接触或近接触
- `100-300 um`: 轻接触或功能相关近接触
- `> 300 um`: 无接触

邻接接触图分别计算冠与近中邻牙、远中邻牙之间的距离场，评价接触区域、无接触区域、穿透区域和接触过强区域。

核心实验中，接触图用于评价 AI 冠与技师冠的功能接触差异。若后续时间充裕，可作为扩展实验加入训练损失。

## 专家盲评

建议从测试集中选择 20-30 例后牙冠，每例展示：

- M0 生成冠
- M2 生成冠
- M3 生成冠
- 技师设计冠

全部随机编码并隐藏来源。

评估者建议：

- 5 名修复医生
- 3 名牙科技师
- 最低不少于 5 名评估者

评分维度：

- 边缘适合性
- 轴面外形
- 邻接关系
- 咬合形态
- 整体临床可接受性
- 修改需求
- 是否可进入 CAD 微调流程
- 是否可直接加工或仅需轻微修改

评分标准：

- 5 分：非常好，无需或几乎无需修改
- 4 分：可接受，仅需轻微修改
- 3 分：基本可用，但需要中度修改
- 2 分：问题明显，需要大量修改
- 1 分：不可接受

临床可接受定义为整体可接受性评分不低于 4 分，或评估者认为仅需轻微修改即可使用。

## 小样本实体冠验证

建议选择 6-10 枚后牙冠，包括：

- 前磨牙：3-4 枚
- 磨牙：4-6 枚

采用“AI 外冠面 + CAD 内表面”的混合实体化路线：

1. AI 生成外冠面或全解剖形态。
2. CAD 软件依据预备体、margin line 和统一 cement gap/spacer 参数生成内表面。
3. 合并为可加工实体冠。
4. 记录是否进行人工修正。

必须记录：

- cement gap
- spacer 参数
- 材料
- 加工方式
- 软件版本
- 是否进行人工修正

验证方法建议采用改良 Triple Scan：

1. 扫描预备体模型。
2. 扫描 AI 设计冠。
3. 将 AI 冠就位于预备体后整体扫描。
4. 三维配准。
5. 测量边缘、轴面和咬合面间隙。

评价指标：

- marginal gap
- axial gap
- occlusal gap
- maximum gap
- virtual-physical measurement bias
- ICC
- Bland-Altman 平均偏倚和 95% 一致性界限

## 统计分析

几何指标比较采用配对设计。

两组比较：

- 正态分布：配对 t 检验
- 非正态分布：Wilcoxon 符号秩检验

多组比较：

- 正态分布：重复测量 ANOVA
- 非正态分布：Friedman 检验

事后比较使用 Holm-Bonferroni 校正。多区域多指标分析使用 Benjamini-Hochberg FDR 校正。

专家评分：

- Likert 评分：Wilcoxon 符号秩检验或 Friedman 检验
- 临床可接受率：McNemar 检验
- 评估者间一致性：ICC 或 Kendall's W
- 评估者内一致性：加权 Kappa

失败风险分析：

- 专家评分不高于 2 分或需要大量修改的病例定义为失败或高风险病例
- 预警特征来自预设几何风险指标，如 R1 误差、咬合穿透深度、邻接接触缺失或过度接触
- 使用分层 K 折交叉验证或留一法进行初步评估
- 通过 bootstrap 报告 AUC、敏感度和特异度的 95% 置信区间

失败风险分析定位为探索性或初步证据，不作为主要结局。

## 预期结果

1. M1 相较 M0 可降低 R1 边缘区误差，说明 margin line 输入具有价值。
2. M2 或 M3 相较 M1 进一步改善边缘区和颈部轴面区形态，说明边缘线锚定生成框架优于普通 margin line 输入。
3. M3 生成冠的咬合接触模式与技师冠差异处于临床可接受范围。
4. M3 生成冠的近远中邻接关系处于临床可接受范围。
5. M3 相较 M0 在整体几何误差、R1-R5 区域化误差、CRCS 和专家临床可接受率方面表现更优。
6. 风险评分可识别需要人工重点修改的病例和区域，提高 AI 冠设计结果的临床可解释性。

## 风险与应对

| 风险 | 应对 |
| --- | --- |
| 外冠面网格与实体化流程质量不稳定 | 优先保证 AI 外冠面稳定；实体冠采用 CAD 内表面半自动流程；记录人工修正时间；必要时只纳入网格质量合格病例 |
| 接触图监督训练不稳定 | 核心实验中先作为评价指标，模型稳定后再作为扩展损失 |
| margin line 质量不一致 | 医生复核 margin line，记录来源，并进行质量分层分析 |
| 模型输出存在局部噪声 | 加入 normal loss，进行 mesh 后处理和平滑 |
| 样本牙位分布不均 | 记录牙位分布，进行牙位分层分析，必要时限制主分析牙位类别 |
| 专家盲评一致性不足 | 制定统一评分标准，评估前培训，设置重复样本计算评估者内一致性 |
| DMC baseline 不可获得或复现成本过高 | 启动前确认代码、配置和算力；若不可用，改用定义清晰、可复现的点云生成基线 |

## 时间安排

建议总周期约 6-7 个月。

| 阶段 | 时间 | 主要任务 |
| --- | --- | --- |
| 第 1 阶段 | 半个月 | 数据整理、服务器版预处理与质控 |
| 第 2 阶段 | 1 个月 | M0 baseline 和 M1 margin line input 模型训练 |
| 第 3 阶段 | 1 个月 | M2 边缘线锚定模型和 M3 风险加权模型训练 |
| 第 4 阶段 | 1 个月 | 咬合接触图、邻接接触图和虚拟适合性评价脚本 |
| 第 5 阶段 | 1 个月 | M0-M3 测试集推理、R1-R5 评价和 CRCS 统计 |
| 第 6 阶段 | 1 个月 | 专家盲评、小样本实体冠加工和 Triple Scan 验证 |
| 第 7 阶段 | 1 个月 | 论文方法、结果、图表和讨论撰写 |

## 当前实施优先级

如果以 6 个月内完成为目标，建议优先级如下：

1. 数据 QC 和训练包稳定生成。
2. M0-M3 四组模型训练和推理。
3. R1-R5 区域化几何评价。
4. CRCS 风险评分。
5. 咬合和邻接接触图评价。
6. 专家盲评。
7. 小样本实体冠验证。

## M0 Baseline 代码

当前仓库已实现 M0 baseline，代码位置：

- `prepare_training_data.py`：从 `processed/` STL 和 margin line 重新生成固定密度训练点云。
- `train_m0.py`：训练 M0 baseline。
- `predict_m0.py`：使用训练好的 M0 checkpoint 生成预测冠点云。
- `src/crown_m0/`：数据集、采样、模型、损失和导出工具。

M0 的定义：

- 输入：`prep_points.npy`、`antagonist_points.npy`、牙位编码、预备体颌位编码。
- 不输入：`margin_points.npy`。
- 输出：tangent coarse-to-fine 解码器逐级生成 `8192 -> 32768 -> 65536` 点的 AI 冠外表面点云，每点包含坐标和法向量。
- 损失：三级 Chamfer/normal/point-to-plane 监督 + tangent sibling repulsion + point uniformity + local-plane consistency + normal-drift regularization。

M0 保留原始 direct MLP decoder 作为历史消融；正式高密度点云实验采用共享参数的 coarse-to-fine decoder：

```text
prep ROI point cloud -> PointNet encoder
antagonist ROI point cloud -> PointNet encoder
tooth_id -> embedding
prep_arch -> embedding
fused feature -> learned coarse seeds -> 8192 coarse points
8192 parent points -> 4 tangent-plane offsets per parent -> 32768 points
32768 parent points -> 2 tangent-plane offsets per parent -> 65536 points
```

上采样点是模型根据病例上下文学习的局部偏移，不是对 16384 点预测结果做随机插值。每个子点主要沿父点的局部切平面展开，第一层/第二层法向偏移分别限制在 `0.02 mm`/`0.01 mm`，避免高密度点云形成有厚度的点层并在 STL 中产生尖刺。

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：

- `numpy`
- `scipy`
- `trimesh`
- `torch`
- `tqdm`

### 统一点云密度

如果原始数据或 `processed/` 数据发生变化，应先重新生成固定密度训练包：

```bash
python3 scripts/prepare_training_data.py --data-dir data
```

该脚本会：

1. 读取每例 `processed/upper_local.stl`、`processed/lower_local.stl`、`processed/crown_{tooth_id}.stl` 和 `processed/margin_local.xyz`。
2. 以 margin line 中心作为 ROI 中心。
3. 自动判定 prep/antagonist 颌位。
4. 对 prep 和 antagonist 裁取 `20 mm x 20 mm x 20 mm` ROI。
5. 使用表面采样 + FPS 生成固定点数点云。
6. 将 margin line 重采样为 1024 点闭合曲线。

默认输出密度：

- `prep_points.npy`: `8192 x 6`
- `antagonist_points.npy`: `8192 x 6`
- `crown_points.npy`: `16384 x 6`
- `margin_points.npy`: `1024 x 3`

处理单个病例：

```bash
python3 scripts/prepare_training_data.py --case data/05580彭欣怡0_36
```

### 训练 M0

先生成一次固定患者级划分：

```bash
python3 scripts/make_m0_split.py \
  --data-dir data \
  --output splits/m0_patient_split_seed20260706.json
```

划分原则：

- 按患者分组，不按单颗牙随机分。
- 同一患者的所有牙位必须进入同一个集合。
- 推荐比例为 `70% / 15% / 15%`。
- 后续 M0、M1、M2、M3 都复用同一个 split，保证模型间比较公平。

当前新增数据的固定划分为：

```text
train: 330 cases
val:    70 cases
test:   70 cases
```

训练时显式传入该 split：

```bash
python3 scripts/train_m0.py \
  --data-dir data \
  --split-file splits/m0_patient_split_seed20260706.json \
  --output-dir runs/m0_tangent_c2f64k \
  --decoder coarse_to_fine_tangent \
  --coarse-points 8192 \
  --first-factor 4 \
  --second-factor 2 \
  --epochs 60 \
  --batch-size 4 \
  --chamfer-points 8192 \
  --normal-weight 0.20 \
  --point-to-plane-weight 0.50 \
  --local-plane-weight 0.20 \
  --normal-drift-weight 0.50
```

固定 split 已显式传入，训练脚本不会重新随机划分。划分结果保存到：

```text
runs/m0_tangent_c2f64k/split.json
```

checkpoint 保存到：

```text
runs/m0_tangent_c2f64k/latest.pt
runs/m0_tangent_c2f64k/best.pt
```

### 推理导出

```bash
python3 scripts/predict_m0.py \
  --checkpoint runs/m0/best.pt \
  --data-dir data \
  --output-dir predictions/m0
```

每例输出：

- `.npy`
- `.xyz`
- `.ply`


### 正式实验结果目录规范

后续所有正式实验结果统一写入 `result/`，不再写入 `runs/`、`predictions/` 或 `visualizations/` 作为主结果目录。

结果目录必须包含实验日期，格式为：

```text
result/YYYYMMDD/<experiment_name>/
```

例如 coarse-to-fine 64k 点云版 M0 正式评估输出为：

```text
result/20260727/m0_tangent_c2f64k/
```

正式实验默认只输出 test split，不再额外复制 `representative10/` STL 子集。若后续需要汇报用代表样本，只保存代表样本名单，或直接从 `test/cases/` 中挑选。

目录结构：

```text
result/YYYYMMDD/m0_tangent_c2f64k/
  config.json
  summary_by_sample_set.csv
  test/
    metrics_by_case.csv
    summary_metrics.csv
    summary_metrics.json
    cases/
      <case_id>/
        <case_id>_pred.npy
        <case_id>_pred.xyz
        <case_id>_pred.ply
        <case_id>_pred_tangent_mls_poisson.stl
        <case_id>_GT_technician.stl
```

要求：

1. `test/metrics_by_case.csv` 必须包含逐病例指标。
2. `test/summary_metrics.csv/json` 必须包含汇总指标。
3. 每个样本的 GT STL、预测点云和预测 STL 必须放在同一个 `cases/<case_id>/` 文件夹内。
4. 后续点云版 M0-M3 统一使用相同的 coarse-to-fine 输出密度和 STL 重建参数。
5. `result/` 是生成结果目录，默认不提交到 Git。

### 统一 STL 生成规则

后续正式点云版 M0-M3 统一采用高密度预测点云和固定的清理/重建流程：

```text
model predicts 65536 tangent-constrained points
-> statistical outlier removal
-> 0.08 mm voxel consolidation
-> normal-guided MLS projection
-> Screened Poisson reconstruction
-> remove low-density vertices
-> keep largest connected component
-> simplify to at most 50000 triangles
-> Taubin smoothing
-> export STL
```

也就是：

```text
official_stl_method = tangent_mls_poisson
```

M0-M3 的实验差别仍只体现在输入、网络模块和 loss，STL 参数保持一致：

```text
M0: prep + antagonist + tooth/arch -> tangent coarse-to-fine 64k points
M1: M0 + margin line input -> tangent coarse-to-fine 64k points
M2: M1 + margin-line anchored / ring-wise module -> tangent coarse-to-fine 64k points
M3: M2 + margin/risk-weighted loss -> tangent coarse-to-fine 64k points
```

正式 M0 评估命令：

```bash
python3 scripts/run_m0_official_experiment.py \
  --checkpoint runs/m0_tangent_c2f64k/best.pt \
  --date YYYYMMDD \
  --experiment-name m0_tangent_c2f64k \
  --stl-method tangent_mls_poisson \
  --poisson-depth 8 \
  --poisson-threads 8 \
  --mls-iterations 2 \
  --smooth-iterations 10
```

旧的 direct 16k/32k/64k MLP、无切平面约束 coarse-to-fine、dynamic template deformation 和 `alpha_clean_taubin` 结果保留为方法消融，不作为当前正式 M0 输出。增加输出点数必须由模型的切平面局部展开层学习，禁止把低密度预测点云机械插值后冒充高密度模型输出。

