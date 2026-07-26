# M0 Baseline 实验结果记录

日期：2026-07-06  
实验：后牙全冠生成 M0 baseline  
数据目录：`data/`  
固定划分：`splits/m0_patient_split_seed20260706.json`

## 1. 数据概况

当前训练包总数：`470` 例。  
患者数：`457` 个。

划分方式采用患者级划分，不按单颗牙随机划分。同一患者的多颗牙必须进入同一个集合，避免患者口内形态信息泄漏。

最终 split：

| Split | Cases | Patients |
| --- | ---: | ---: |
| Train | 330 | 320 |
| Val | 70 | 67 |
| Test | 70 | 70 |

患者泄漏检查结果：`0`。

### Test 牙位分布

| Tooth | Count |
| --- | ---: |
| 14 | 2 |
| 15 | 5 |
| 16 | 6 |
| 17 | 6 |
| 24 | 2 |
| 25 | 3 |
| 26 | 9 |
| 27 | 8 |
| 34 | 1 |
| 36 | 6 |
| 37 | 3 |
| 44 | 1 |
| 45 | 3 |
| 46 | 10 |
| 47 | 5 |

Test 上下颌分布：

| Arch | Count |
| --- | ---: |
| Upper | 42 |
| Lower | 28 |

说明：测试集没有 `35`，因为该牙位总量较少，当前随机搜索划分优先保证患者不泄漏和总体比例接近 70/15/15。后续如果专门做牙位亚组分析，需要单独设计分层 split。

## 2. M0 模型定义

M0 是最基础的 baseline，不使用 margin line。

输入：

- `prep_points.npy`: `8192 x 6`
- `antagonist_points.npy`: `8192 x 6`
- `tooth_id` embedding
- `prep_arch` embedding

输出：

- `16384 x 6` 预测冠外表面点云

模型结构：

```text
prep ROI point cloud -> PointNet-style encoder
antagonist ROI point cloud -> PointNet-style encoder
tooth_id -> embedding
prep_arch -> embedding
fused feature -> MLP decoder -> 16384 crown points
```

训练损失：

```text
loss = sampled Chamfer Distance + 0.05 * normal cosine loss
```

Chamfer 每轮随机采样点数：`2048`。

## 3. 训练配置

训练命令对应参数：

| Parameter | Value |
| --- | --- |
| Output dir | `runs/m0_new` |
| Epochs | 100 |
| Best epoch | 100 |
| Batch size | 2 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Chamfer points | 2048 |
| Normal weight | 0.05 |
| Device | CUDA |
| Split file | `splits/m0_patient_split_seed20260706.json` |

产物：

| File/Dir | Description |
| --- | --- |
| `runs/m0_new/best.pt` | 最佳 checkpoint |
| `runs/m0_new/latest.pt` | 最后一轮 checkpoint |
| `runs/m0_new/split.json` | 本次训练实际使用的 split |

Checkpoint 目录大小约 `2.4G`。

## 4. 预测结果

预测目录：`predictions/m0_new_all/`

全量 470 例均已导出：

| Format | Count |
| --- | ---: |
| `.npy` | 470 |
| `.xyz` | 470 |
| `.ply` | 470 |

说明：M0 原始输出是点云，不是天然 STL mesh。`.ply` 和 `.xyz` 是点云可视化格式。

## 5. Test 可视化结果

可视化目录：`visualizations/m0_new_test/`

已生成 test split 70 例结果：

| Output | Count | Description |
| --- | ---: | --- |
| `png/*.png` | 70 | GT、预测、误差热图三联图 |
| `error_ply/*.ply` | 70 | 误差着色预测点云 |
| `stl/*/*_M0_pred.stl` | 70 | M0 预测 STL |
| `stl/*/*_GT_technician.stl` | 70 | 技师设计冠 STL |
| `metrics.csv` | 1 | 每例误差指标 |
| `metrics.json` | 1 | 每例误差指标 JSON |
| `index.html` | 1 | 按 RMS 从高到低排序的 HTML 总览 |

STL 说明：

- `*_GT_technician.stl` 直接来自 `processed/crown_*.stl`，是技师冠原始网格。
- `*_M0_pred.stl` 是从 M0 预测点云通过 alpha-shape 重建得到的 STL。
- 因为 M0 输出是无拓扑点云，预测 STL 会受点云噪声和重建算法影响，可能出现坑洼、破面或不平整。

## 6. Test 定量指标

当前指标基于 test split 的 70 例，使用预测点云和技师冠采样点云之间的最近邻距离计算。

### Pred -> GT

| Metric | Mean |
| --- | ---: |
| Mean nearest distance | 0.412 mm |
| RMS | 0.508 mm |
| HD95 | 0.982 mm |

### GT -> Pred

| Metric | Mean |
| --- | ---: |
| Mean nearest distance | 0.225 mm |
| RMS | 0.272 mm |
| HD95 | 0.529 mm |

解释：

- `Pred -> GT` 更能反映预测冠点云是否偏离技师冠表面。
- `GT -> Pred` 更能反映技师冠区域是否被预测点云覆盖。
- 两个方向不对称，说明 M0 预测点云可能存在局部外飘、点分布不均或表面噪声。

## 7. Test 最差病例

按 `Pred -> GT RMS` 从高到低排序：

| Rank | RMS | HD95 | Case |
| ---: | ---: | ---: | --- |
| 1 | 0.843 | 1.697 | `data/05801赵俊Z_46` |
| 2 | 0.671 | 1.389 | `data/07447陈媛W_16` |
| 3 | 0.639 | 1.332 | `data/48452钦_44` |
| 4 | 0.625 | 1.353 | `data/07260卫桂荣W_27` |
| 5 | 0.622 | 1.266 | `data/07410余素兰W_37` |
| 6 | 0.617 | 1.210 | `data/53384林娟Z_16` |
| 7 | 0.602 | 1.125 | `data/18624王筱萍Z_26` |
| 8 | 0.595 | 1.155 | `data/53169舒博闻W_26` |
| 9 | 0.595 | 1.194 | `data/46800徐玲芝0_47` |
| 10 | 0.595 | 1.120 | `data/52467宋江锋W_47` |

建议优先打开这些病例的 STL 和误差 PLY，观察主要失败模式。

## 8. Test 最好病例

按 `Pred -> GT RMS` 从低到高排序：

| Rank | RMS | HD95 | Case |
| ---: | ---: | ---: | --- |
| 1 | 0.405 | 0.771 | `data/05797赵娟Z_26` |
| 2 | 0.406 | 0.767 | `data/24118彭璐Z_15` |
| 3 | 0.410 | 0.799 | `data/05631赵九0_24` |
| 4 | 0.419 | 0.789 | `data/05754杨馨洁W_14` |
| 5 | 0.420 | 0.793 | `data/49815叶庆恩Z_27` |

## 9. 结果判断

M0 baseline 的结果符合预期：它能学习到大致冠体位置和粗略形态，但表面质量较差。

主要问题：

1. **表面不平整**：M0 直接解码无拓扑点云，缺少连续曲面约束。
2. **点云转 STL 放大噪声**：alpha-shape 会把预测点云噪声转成坑洼网格。
3. **边缘区不可控**：M0 不使用 margin line，因此边缘和颈部轴面区域难以稳定。
4. **局部牙尖窝沟形态弱**：PointNet-style 全局特征 + MLP decoder 对细节生成能力有限。
5. **临床功能关系未建模**：M0 没有显式咬合/邻接约束。

因此，M0 的价值主要是作为下限 baseline，而不是可临床使用模型。

## 10. 下一步建议

优先级建议：

1. **实现 M1：加入 margin line input**
   - 增加 margin encoder。
   - 验证 `M1 vs M0` 是否降低 R1 边缘区误差。

2. **实现更合理的点云解码方式**
   - 当前 MLP 直接吐 16384 点容易产生噪声。
   - 可改为 template deformation、FoldingNet/AtlasNet patch decoder 或 coarse-to-fine decoder。

3. **增加 mesh/表面平滑约束**
   - Laplacian / repulsion / uniformity loss。
   - 避免点聚集和局部坑洼。

4. **建立 R1-R5 区域化评价脚本**
   - 当前只有整体点云误差。
   - 后续需要边缘区、颈部轴面区、咬合区、邻接区分区指标。

5. **固定当前 split 作为后续所有模型对照**
   - M1、M2、M3 必须复用 `splits/m0_patient_split_seed20260706.json`。
   - 否则模型间指标不可直接比较。

## 11. 关键路径

```text
splits/m0_patient_split_seed20260706.json
runs/m0_new/best.pt
predictions/m0_new_all/
visualizations/m0_new_test/index.html
visualizations/m0_new_test/metrics.csv
visualizations/m0_new_test/stl/
visualizations/m0_new_test/error_ply/
```
