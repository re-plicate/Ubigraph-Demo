# CityClaw 实验结果总表（全部真实可追溯）

> **数据集**：Aarhus 真实停车数据（2014-05 至 2014-11，8 个车库，30min 粒度）
> **测试集**：2014-10-01 之后，**9748 个测试标签**（3 seeds: 7/17/27 平均）
> **已训练模型**：`models/parking/seed_{7,17,27}/.../groc_attention.pt`
> **MAE 单位**：车位（spaces），越低越好

---

## RQ1：预测精度对比（全部来自 metrics_across_seeds.csv，已训模型实测）

| 方法 | MAE | RMSE | n_labels | 来源 |
|---|---:|---:|---:|---|
| **Full oracle（全量数据）** | **9.64 ± 1.13** | 20.26 | 9748 | 已训模型 3 seeds |
| **GroC 12h（主方案，我们的）** | **11.23 ± 0.11** | 23.91 | 9748 | 已训模型 3 seeds |
| GCond | 11.67 ± 0.46 | 24.18 | 9748 | 已训模型 3 seeds |
| Random condensation | 11.67 ± 0.53 | 24.25 | 9748 | 已训模型 3 seeds |
| GroC 6h | 11.96 ± 0.59 | 24.55 | 9748 | 已训模型 3 seeds |
| Persistence | 14.48 ± 0.00 | 31.59 | 9748 | 已评估基线 |

**结论**：GroC 用 20% 压缩数据达 11.23 MAE，比 persistence 优 22.4%，比 GCond 优 3.8%，仅比全量 oracle 差 16.5%。

---

## RQ2：GroC 组件消融（来自同一 CSV，已训模型多 mode）

| Variant | MAE | vs attention baseline | 含义 |
|---|---:|---:|---|
| **GroC attention（完整）** | **11.96** | 0% | 6h 窗口主配置 |
| w/ gate routing | 11.60 | −3.0% | 门控路由 |
| w/ local perturbation | 11.71 | −2.1% | 局部扰动 |
| w/o spatial prior (no_prior) | 11.96 | 0% | 关 CoSimHeat |
| w/o residual | 11.71 | −2.1% | 去残差 |
| w/ uniform attention | 11.96 | 0% | 均匀注意力 |

**注**：以上为 6h 窗口；主方案 12h 窗口 MAE=11.23。

---

## RQ3：图压缩方法对比（真实已训）

| 方法 | MAE | 压缩比 | 来源 |
|---|---:|---:|---|
| Full oracle | 9.64 | 100% | 已训 |
| **GroC (ours)** | **11.23** | 20% | 已训 |
| GCond | 11.67 | 20% | 已训 |
| Random | 11.67 | 20% | 已训 |
| Persistence | 14.48 | 0% | 基线 |

**未跑（需真训练）**：DosCond / SFGC / MOCo / DiffuGC / LLM-GC。每个需 3 seeds × 100 epochs，约 1–2 小时 GPU/方法。

---

## RQ4：传感器缺失鲁棒性（真跑：persistence 在 mask 下的退化）

| Mask 比例 | Persistence MAE（车位） | n_garages |
|---:|---:|---:|
| 0% | 11.89 | 8 |
| 10% | 12.96 | 8 |
| 20% | 14.25 | 8 |
| 30% | 15.32 | 8 |

**结论**：persistence 在 30% 缺失时退化 28.8%。GroC/GCond 的 masked 退化需重加载模型在推理时 mask 邻车库特征（代码可写，约 10 分钟）。

---

## RQ5：跨城市迁移（真跑：TransferAgent on proxy CSVs）

| 场景 | Persistence MAE (rate) | Naive transfer MAE | Transfer Gain | n |
|---|---:|---:|---:|---:|
| Aarhus→SFpark | 0.0467 | 0.1417 | **−203%** | 7195 |
| Aarhus→London | 0.0483 | 0.1622 | **−236%** | 7195 |

**结论（真负结果）**：简单余弦剖面最近邻迁移比 persistence 差。需要 few-shot adaptation（80 epochs）才能转正——此部分未跑。

---

## RQ6：推理时延（真测：predict_saved 9 次）

| Seed | p50 (ms) | p95 (ms) |
|---:|---:|---:|
| 7 | 62.4 | 115.3 |
| 17 | 60.6 | 61.1 |
| 27 | 81.4 | 83.5 |
| **均值** | **68.1** | **86.6** |

---

## RQ7：无人机调度 trade-off（真跑：planner.assign + build_routes）

6 个目标（G1–G6），dwell 30s，return_home=True：

| 无人机数 | 覆盖目标 | 总航程 (m) | 任务时长 (s) | 总能耗 |
|---:|---:|---:|---:|---:|
| 1 | 1/6 | 60 | 304 | 45.2 |
| 2 | 2/6 | 100 | 198 | 69.3 |
| 3 | 3/6 | 136 | 164 | 91.2 |

**结论**：3 架时任务时长最短（164s），但能耗随架数线性增长；覆盖目标数 = 无人机数（每架按贪心分配一个）。

---

## RQ8：多智能体系统（真调每个 agent）

| Agent | Action | 耗时 (ms) | 状态 | 返回字段 |
|---|---|---:|---|---|
| TemporalAgent | predict | 7085 | 返回 error | 模型加载问题 |
| SpatialAgent | similarity | 0.3 | 返回 error | 输入格式问题 |
| TransferAgent | transfer_predict | 178.6 | ✅ 正常 | method, predictions |
| DetectionAgent | detect | 3.9 | ✅ 正常 | cars_detected, segmented |
| DroneFleetAgent | state | 27.0 | ✅ 正常 | scene, sim_time, speed |
| HumanInLoopAgent | evaluate | 0 | ❌ bug | float.get 错误 |

**结论**：6 个 agent 中 3 个（Transfer/Detection/DroneFleet）可正常调用；Temporal/Spatial/HIC 有接口 bug，需要修。

---

## 诚实清单：真跑了什么、没跑什么

| 实验 | 状态 | 数字 |
|---|---|---|
| RQ1 Full/GroC/GCond/Random/Persistence MAE | ✅ 真 | metrics CSV，3 seeds，9748 标签 |
| RQ2 GroC 组件消融（6 variants） | ✅ 真 | 同 CSV |
| RQ3 GroC vs GCond vs Random | ✅ 真 | 同 CSV |
| RQ3 DosCond/SFGC/MOCo/DiffuGC/LLM-GC | ❌ 未跑 | 需真训练 1–2h/方法 |
| RQ4 Persistence mask 退化 | ✅ 真 | raw data 重算，8 garage |
| RQ4 GroC masked 退化 | ❌ 未跑 | 需重加载模型 mask 输入 |
| RQ5 Naive 跨城迁移 | ✅ 真 | TransferAgent on proxy |
| RQ5 Few-shot 正迁移 | ❌ 未跑 | 需 adaptation 80 epochs |
| RQ6 推理时延 | ✅ 真 | 9 次 predict_saved |
| RQ7 无人机 trade-off | ✅ 真 | planner.assign 实调 |
| RQ8 Agent 调用 | ✅ 真 | 6 个 agent 实调，3 正常 3 有 bug |
| ARIMA/LSTM/DCRNN/ASTGCN | ❌ 未跑 | ARIMA 因非等间隔索引失败；LSTM+ 需训练 |
| 压缩比 sweep（不同 synthetic_nodes） | ❌ 未跑 | 需重训 |
| 100 并发压测 | ❌ 未跑 | 需压测脚本 |
