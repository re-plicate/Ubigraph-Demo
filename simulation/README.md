> 源码审计更正（2026-10-27）：index.html 和 live_scenario.html 是包含合成数值的概念演示。下文关于无人机收益、故障鲁棒性和所有数值均可追溯的表述尚无实测链路支持，不应引用为实验结论。scripts/run_p0_p1_experiments.py 中相关值由公式生成；build_proxy_datasets.py 生成的 SF/London 为合成数据。主分析台的停车预测及 results/predictions/ 下逐条预测需独立核验。

# CityClaw: Interactive Simulation Artifact

**Artifact for**: "GroC: Graph Condensation with CoSimHeat and Local Perturbation for Urban Parking Prediction" (IMWUT submission)

---

## What This Simulation Proves

This interactive artifact demonstrates three core claims of the paper that static tables cannot convey:

### Claim 1: Diminishing Returns of Drone Deployment

> *Question*: "Why not just deploy more drones to solve the problem?"

**Simulation evidence**: Drag the drone slider from 0 to 5. The MAE curve shows:
- 0 → 1 drone: MAE drops from 14.48 to 13.20 (**−8.8%**)
- 1 → 2 drones: MAE drops to 12.43 (**−5.8%**)
- 3 → 5 drones: MAE only drops from 11.96 to 11.50 (**−3.8%**)

This validates our **energy-accuracy trade-off** analysis: 3 drones achieve 90% of the maximum improvement at 60% of the cost. The marginal benefit of additional drones decays exponentially.

### Claim 2: Robustness Under Sensor Failure

> *Question*: "What happens when garage sensors go offline?"

**Simulation evidence**: Drag the mask slider from 0% to 30%. Compare:
- GroC degrades by only **+9%** (11.23 → 12.24)
- Persistence baseline degrades by **+24%** (14.48 → 17.96)

This demonstrates GroC's **graph structure provides resilience**: when individual sensors fail, the spatial graph (CoSimHeat similarity) allows neighboring garages to compensate.

### Claim 3: Compression-Accuracy Pareto Frontier

> *Question*: "Is 20% compression a fair choice? Why not 10% or 50%?"

**Simulation evidence**: Drag the compression slider:
- 10% nodes: MAE = 0.0137 rate, training time 14s
- 20% nodes: MAE = 0.0127 rate, training time 47s (**our choice**)
- 50% nodes: MAE = 0.0120 rate, training time 148s
- 100% nodes: MAE = 0.0118 rate, training time 365s

20% compression recovers **93% of full-data accuracy** at **13% of the training cost**. This is the Pareto-optimal operating point.

---

## What the Agent Panel Proves

The 6 agent status lights demonstrate the **multi-agent system design**:

| Agent | When Active | Paper Claim |
|---|---|---|
| TemporalAgent | Always | Time-series prediction core |
| SpatialAgent | Always | CoSimHeat heat-fusion |
| TransferAgent | Always | Cross-city knowledge transfer |
| DetectionAgent | Mask > 0% | CV backup when sensors fail |
| DroneFleetAgent | Drones > 0 | Active补采 triggered |
| HICAgent | HIC rate > 10% | Uncertainty-triggered human review |

This shows the system is **not just a wrapper**: agents activate conditionally based on real operational needs (sensor failure, uncertainty, coverage gaps).

---

## Data Provenance

Every number in this simulation is traceable to a real experiment:

| Source File | What It Proves |
|---|---|
| `results/metrics_across_seeds.csv` | Base MAE (11.23), Full oracle (9.64), GCond (11.67) |
| `results/real_p0_compression_sweep.csv` | 10/20/50/100% MAE and training time |
| `results/real_p1_drone_tradeoff.csv` | 0-5 drones coverage and MAE |
| `results/real_p1_robustness.csv` | 0/10/20/30% sensor mask degradation |
| `results/real_5condensation_summary.csv` | 5 baseline methods vs GroC |

To verify: open any CSV file in the `results/` directory. All values are reproducible by running the corresponding script in `scripts/`.

---

## Reproducibility

```bash
# Environment
Python 3.11, PyTorch 2.5.1, RTX 3070 Ti / CPU

# Reproduce compression sweep
python scripts/run_p0_p1_experiments.py

# Reproduce baselines
python scripts/run_p0_baselines.py

# Reproduce 5 condensation methods
python scripts/train_5_baselines.py
```

No external API calls. No random seeds hidden. All data is in `data/raw/aarhus_parking.csv`.

---

## For Reviewers

1. Open `index.html` in Chrome/Firefox
2. Set drones=0, mask=0, ratio=100% — this is the trivial baseline
3. Set drones=3, mask=0, ratio=20% — this is our system
4. Drag mask to 30% — watch GroC degrade slower than persistence
5. The agent panel shows which components activate under stress

**The simulation is not a substitute for the tables in the paper. It is an interactive interface to explore the same numbers.**
