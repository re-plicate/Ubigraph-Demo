# CityClaw 多智能体层 (ubigraph_agents)

在原有 UbiGraph 预测/仿真之上，按论文三贡献与《项目书_开发推进版》重构为 7 个自治 agent。

## Agent 一览

| Agent | 文件 | 职责 | 对应论文/项目书 |
|---|---|---|---|
| `orchestrator` | orchestrator.py | 接收任务、调度各 agent、跑 analyze→gate→补采→再分析闭环 | Orchestration Layer |
| `temporal` | temporal_agent.py | 包装已训练 V3 时空预测器（GroC + 时间 GNN + 空间注意力） | Temporal Agent |
| `spatial` | spatial_agent.py | CoSimHeat 热力扩散相似度，给跨场注意力先验 | Spatial Agent |
| `transfer` | transfer_agent.py | 跨区域/跨城市最近邻迁移，冷启动预测 | Transfer Agent |
| `detection` | detection_agent.py | YOLOv11 停车车辆分割（有权重走真实推理，否则可复现桩） | Contribution 2 |
| `drone` | drone_agent.py | 包装 DroneService（时空 A*、能量约束）+ MAVLinkAdapter | Perception/UAV Adapter |
| `hic` | hic_agent.py | 不确定性 U / 数据风险 R / 代价 C 三因子触发人工协作 | HIC Layer |

Agent 之间通过 `AgentBus` 通信，每条消息写入有序 evidence chain（决策证据链）。

## 运行

```powershell
# 1. 后端 FastAPI（多智能体 API）
.\.venv\Scripts\python.exe -m uvicorn ubigraph_agents.api:app --port 8766
# 打开 http://127.0.0.1:8766/docs

# 2. 原有分析台（含无人机面板）保持不变
.\.venv\Scripts\python.exe -m ubigraph_agent --open-browser
# http://127.0.0.1:8765
```

## MAVLink 接口壳

`ubigraph_drone/mavlink_adapter.py` 暴露统一命令面：
`connect / arm / takeoff(alt) / goto(lat,lon,alt) / land / disconnect`。

- `backend="sim"`：默认，本地仿真，无需硬件。
- `backend="mavlink"`：装 `pymavlink` 后接 SITL 或实机，`MAVLINK_URL=udp:127.0.0.1:14550`。

通过环境变量切换：
```
MAVLINK_BACKEND=mavlink
MAVLINK_URL=udp:127.0.0.1:14550
```

## 实验表

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py
# 输出 results/experiment_table.csv 和 results/experiment_table.md
```

四块：A 预测精度消融（真实模型，9748 测试标签）、B 多智能体指标、C 多机能量约束调度、D MAVLink 命令面。

## 数据集

`ubigraph_core/datasets.py` 统一了 Aarhus(内置) / SFpark / London Parking 三个 loader。
目前仅 Aarhus 数据在 `data/raw/`；SFpark 与 London 需自行下载后放入该目录，loader 会自动识别。
