# CityClaw 大实验表

## A. 预测精度消融（Aarhus/CityPulse, 3 seeds, 9748 test labels）

| protocol     | method      | mode        | variant     |   mae |   mae_std |   rmse |   nmae |
|:-------------|:------------|:------------|:------------|------:|----------:|-------:|-------:|
| fewshot      | full        | attention   | fewshot_6h  | 18.30 |      3.88 |  33.51 |   2.46 |
| fewshot      | gcond       | attention   | fewshot_6h  | 16.92 |      1.00 |  33.58 |   2.17 |
| fewshot      | groc        | attention   | fewshot_6h  | 17.06 |      0.74 |  33.85 |   2.19 |
| fewshot      | groc        | gate        | fewshot_6h  | 14.14 |      1.18 |  28.89 |   1.88 |
| fewshot      | groc        | local       | fewshot_6h  | 15.52 |      0.10 |  31.69 |   1.98 |
| fewshot      | persistence | local       | fewshot_6h  | 14.40 |      0.00 |  31.49 |   2.01 |
| fewshot      | random      | attention   | fewshot_6h  | 17.27 |      1.63 |  33.93 |   2.25 |
| full_history | full        | attention   | baseline_6h |  9.64 |      1.13 |  20.26 |   1.34 |
| full_history | full        | local       | baseline_6h |  9.62 |      0.63 |  19.92 |   1.35 |
| full_history | gcond       | attention   | baseline_6h | 11.67 |      0.46 |  24.18 |   1.59 |
| full_history | groc        | attention   | main_12h    | 11.23 |      0.11 |  23.91 |   1.54 |
| full_history | groc        | attention   | baseline_6h | 11.96 |      0.59 |  24.55 |   1.62 |
| full_history | groc        | gate        | baseline_6h | 11.60 |      0.20 |  25.32 |   1.60 |
| full_history | groc        | local       | baseline_6h | 11.71 |      0.19 |  24.21 |   1.59 |
| full_history | groc        | no_prior    | baseline_6h | 11.96 |      0.59 |  24.55 |   1.63 |
| full_history | groc        | no_residual | baseline_6h | 11.71 |      0.28 |  24.08 |   1.59 |
| full_history | groc        | uniform     | baseline_6h | 11.96 |      0.59 |  24.55 |   1.63 |
| full_history | persistence | local       | persistence | 14.48 |      0.00 |  31.59 |   2.02 |
| full_history | random      | attention   | baseline_6h | 11.67 |      0.53 |  24.25 |   1.60 |
| full_history | random      | local       | baseline_6h | 11.39 |      0.43 |  24.40 |   1.55 |


## B. 多智能体系统指标

| section       | metric                 |   value | detail                                                                                                                                                                                                                                                                     |
|:--------------|:-----------------------|--------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| B_multi_agent | hic_trigger_rate_mean  |    0.15 | [{"seed": 17, "n_samples": 9748, "residual_std": 20.951, "hic_trigger_rate@85pct": 0.15}, {"seed": 27, "n_samples": 9748, "residual_std": 20.978, "hic_trigger_rate@85pct": 0.15}, {"seed": 7, "n_samples": 9748, "residual_std": 21.385, "hic_trigger_rate@85pct": 0.15}] |
| B_multi_agent | agent_call_latency_ms  |    0.19 | 6 in-process calls                                                                                                                                                                                                                                                         |
| B_multi_agent | evidence_chain_entries |    6    | ordered, auditable                                                                                                                                                                                                                                                         |
| B_multi_agent | agents_registered      |    7    | temporal,spatial,transfer,detection,drone,hic,orchestrator                                                                                                                                                                                                                 |


## C. 无人机调度（能量约束下的多机协同）

| section            |   n_drones |   n_targets | feasible   |   makespan_s |   total_distance_m | energy_per_drone_pct              | rejected   |
|:-------------------|-----------:|------------:|:-----------|-------------:|-------------------:|:----------------------------------|:-----------|
| C_drone_scheduling |          1 |           2 | True       |           84 |                 36 | {"1": 22.4}                       | []         |
| C_drone_scheduling |          2 |           4 | True       |           92 |                 72 | {"1": 23.7, "2": 21.2}            | []         |
| C_drone_scheduling |          3 |           5 | True       |           96 |                 98 | {"1": 22.4, "2": 13.9, "3": 25.6} | []         |


## D. MAVLink 接口壳

| section         | metric                    | value                                    | detail                                              |
|:----------------|:--------------------------|:-----------------------------------------|:----------------------------------------------------|
| D_mavlink_shell | command_surface           | connect/arm/takeoff/goto/land/disconnect | 6/6 ok; latency_ms=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   |
| D_mavlink_shell | mavlink_backend_supported | sim, mavlink(pymavlink), dryrun          | real SITL via udp:IP:14550 when pymavlink installed |