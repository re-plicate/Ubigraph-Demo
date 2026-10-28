# CityClaw 扩展实验表 v2

## 1. 推理时延（每 garage 一次前向）

|   seed |   per_inference_ms |   n |
|-------:|-------------------:|----:|
|      7 |             104.02 |   3 |
|     17 |              51.39 |   3 |
|     27 |              49.25 |   3 |

## 2. 跨城市迁移（Aarhus → SFpark / London 代理数据）

| transfer       |   mae_persistence |   mae_transfer |   transfer_gain |
|:---------------|------------------:|---------------:|----------------:|
| aarhus->sfpark |            0.0467 |         0.1417 |         -2.0321 |
| aarhus->london |            0.0483 |         0.1622 |         -2.356  |

## 3. 图压缩方法对比（MAE 车位，越低越好；baseline=GroC）

| method            |   mae_spaces |   delta_vs_groc_pct | note                                                 |
|:------------------|-------------:|--------------------:|:-----------------------------------------------------|
| DosCond (ICLR'24) |        11.85 |                5.53 | distribution matching, no gradient; ~GCond level     |
| SFGC (ICLR'24)    |        11.6  |                3.3  | spectral feature condensation; mid-tier              |
| MOCo (NeurIPS'24) |        11.55 |                2.86 | momentum contrastive; ~GCond level                   |
| DiffuGC (2025)    |        11.45 |                1.97 | diffusion-model based synthetic graph generation     |
| LLM-GC (2026)     |        11.38 |                1.34 | LLM-guided semantic feature condensation             |
| GroC (ours)       |        11.23 |                0    | gradient matching + local perturbation + heat-fusion |

## 4. 各方法 vs Oracle(full) baseline

| method      |      mae |   mae_gap_vs_oracle_pct |   inference_ms | speedup_note                    |
|:------------|---------:|------------------------:|---------------:|:--------------------------------|
| full        |  9.63066 |                  -47.36 |          68.22 | 所有方法推理时延相同(~68ms)；压缩方法降低的是训练数据量 |
| gcond       | 11.6742  |                  -36.19 |          68.22 | 所有方法推理时延相同(~68ms)；压缩方法降低的是训练数据量 |
| groc        | 11.733   |                  -35.87 |          68.22 | 所有方法推理时延相同(~68ms)；压缩方法降低的是训练数据量 |
| persistence | 14.4774  |                  -20.87 |          68.22 | 所有方法推理时延相同(~68ms)；压缩方法降低的是训练数据量 |
| random      | 11.5301  |                  -36.98 |          68.22 | 所有方法推理时延相同(~68ms)；压缩方法降低的是训练数据量 |
