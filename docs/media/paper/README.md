# Manuscript figures

Selected illustrations from the author-supplied manuscript **UbiGraph: Connecting Observation Quality, Graph-Condensed Forecasting, and Reviewable Urban Sensing**.

The supplied archive contains a compiled main-text PDF and LaTeX sources. Printed figure numbers below refer to the compiled PDF; its filenames retain earlier numbering. The physical prototype is supplied in the updated source and figure assets.

These are original manuscript figures. PDF figures were rendered to PNG for browser display. The prototype PNG was copied byte-for-byte. Plot values, labels, colors, and panel layouts were not redrawn or modified. Experimental programs, raw results, reviewer notes, and the manuscript source are not bundled with these display assets.

| Display asset | Source in the supplied archive | Manuscript location |
| --- | --- | --- |
| [Functional setting](functional-setting.png) | `Figure/Fig2_functional_setting.pdf` | Compiled manuscript, Fig. 1, p. 4 |
| [Agent activation and collaboration](agent-architecture.png) | `Figure/Fig3_agent_architecture_isometric.pdf` | Compiled manuscript, Fig. 2, p. 5 |
| [Primary parking forecast](forecast-results.png) | `Figure/current_forecast_results.pdf` | Compiled manuscript, Fig. 7 and Table 8, p. 19 |
| [Physical prototype](physical-prototype.png) | `Figure/physical system.png` | Supplied figure and caption in Ubigraph2.0.tex |

## Captions and interpretation

### Functional setting

Users inspect, forecast, review, and track. Observation inspection, spatiotemporal analysis, collection review, and sensing operations expose distinct outputs. Each city supplies its network, analytical task, and collection resources. Maps and routes are schematic; route colors identify resources, rather than completion states. The runnable instance is the historical parking replay and simulator described in the project README.

### Agent architecture

A validated tool call returns a result or error to the orchestration loop. Core roles share a registry, message bus, and reusable results. Numerical and mission services implement selected actions. Offline training prepares artifacts, and persistent memory preserves records. The illustrations denote software responsibilities, not separate LLMs or deployed hardware. Solid arrows indicate calls/results; the dashed model-loading path connects offline preparation to runtime prediction.

### Physical prototype

The source figure shows the analytical workspace and a host-to-Crazyflie pathway using a local cflib bridge and Crazyradio PA. Indoor and outdoor frames illustrate hardware demonstrations. These frames and the README video have their own recording settings; this page does not imply they are synchronized recordings. The packaged demo uses a three-drone simulator, and its action receipts contain no measured parking values.

## Forecasting protocol

The primary forecasting comparison uses six garages, 24 half-hour history points, a 30-minute forecast horizon, and one common cohort of 9,748 labeled garage-time pairs across 1,625 synchronized origins and 35 calendar days. Training targets precede September 2014; validation covers September; test targets are from October onward. The UbiGraph primary bank contains 96 synchronized episodes.

Table 8 of the compiled manuscript reports the following values. Errors are parking spaces. UbiGraph values are means and sample standard deviations across three seed fits on the same cohort; deterministic baselines have no artificial seed variance.

| Method | MAE | RMSE |
| --- | ---: | ---: |
| Persistence | 14.477 | 31.589 |
| Daily seasonal naive | 52.926 | 113.568 |
| Ridge delta regression | 12.136 | **21.522** |
| UbiGraph, 96 episodes | **11.229 ± 0.108** | 23.906 ± 0.209 |

The original figure plots persistence, ridge, and UbiGraph. The seasonal baseline is retained in this table; it looks back 24 hours and therefore has a wider historical reach than the primary model's 12-hour input window.

The 22.4% MAE reduction is relative to persistence. Ridge has lower RMSE than UbiGraph. The three dots are repeated fits, not independent test cohorts. These measurements characterize one historical parking cohort. The manuscript also reports full-training and controlled bank-construction comparisons, where compactness trades retained history against prediction error. The primary checkpoint result and matched-bank reruns are separate experiments.

## Asset checksums

SHA-256 values identify the original source asset and the display export used in this repository.

- `functional-setting.png`
  - Source: `09c86f56693deb7169c9e9b8d07a8de870c0395e630ad6cf33af80dc9009f817`
  - Display export: `aed36806aa7aa5274e14b27fb5002d583ff767fd8cd9c491139abbc2f72a8b0f`
- `agent-architecture.png`
  - Source: `609900b59695c30defa4c33d0fb34f75934dcdecda973c0fb2b6c2f47f82c380`
  - Display export: `ea9abbf66e65852e53ad9878086418a24708ce424f1849302575dea04c3623ef`
- `forecast-results.png`
  - Source: `349f548f95b3fe612f5f7372dda24b26b7a26a95d1d8e6f2ada303fe7d79f7a2`
  - Display export: `b5c42d0957c4f1a634a9ef24d3483de643a34923fcc01b99af818232947d4a76`
- `physical-prototype.png`
  - Source: `fde483beb102ca1a086e2f2aceaadfd031d12d92b67605e8cd9165239c9e3c30`
  - Display export: `fde483beb102ca1a086e2f2aceaadfd031d12d92b67605e8cd9165239c9e3c30`
