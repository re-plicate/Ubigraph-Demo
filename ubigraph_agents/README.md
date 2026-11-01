# UbiGraph specialist agents

The product entry point is `python -m ubigraph_agent`, or `bash run_agent.sh`.
Its `AgentHub` shares one in-process `AgentBus` across chat and fleet operations.
One configured LLM selects typed tools; specialist handlers perform the computation.

| Role | Handler | Responsibility |
|---|---|---|
| Orchestration | `ubigraph_agent/runtime.py` | Bounded LLM and tool loop, cancellation and streamed progress. |
| Data inspection | `ubigraph_agent/hub.py` | Causal historical inputs, gaps and freshness. |
| Temporal forecasting | `temporal_agent.py` | Local pretrained V3 forecasts. |
| Spatial analysis | `spatial_agent.py` | CoSimHeat geographic relationships. |
| Decision review | `hic_agent.py` | Observation-quality collection recommendations. |
| Fleet dispatch | `drone_agent.py` | Reviewed plans and simulator execution. |
| Vision detection | `detection_agent.py` | Optional local vehicle detection; weights required. |
| Transfer | `transfer_agent.py` | Optional profile-borrowing baseline; experimental. |

`system.py`, `api.py`, `orchestrator.py` and `requirement_agent.py` also expose a
standalone Python integration. The browser demo uses the shared product runtime,
so a second API service is unnecessary. The demo connects to the motion simulator;
physical flight requires a separate integration and is not enabled by default.
