# UbiGraph

**An English-language workspace for parking analytics, specialist-agent collaboration, and a three-drone simulation.** Ask a question, inspect real model results, review a flight plan, and watch the simulator produce execution receipts.

UbiGraph runs locally with a Python backend and a browser interface. Numerical forecasts use the included trained V3 model. Chat uses your configured LLM API. You do **not** need to train a model, install Node.js, build the frontend, or connect a physical drone to run the demo.

## Quick start

### 1. Download the project

On GitHub, choose **Code → Download ZIP**, then extract the ZIP. Alternatively, clone the repository using the URL shown in GitHub's **Code** menu. Open a terminal in the extracted repository root: the folder containing `pyproject.toml`, `run_agent.sh`, `data/`, and `models/`.

The commands below assume this folder is your current directory. Repository ZIPs usually add a branch suffix to the folder name; use your actual extracted folder name.

**Requirements:** Python **3.12 recommended**, internet access for installing dependencies and calling the LLM, and a browser. The main service uses port **8765**. The existing demo has been verified on macOS with Python 3.12. Python 3.13+ is not recommended for the pinned NumPy/SciPy dependencies. Python 3.10–3.12 are permitted by the project metadata; Windows/Linux instructions are provided below but have not been verified on those operating systems in this release.

### 2. Install dependencies

**macOS / Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

If `python3.12` is unavailable, install Python 3.12 first, or use `python3` after confirming `python3 --version` reports a compatible version. On macOS Intel, PyTorch 2.2.2, NumPy 1.26 and SciPy 1.14 are the verified combination. If dependency resolution fails there, install these explicitly before installing the project:

```bash
.venv/bin/python -m pip install 'torch==2.2.2' 'numpy==1.26.4' 'scipy==1.14.1'
.venv/bin/python -m pip install -e .
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

No activation command is needed because these commands call the environment's interpreter directly. Do not copy another computer's `.venv` or `.venv-macos` directory; create your own environment.

### 3. Configure chat

Copy `.env.example` to **`.env.local`** in the repository root. Do this only if `.env.local` does not already exist; preserve any existing credentials.

```bash
# macOS / Linux — first setup only
cp .env.example .env.local
```

```powershell
# Windows PowerShell — first setup only
Copy-Item .env.example .env.local
```

Open `.env.local` in a text editor and set:

```dotenv
OPENAI_API_KEY=YOUR_DEEPSEEK_API_KEY
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-flash
OPENAI_REASONING_EFFORT=low
```

`OPENAI_*` are the configuration variable names used by the application even when the provider is DeepSeek. A working key, available API balance, and a model supporting the application's **Responses API streaming and function calling** are required for chat. An arbitrary Chat Completions-only endpoint is not interchangeable. The supplied values reflect the configuration used to verify this demo; if your provider changes model availability, select a compatible model it actually offers.

Credentials stay on the local backend and are not sent to the browser. `.env.local` is ignored by Git. Never commit keys. Shell environment variables override `.env.local`; restart the server after changing configuration.

**Without a model key:** you can still inspect the spatial graph and evaluation results, and plan/execute missions manually in the fleet page. Conversational analysis requires a configured model. The numerical predictor is local; chat routes requests to it rather than generating forecast values itself.

### 4. Start UbiGraph

**macOS / Linux**

```bash
bash run_agent.sh
```

The script opens your browser when ready and reuses an already healthy UbiGraph service. Keep the serving terminal open. To stop the service, press **Ctrl+C** in that terminal. Initial model loading can take tens of seconds.

On macOS, after installing dependencies and configuring the key, you can use the double-click launcher:

```bash
chmod +x run_agent.sh "Start Demo.command"
```

Then double-click **Start Demo.command** in Finder. The older Chinese-named launcher is an equivalent compatibility entry point; the product interface is English.

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m ubigraph_agent --open-browser
```

Open **http://127.0.0.1:8765** if your browser does not open automatically.

The shell launcher prefers `UBIGRAPH_PYTHON`, then `.venv-macos/bin/python`, then `.venv/bin/python`. To explicitly use the environment you just created on macOS/Linux:

```bash
UBIGRAPH_PYTHON="$PWD/.venv/bin/python" bash run_agent.sh
```

For a terminal-only start or a different port:

```bash
bash run_agent.sh --no-browser
bash run_agent.sh --port 8875
```

Do not start another service against the same runtime directory while an instance is already running. For an occupied port, first check whether UbiGraph is already available in your browser.

## A five-minute walkthrough

### 1. Explore the workspace

Open **Overview** for observation coverage, available simulated drones and recent tasks. **Agent Team** shows actual readiness for each specialist. The interface and new assistant replies use English. User-authored conversation text is preserved in its original language.

### 2. Ask for a real forecast

Open **Team chat**. Use the default cutoff **2014-09-30T23:30**, **Full history**, and the default seed 7. Enter:

> Forecast BRUUNS availability for the next 30 minutes and find the two most similar garages. Give a short English summary.

The timeline shows actual tool calls. The result cards show a forecast, history and spatial weights. With the included checkpoint, this example produces approximately **902.95 available spaces** for **2014-10-01 00:00**. KALKVAERKSVEJ and SALLING are its closest CoSimHeat neighbors. These numbers describe historical replay, not today's parking conditions.

You can select a specialist in **Delegate to**, or leave **Orchestrator** selected for automatic tool routing.

### 3. Inspect an observation gap

Set **Data cutoff** to **2014-10-01T16:00** using the displayed `YYYY-MM-DDTHH:mm` format. Cutoffs must lie on a 30-minute boundary within the supported historical range. Changing cutoff, protocol or seed creates a fresh conversation.

> Inspect observation quality at all six garages and assess whether simulated collection is needed. Explain the evidence. Recommend targets only; do not execute a mission.

This cutoff contains a real observation gap. The review reports observed/missing history and recommends collection based on input quality. Attention weights and similarity are not confidence scores, and low parking availability alone does not justify dispatch.

### 4. Plan, review, then execute

Keep **Simulation execution → Review commands first** and ask:

> Check fleet status and plan simulated collection at G1, G3 and G5 using up to three available drones. Collect for 3 seconds per target, return and land. Generate commands, but do not execute yet.

Open **Fleet mission** and inspect the **Command workspace**. You can download the structured JSON. Click **Execute simulation commands** to start the plan. Alternatively, select targets in **External mission input**, click **Generate plan**, then **Execute plan**.

Watch **Flight scene**, **Fleet status**, **Mission & collection receipts**, and **Execution timeline**. The optimizer may use fewer than all three drones. The example commonly assigns drones 1 and 3. A requested 3-second dwell rounds up to 4 seconds because the simulator uses 2-second slots.

Use **Cancel remaining visits & return** to request a return. Cancellation does not mean the drones have already landed. The simulator runs independently of chat; stopping an assistant response does not stop a flight. Speed controls change simulation speed, not model inference speed.

To let an explicitly requested chat action submit a mission, select **Allow simulation execution** before asking it to plan and execute. Review-first mode blocks takeoff/navigation/mission submission by the model; manual execution remains available in the command workspace.

### 5. Inspect evidence and results

- **Spatial relationships:** the checkpoint's CoSimHeat geographic prior. Similarity is not forecast confidence.
- **Model evaluation:** offline comparisons on matching test labels, with separate full-history and few-shot scopes.
- **Mission receipts:** actual simulated arrival/collection/return/landing events. Completion is determined from execution, not estimated duration.

## Capabilities and limits

| Component | Current behavior |
|---|---|
| Orchestrator | One configured Responses LLM selects typed specialist tools and explains their results. |
| Data inspection | Reads causal historical windows, observation masks and missing/stale inputs. |
| Temporal forecasting | Runs the registered frozen V3 checkpoint for a 30-minute single-step prediction. |
| Spatial analysis | Reads the geographic CoSimHeat relationships used by that checkpoint. |
| Decision review | Rule-based collection recommendations from observation quality; not calibrated confidence. |
| Fleet dispatch | Validated plans, structured commands, background three-drone motion simulation and persisted receipts. |
| Vision detection | Requires valid local COCO detection weights; otherwise reports configuration needed. |
| Transfer research | Experimental nearest-profile borrowing; not validated cross-city transfer. |

The directory contains eight roles: one LLM orchestrator plus seven specialist modules. These are not eight independent LLMs. Shared specialists execute registered Python handlers against the same model engine and fleet service.

**No physical drone is connected through the demo.** `ubigraph.flight.v1` is a semantic command manifest for the simulator, not a MAVLink packet stream. Simulation collection produces action receipts with `measurement=null`; it does not generate measured parking availability, repair the historical dataset, retrain V3, or establish improved forecasting accuracy.

Optional vision setup:

```bash
.venv/bin/python -m pip install -e '.[vision]'
```

Set `UBIGRAPH_DETECTION_WEIGHTS=/absolute/path/to/your/coco-weights.pt` in `.env.local`, then restart. Do not expect simulated collection to provide a real image or a measured vehicle count.

## Data, models and reproducibility

The original Aarhus parking CSV contains **55,264 rows from eight garages**, spanning **2014-05-22 to 2014-11-04**. The serving model uses six: BRUUNS, KALKVAERKSVEJ, MAGASIN, SALLING, SCANDCENTER and SKOLEBAKKEN.

Data provenance is the **CityPulse EU FP7 project's historical Aarhus parking dataset**. This is the dataset attribution, separate from the product name **UbiGraph**. The [EU-hosted D2.3 report](https://cordis.europa.eu/docs/projects/cnect/5/609035/080/deliverables/001-D23Ares2015916740.pdf) lists real Aarhus parking data for May–November 2014 and documents CC BY 4.0 attribution requirements. The [archived dataset catalog](https://smartcity.linkeddata.es/datasets/datasets/iot.ee.surrey.ac.uk8080datasets.htmlparking.html) describes eight lots and 55,264 observations. Local characteristics match that description; this repository does not retain an original download receipt or independently verified upstream file checksum.

The primary serving configuration uses 24 half-hour observations (12 hours), 8 temporal neighbors, GroC condensation and spatial attention. Full-history evaluation is primary; few-shot is an additional experiment with a separate evaluation scope. Source timestamps are irregular; 30-minute alignment is preprocessing, not the raw sampling interval.

Keep these assets in a downloaded release:

| Path | Purpose |
|---|---|
| `data/raw/aarhus_parking.csv` | Historical parking records. |
| `data/raw/aarhus_parking_address.csv` | Garage metadata and coordinates. |
| `models/parking/config.json`, `fewshot_config.json` | Input/model configuration. |
| `models/parking/serving.json` | Checkpoint registry and serving metadata. |
| Checkpoint paths referenced by `serving.json` | Trained full-history/few-shot models for registered seeds. |
| Other files in `models/parking/` | Prepared assets and comparison results used by the included release. |

The included assets enable the demo without retraining. If a GitHub download omits them, obtain the complete release from the repository maintainer before starting; an API key alone cannot replace missing local model/data files.

Runtime records are created under **`runtime/agent/`**. Conversations, tasks and simulator state persist across restarts. Incomplete tasks are marked interrupted; flights do not automatically resume. Do not delete this directory merely to resolve a startup error; it contains your history.

## Architecture and repository map

```text
Browser: HTML / CSS / JavaScript
    ↕ Same-origin HTTP APIs + server-sent events
Local API + AgentRuntime ↔ Responses API / configured LLM
    ├─ AgentHub / AgentBus → registered specialist tools
    ├─ V3Engine → causal historical data + frozen model checkpoint
    ├─ DroneService → planner + background simulator + action receipts
    └─ SQLite → conversations, tasks, events and fleet snapshots
```

| Directory / file | Role |
|---|---|
| `ubigraph_agent/web/` | English product interface; no frontend build step. |
| `ubigraph_agent/server.py` | Main local HTTP/SSE entry point. |
| `ubigraph_agent/runtime.py`, `responses.py` | Bounded LLM/tool loop and provider transport. |
| `ubigraph_agent/hub.py`, `tools.py` | Shared specialist registry, schemas and execution policy. |
| `ubigraph_agent/engine.py` | Serving adapter for registered V3 checkpoints. |
| `ubigraph_agent/english.py`, `english.json` | English presentation of system messages and legacy records. |
| `ubigraph_core/`, `ubigraph_st/` | Data processing, graph condensation, forecasting and research training. |
| `ubigraph_drone/` | Simulation scene, assignment, route planning, control and receipts. |
| `ubigraph_agents/` | Specialist handlers and a separate legacy research API. |
| `simulation/` | Older concept demo with synthetic frontend elements; not the main product entry point. |
| `tests/`, `scripts/`, `configs/`, `results/` | Regression checks and research workflows. |

Only the main service on **8765** is needed. Do not start the legacy research API on 8766 to use the product.

Useful main-service endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Local service readiness and provider configuration status. |
| `GET /api/bootstrap` | Context, coverage, spatial and evaluation data. |
| `GET /api/agents` | Actual agent readiness. |
| `GET /api/operations` | Workspace overview and fleet summary. |
| `POST /api/sessions` | Create a conversation for a context. |
| `POST /api/chat` | Submit `session_id`, `message`, optional context/model/agent, and review/execute policy. |
| `GET /api/tasks/{id}/events` | Stream model/tool task events. |
| `GET /api/drone/state` | Current simulation state. |
| `GET /api/drone/events` | Live simulation stream. |
| `POST /api/drone/plan`, `/api/drone/start` | Generate a plan and start a registered feasible plan. |
| `GET /api/drone/plans/{id}/commands` | Structured command manifest. |

The default server binds to loopback and has no multi-user authentication. It is intended for local use; review deployment access controls before exposing it to a network.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Double-click launcher is denied | Run the `chmod +x` command above, or use `bash run_agent.sh`. |
| No usable Python environment | Complete installation; verify `.venv/bin/python` or use `UBIGRAPH_PYTHON`. |
| NumPy/SciPy/PyTorch installation fails | Check the Python version; use Python 3.12. On Intel macOS, try the verified versions above. |
| Missing data, `serving.json` or checkpoint | Confirm you downloaded the complete project including `data/raw/` and `models/parking/`. |
| Port/runtime already in use | Open the existing service. Stop its owning terminal before starting another instance or changing configuration. |
| Browser does not open | Visit `http://127.0.0.1:8765` manually; check terminal output for the actual port. |
| Chat says not configured | Save `.env.local` in the repository root and restart; check for overriding shell environment variables. |
| Model authentication, balance or quota error | Check your provider account/key and API balance. Separate providers have separate balances. |
| Model endpoint fails | Verify base URL and model support for Responses streaming/function calling; catalog connectivity alone does not verify chat. |
| Historical input invalid | Use a supported cutoff on a half-hour boundary. Inspect data quality; simulation cannot fill historical observations. |
| Plan expired or infeasible | Recheck battery, faults, targets and fleet state, then generate a fresh plan. |
| Agent reply stops but drone continues | Chat cancellation and flight cancellation are separate. Use the fleet cancellation control. |
| Vision reports configuration needed | Supply valid local detection weights and the optional vision dependency, then restart. |

## Development checks and training

Install the optional test dependency and run the existing functional regression suite:

```bash
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -k 'not test_detection_stub_without_weights'
```

On Windows, use `.\.venv\Scripts\python.exe` in place of `.venv/bin/python`. This release has 93 passing existing checks with that selection. One legacy test expects fabricated detections without weights; it is intentionally excluded because the product reports unconfigured vision instead. The unfiltered suite still contains that outdated expectation.

Training is optional research work, not a demo prerequisite. It can be expensive. Use a separate output directory to preserve the supplied serving checkpoints:

```bash
.venv/bin/python -m ubigraph_st run --config configs/st_main.json --output outputs/st_main_retrain
```

Training a new run does not automatically replace the registered product checkpoint. Serving paths and input specifications must remain consistent before registering a replacement.
