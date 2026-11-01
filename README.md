# UbiGraph

A conversational workspace for urban data analysis and reviewed sensing missions. An LLM orchestrator coordinates specialist tools; forecasting and fleet simulation run locally.

## Demo

https://github.com/user-attachments/assets/713098c2-af6b-46cf-860e-aaaa38d2a6d0

**2:05 · English narration and subtitles.** Recorded on an earlier build; the current demo provides the workflows documented below.

[Download video](docs/media/ubigraph-full-walkthrough.mp4) · [Quick preview](docs/media/ubigraph-demo.mp4) · [Usage guide](docs/USAGE.md)

## Features

- Inspect historical observations, missing data and spatial relationships.
- Generate local spatiotemporal forecasts with traceable model evidence.
- Review flight plans and structured commands before simulation execution.
- Track a three-drone fleet through collection, return and landing, with persistent action receipts.

## Quick start

Requires **Python 3.12** and a browser. Historical data and pretrained models are included.

### 1. Install

```bash
git clone https://github.com/re-plicate/Ubigraph-Demo.git
cd Ubigraph-Demo
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
cp -n .env.example .env.local
```

### 2. Configure chat

Edit `.env.local` with your provider credentials:

```dotenv
OPENAI_API_KEY=YOUR_API_KEY
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-flash
UBIGRAPH_API_STYLE=auto
```

Use a model available to your account with streaming and tool calling. Chat requires API credit; spatial inspection and manual fleet operations work without a key. Credentials stay on the backend and are excluded from Git.

### 3. Run

```bash
bash run_agent.sh
```

Open **http://127.0.0.1:8765**. Press **Ctrl+C** in the serving terminal to stop.

For Windows, alternative providers and installation troubleshooting, see the [usage guide](docs/USAGE.md#quick-start).

## Example workflow

In **Team chat**, keep the default historical cutoff and ask:

> Forecast BRUUNS parking availability for the next 30 minutes and find the two most similar garages.

Keep **Review commands first**, then ask:

> Plan simulated collection at G1, G3 and G5. Return and land after collection. Generate commands for review only.

Open **Fleet mission**, review the command manifest, and select **Execute simulation commands**. Follow the routes, fleet state and execution receipts.

<details>
<summary>Screenshots</summary>

![Forecast and specialist tool calls](docs/media/forecast.jpg)

![Fleet simulation](docs/media/fleet.jpg)

</details>

## Scope and documentation

The demo uses **2014 Aarhus parking-data replay** for six garages and a **three-drone motion simulator**. Collection receipts record simulated actions with `measurement=null`. Optional vision requires local detection weights.

[Configuration and workflows](docs/USAGE.md) · [Data attribution](docs/USAGE.md#data-attribution) · [Regression checks](docs/USAGE.md#regression-checks)
