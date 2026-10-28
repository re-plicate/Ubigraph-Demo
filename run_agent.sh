#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${UBIGRAPH_PYTHON:-}" ]]; then
  agent_python="$UBIGRAPH_PYTHON"
elif [[ -x ".venv-macos/bin/python" ]]; then
  agent_python=".venv-macos/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  agent_python=".venv/bin/python"
else
  printf 'No usable Python environment found. See README or set UBIGRAPH_PYTHON.\n' >&2
  exit 1
fi
# Open the workspace by default; the server reuses a healthy existing instance.
# --no-browser is available for terminal-only launches.
agent_open_browser=true
agent_args=()
for agent_arg in "$@"; do
  if [[ "$agent_arg" == "--no-browser" ]]; then
    agent_open_browser=false
  else
    agent_args+=("$agent_arg")
  fi
done
if [[ "$agent_open_browser" == true ]]; then
  exec "$agent_python" -m ubigraph_agent --open-browser "${agent_args[@]+"${agent_args[@]}"}"
fi
exec "$agent_python" -m ubigraph_agent "${agent_args[@]+"${agent_args[@]}"}"
