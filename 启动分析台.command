#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")" || exit 1
printf 'Starting UbiGraph...\nThe browser opens automatically when ready.\nFirst launch may take 10–30 seconds.\nIf this window is serving the demo, keep it open; press Ctrl+C to stop.\n\n'
./run_agent.sh "$@"
agent_status=$?
if [[ "$agent_status" -ne 0 ]]; then
  printf '\nStartup failed. Read the message above.\n'
  if [[ -t 0 ]]; then read -r -p 'Press Enter to close...' agent_reply; fi
fi
exit "$agent_status"
