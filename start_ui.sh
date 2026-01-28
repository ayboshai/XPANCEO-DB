#!/bin/bash
set -euo pipefail
cd /root/2

exec 9>/tmp/xpanceo_ui.lock
if ! flock -n 9; then
    echo "start_ui.sh already running. Exiting."
    exit 0
fi

while true; do
    echo "Starting Streamlit UI..."
    streamlit run /root/2/ui/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true >> /root/2/ui.log 2>&1
    echo "Streamlit crashed. Restarting in 2 seconds..."
    sleep 2
done
