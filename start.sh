#!/bin/bash
# Start Text-to-Material services

echo "🎨 Starting Text-to-Material..."

# Activate venv
source /home/mboard76/nvidia-workbench/stillion-ai/venv/bin/activate

# Start API server in background
echo "Starting API server on port 8001..."
cd /home/mboard76/nvidia-workbench/text-to-material
python -m uvicorn api.server:app --host 0.0.0.0 --port 8001 &
API_PID=$!

# Wait for API to start
sleep 3

# Start UI dev server
echo "Starting UI on port 5174..."
cd /home/mboard76/nvidia-workbench/text-to-material/ui
npm run dev -- --host --port 5174 &
UI_PID=$!

echo ""
echo "✅ Services started!"
echo "   API: http://localhost:8001"
echo "   UI:  http://localhost:5174"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "kill $API_PID $UI_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
