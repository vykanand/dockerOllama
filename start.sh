#!/bin/bash

# Start Ollama in the background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
  echo "Waiting for Ollama server to be ready..."
  sleep 2
done

# Pull the model if specified
if [ ! -z "$OLLAMA_MODEL" ]; then
  echo "Pulling model: $OLLAMA_MODEL"
  ollama pull $OLLAMA_MODEL
fi

# Start Node.js server
echo "Starting Node.js server..."
cd /app
npm start &
NODE_PID=$!

# Handle shutdown
function cleanup {
  echo "Shutting down services..."
  kill $NODE_PID
  kill $OLLAMA_PID
  wait
}

trap cleanup SIGINT SIGTERM

# Keep the container running
wait $OLLAMA_PID $NODE_PID
