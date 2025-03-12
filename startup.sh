#!/bin/bash

# Wait a moment for Ollama to start
sleep 5

# Pull the TinyLlama model
echo "Pulling TinyLlama model..."
ollama pull tinyllama

# Check if model was pulled successfully
if [ $? -eq 0 ]; then
  echo "TinyLlama model pulled successfully"
else
  echo "Failed to pull TinyLlama model, retrying..."
  sleep 5
  ollama pull tinyllama
fi

# Keep this script running to maintain the container active
tail -f /dev/null
