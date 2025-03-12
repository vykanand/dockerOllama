#!/bin/bash

# Create a flag file to track if the model was previously pulled
MODEL_FLAG="/root/.ollama/.tinyllama_pulled"

# Check if model was already pulled in a previous build
if [ -f "$MODEL_FLAG" ]; then
  echo "TinyLlama model already pulled, skipping..."
else
  # Wait a moment for Ollama to start
  sleep 5

  # Pull the TinyLlama model
  echo "Pulling TinyLlama model..."
  ollama pull tinyllama

  # Check if model was pulled successfully
  if [ $? -eq 0 ]; then
    echo "TinyLlama model pulled successfully"
    # Create flag file to indicate successful pull
    touch "$MODEL_FLAG"
  else
    echo "Failed to pull TinyLlama model, retrying..."
    sleep 5
    ollama pull tinyllama
    
    if [ $? -eq 0 ]; then
      touch "$MODEL_FLAG"
    fi
  fi
fi

# Keep this script running to maintain the container active
tail -f /dev/null
