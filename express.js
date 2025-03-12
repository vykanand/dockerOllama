const express = require("express");
const axios = require("axios");

const app = express();
app.use(express.json());

// Configuration constants
const MODEL_NAME = process.env.MODEL_NAME || "tinyllama"; // Use TinyLlama by default
const MAX_HISTORY_LENGTH = 6; // Keep last 3 exchanges (6 messages)
const OLLAMA_HOST = "localhost"; // Since both services are in the same container

// Session store
const sessions = new Map();

// Health check endpoint
app.get("/health", (req, res) => {
  res.status(200).send("OK");
});

// Model status endpoint
app.get("/model-status", async (req, res) => {
  try {
    const response = await axios.get("http://localhost:11434/api/tags");
    const models = response.data.models || [];
    const modelLoaded = models.some((model) => model.name === MODEL_NAME);

    res.json({
      modelName: MODEL_NAME,
      loaded: modelLoaded,
      allModels: models.map((m) => m.name),
    });
  } catch (error) {
    res.status(500).json({
      error: "Failed to check model status",
      details: error.message,
    });
  }
});

app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { messages, sessionId = "" } = req.body;

    // Get or create session
    const history = sessions.get(sessionId) || [];
    const prompt = messages.map((m) => `${m.role}: ${m.content}`).join("\n");

    // Generate response using Ollama
    const response = await axios.post(
      `http://${OLLAMA_HOST}:11434/api/generate`,
      {
        model: MODEL_NAME,
        prompt,
        stream: false,
      }
    );

    // Update session
    history.push({ role: "user", content: prompt });
    history.push({ role: "assistant", content: response.data.response });
    sessions.set(sessionId, history.slice(-MAX_HISTORY_LENGTH));

    res.json({
      choices: [
        {
          message: {
            role: "assistant",
            content: response.data.response,
          },
        },
      ],
      sessionId,
    });
  } catch (error) {
    console.error(
      "Error details:",
      error.response ? error.response.data : error.message
    );
    res.status(500).json({
      error: "Failed to generate response",
      details: error.message,
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
