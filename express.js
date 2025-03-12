import express from "express";
import axios from "axios";

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
      error: "AI processing failed",
      details: error.message,
    });
  }
});

// Also support the /aiserver endpoint for backward compatibility
app.post("/aiserver", async (req, res) => {
  try {
    const messages = [
      {
        role: "user",
        content: req.body.aiquestion,
      },
    ];

    const sessionId =
      req.body?.sessionId || Math.random().toString(36).substring(7);
    const history = sessions.get(sessionId) || [];
    const prompt = messages.map((m) => `${m.role}: ${m.content}`).join("\n");

    const response = await axios.post(
      `http://${OLLAMA_HOST}:11434/api/generate`,
      {
        model: MODEL_NAME,
        prompt,
        stream: false,
      }
    );

    history.push({ role: "user", content: prompt });
    history.push({ role: "assistant", content: response.data.response });
    sessions.set(sessionId, history.slice(-MAX_HISTORY_LENGTH));

    res.json({
      response: response.data.response,
      sessionId: sessionId,
    });
  } catch (error) {
    console.error(
      "Error details:",
      error.response ? error.response.data : error.message
    );
    res.status(500).json({
      error: "AI processing failed",
      details: error.message,
    });
  }
});

// Wait for Ollama to be ready before starting the server
const waitForOllama = async () => {
  let ready = false;
  while (!ready) {
    try {
      await axios.get("http://localhost:11434/api/tags");
      ready = true;
      console.log("Ollama service is ready");
    } catch (error) {
      console.log("Waiting for Ollama service to be ready...");
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
};

// Start server after Ollama is ready
waitForOllama().then(() => {
  app.listen(3000, () =>
    console.log(`HTTP Server running on port 3000 using model: ${MODEL_NAME}`)
  );
});
