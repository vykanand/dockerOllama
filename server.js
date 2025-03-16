const express = require("express");
const cors = require("cors");
const axios = require("axios");
const bodyParser = require("body-parser");

const app = express();
const PORT = process.env.PORT || 3000;

// The URL of the Ollama service running locally in the same container
const OLLAMA_URL = "http://localhost:11434";

// Middleware
app.use(cors()); // Enable CORS for all routes
app.use(bodyParser.json()); // Parse JSON request bodies

// Health check endpoint
app.get("/health", (req, res) => {
  res.status(200).json({ status: "OK", message: "Ollama Gateway is running" });
});

// List available models
app.get("/models", async (req, res) => {
  try {
    const response = await axios.get(`${OLLAMA_URL}/api/tags`);
    res.status(200).json(response.data);
  } catch (error) {
    console.error("Error fetching models:", error.message);
    res.status(error.response?.status || 500).json({
      error: "Failed to fetch models",
      details: error.message,
    });
  }
});

// Generate text endpoint
app.post("/generate", async (req, res) => {
  try {
    console.log("Received generate request:", JSON.stringify(req.body));

    const response = await axios.post(`${OLLAMA_URL}/api/generate`, req.body, {
      responseType: "stream",
      headers: {
        "Content-Type": "application/json",
      },
    });

    res.setHeader("Content-Type", "application/json");
    response.data.pipe(res);

    response.data.on("error", (err) => {
      console.error("Stream error:", err);
      if (!res.headersSent) {
        res.status(500).json({ error: "Stream error", details: err.message });
      }
    });
  } catch (error) {
    console.error("Error generating text:", error.message);
    res.status(error.response?.status || 500).json({
      error: "Failed to generate text",
      details: error.message,
      request: req.body,
    });
  }
});

// Chat endpoint
app.post("/chat", async (req, res) => {
  try {
    console.log("Received chat request:", JSON.stringify(req.body));

    const response = await axios.post(`${OLLAMA_URL}/api/chat`, req.body, {
      responseType: "stream",
      headers: {
        "Content-Type": "application/json",
      },
    });

    res.setHeader("Content-Type", "application/json");
    response.data.pipe(res);

    response.data.on("error", (err) => {
      console.error("Stream error:", err);
      if (!res.headersSent) {
        res.status(500).json({ error: "Stream error", details: err.message });
      }
    });
  } catch (error) {
    console.error("Error in chat:", error.message);
    res.status(error.response?.status || 500).json({
      error: "Failed to process chat",
      details: error.message,
      request: req.body,
    });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Ollama Gateway running on port ${PORT}`);
  console.log(`Forwarding requests to Ollama at ${OLLAMA_URL}`);
});
