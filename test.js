import axios from "axios";

const testOllamaConnection = async () => {
  try {
    // First check if we can access the base API
    const modelsResponse = await axios.get("http://localhost:11434/api/tags");
    console.log("Available models:", modelsResponse.data);

    // Then try a simple generation
    const response = await axios.post("http://localhost:11434/api/generate", {
      model: "tinyllama", // Use a model from the list above
      prompt: "Hello, how are you?",
      stream: false,
    });

    console.log("Response:", response.data);
  } catch (error) {
    console.error(
      "Error details:",
      error.response ? error.response.data : error.message
    );
  }
};

testOllamaConnection();
