import os
import logging
import subprocess
import gc
import signal
import sys
import time
from flask import Flask, request, jsonify
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to store the model
MODEL = None

def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    logger.info("Shutting down server...")
    if MODEL is not None:
        del MODEL
        gc.collect()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_model():
    """Get or initialize the model."""
    global MODEL
    if MODEL is None:
        model_path = "models/phi-2.Q4_K.gguf"
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Check if model exists, download if not
        if not os.path.exists(model_path):
            logger.info(f"Model not found. Downloading from HuggingFace...")
            try:
                model_url = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K.gguf"
                logger.info(f"Downloading model from {model_url}")
                subprocess.run(["wget", "-O", model_path, model_url], check=True)
                logger.info("Model downloaded successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download model: {e}")
                raise ValueError(f"Failed to download model: {e}")
        
        logger.info(f"Loading model: {model_path}")
        try:
            MODEL = Llama(
                model_path=model_path,
                n_ctx=4096,         # Increased context window
                n_threads=4,        # Increased threads for better performance
                n_batch=8,          # Increased batch size
                use_mlock=True,     # Lock memory to prevent swapping
                verbose=False       # Reduce verbosity
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    return MODEL

def format_prompt(messages):
    """Format messages for chat-style interaction."""
    formatted_prompt = ""
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            formatted_prompt = f"System: {content}\n\n"
        elif role == "user":
            formatted_prompt += f"Human: {content}\n\nAssistant: "
        elif role == "assistant":
            formatted_prompt += f"{content}\n\n"
            
    return formatted_prompt

def safe_generate(prompt, max_tokens=512, temperature=0.7, top_p=0.9):
    """Generate text with safety measures to prevent crashes."""
    # Force garbage collection before inference
    gc.collect()
    
    # Removed prompt length limitation
    
    try:
        model = get_model()
        response = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|endoftext|>", "Human:", "System:"]
        )
        
        # Force garbage collection after inference
        gc.collect()
        
        return response['choices'][0]['text']
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        return f"Error generating text: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok", 
        "model": "Phi-2 (4-bit quantization)", 
        "memory_usage": "Variable (supports longer contexts)"
    })

@app.route('/v1/completions', methods=['POST'])
def completions():
    """OpenAI-compatible completions endpoint."""
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({"error": {"message": "Missing 'prompt' in request", "type": "invalid_request_error"}}), 400
    
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 512)  # Increased default max tokens
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    
    logger.info(f"Generating completion for prompt: {prompt[:100]}...")
    
    try:
        response_text = safe_generate(
            prompt, 
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Format response in OpenAI format
        response = {
            "id": f"cmpl-{os.urandom(4).hex()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "phi-2-gguf",
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": None
            }
        }), 500

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    data = request.json
    
    if not data or 'messages' not in data:
        return jsonify({"error": {"message": "Missing 'messages' in request", "type": "invalid_request_error"}}), 400
    
    messages = data['messages']
    max_tokens = data.get('max_tokens', 512)  # Increased default max tokens
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    
    logger.info(f"Processing chat completion with {len(messages)} messages")
    
    try:
        # Removed message count limitation
        
        formatted_prompt = format_prompt(messages)
        
        # Removed prompt length limitation
        
        response_text = safe_generate(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Format response in OpenAI format
        response = {
            "id": f"chatcmpl-{os.urandom(4).hex()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "phi-2-gguf",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": len(formatted_prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(formatted_prompt.split()) + len(response_text.split())
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": None
            }
        }), 500

# Keep the original endpoints for backward compatibility
@app.route('/generate', methods=['POST'])
def generate_text():
    """Generate text based on a prompt."""
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing 'prompt' in request"}), 400
    
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 512)  # Increased default max tokens
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    
    logger.info(f"Generating text for prompt: {prompt[:100]}...")
    
    try:
        response = safe_generate(
            prompt, 
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        return jsonify({
            "response": response,
            "prompt": prompt
        })
    
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Chat with the model."""
    data = request.json
    
    if not data or 'messages' not in data:
        return jsonify({"error": "Missing 'messages' in request"}), 400
    
    messages = data['messages']
    max_tokens = data.get('max_tokens', 512)  # Increased default max tokens
    temperature = data.get('temperature', 0.7)
    
    logger.info(f"Processing chat with {len(messages)} messages")
    
    try:
        # Removed message count limitation
        
        formatted_prompt = format_prompt(messages)
        
        # Removed prompt length limitation
        
        response = safe_generate(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return jsonify({
            "response": response,
            "messages": messages
        })
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Enhanced Inference Server on port 5020")
    # Pre-load the model at startup to catch any initialization errors
    try:
        get_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)
        
    # Run with optimized settings for better performance
    app.run(
        host='0.0.0.0', 
        port=5020, 
        debug=False, 
        threaded=True,  # Enable threading for multiple requests
        use_reloader=False  # Disable reloader
    )
