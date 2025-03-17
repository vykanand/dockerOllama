import os
import logging
import subprocess
import gc
import signal
import sys
import time
import json
import re
from flask import Flask, request, jsonify, render_template, Response
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to store models
MODELS = {}
ACTIVE_MODEL = None

# Available models mapping
MODEL_CONFIGS = {
    "deepseek-coder": {
        "path": "models/deepseek-coder-1.3b-instruct.Q4_K.gguf",
        "url": "https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct.Q4_K.gguf",
        "name": "DeepSeek Coder 1.3B Instruct (Q4_K)"
    },
    "deepseek-r1": {
        "path": "models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",
        "url": "https://huggingface.co/TheBloke/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",
        "name": "DeepSeek R1 Distill Qwen 1.5B (Q4_0)"
    },
    "phi-2": {
        "path": "models/phi-2.Q4_K.gguf",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K.gguf",
        "name": "Phi-2 (Q4_K)"
    }
}

def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    logger.info("Shutting down server...")
    unload_model()  # Unload all models
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_model(model_name=None):
    """Get or initialize the model dynamically."""
    global MODELS, ACTIVE_MODEL
    
    # If no model specified, use active or default
    if model_name is None:
        if ACTIVE_MODEL is None:
            model_name = "deepseek-r1"  # Default model from memory
        else:
            return MODELS[ACTIVE_MODEL]
    
    # Validate model name
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model name. Available models: {list(MODEL_CONFIGS.keys())}")
    
    # Return cached model if already loaded
    if model_name in MODELS:
        ACTIVE_MODEL = model_name
        return MODELS[model_name]
    
    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Check if model exists, download if not
    if not os.path.exists(model_path):
        logger.info(f"Model {model_name} not found. Downloading...")
        try:
            subprocess.run(["wget", "-O", model_path, config["url"]], check=True)
            logger.info(f"Model {model_name} downloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            raise ValueError(f"Failed to download model {model_name}: {e}")
    
    logger.info(f"Loading model: {model_name}")
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            n_batch=8,
            use_mlock=True,
            verbose=False
        )
        MODELS[model_name] = model
        ACTIVE_MODEL = model_name
        logger.info(f"Model {model_name} loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

def unload_model(model_name=None):
    """Unload a specific model or all models."""
    global MODELS, ACTIVE_MODEL
    
    if model_name is None:
        # Unload all models
        for name in list(MODELS.keys()):
            del MODELS[name]
        MODELS = {}
        ACTIVE_MODEL = None
        gc.collect()
        logger.info("All models unloaded")
    elif model_name in MODELS:
        # Unload specific model
        del MODELS[model_name]
        if ACTIVE_MODEL == model_name:
            ACTIVE_MODEL = None
        gc.collect()
        logger.info(f"Model {model_name} unloaded")
    else:
        logger.warning(f"Model {model_name} not found in loaded models")

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

def safe_generate(prompt, max_tokens=4096, temperature=0.7, top_p=0.9):
    """Generate text with safety measures to prevent crashes and ensure complete responses."""
    # Force garbage collection before inference
    gc.collect()
    
    try:
        model = get_model()
        response = ""
        current_tokens = 0
        max_attempts = 5
        attempt = 0
        
        while current_tokens < max_tokens and attempt < max_attempts:
            attempt += 1
            
            # Generate a chunk of response
            chunk = model.create_completion(
                prompt + response,
                max_tokens=min(1024, max_tokens - current_tokens),  # Generate in chunks
                temperature=temperature,
                top_p=top_p,
                stop=["\n\n", "Human:", "System:"]
            )
            
            # Get the generated text
            chunk_text = chunk['choices'][0]['text'].strip()
            
            # Check if we have a complete response
            if chunk_text and not chunk_text.endswith(('}', ';', '.', '?', '!')):
                # If not complete, try to get more context
                additional = model.create_completion(
                    chunk_text,
                    max_tokens=256,
                    temperature=temperature,
                    top_p=top_p,
                    stop=["\n\n", "Human:", "System:"]
                )
                chunk_text += additional['choices'][0]['text'].strip()
            
            # Add to the response
            response += chunk_text
            current_tokens += len(chunk_text.split())
            
            # Force garbage collection after each chunk
            gc.collect()
            
            # Check if we have a complete response
            if response and any(response.endswith(x) for x in ['}', ';', '.', '?', '!']):
                break
            
            # Add a small delay to prevent overwhelming the model
            time.sleep(0.1)
        
        # Ensure the response is complete and remove padding words
        response = response.strip()
        
        # Remove common padding words and tags
        padding_patterns = [
            r'\s*</?think\s*>',
            r'\s*</?code\s*>',
            r'\s*</?pre\s*>',
            r'\s*\\n\s*',
            r'\s*\\t\s*'
        ]
        
        for pattern in padding_patterns:
            response = re.sub(pattern, ' ', response)
        
        # Ensure the response ends properly
        if not response.strip().endswith(('{', '}', ';', '.', '?', '!')):
            # Try one final attempt to get a complete response
            final = model.create_completion(
                response,
                max_tokens=256,
                temperature=temperature,
                top_p=top_p,
                stop=["\n\n", "Human:", "System:"]
            )
            response += final['choices'][0]['text'].strip()
            
            # Force final garbage collection
            gc.collect()
        
        return response.strip()
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        return f"Error generating text: {str(e)}"

@app.route('/')
def index():
    """Serve the model management interface."""
    return render_template('index.html')

@app.route('/load')
def load():
    """Serve the model loading interface."""
    return render_template('load.html', models=list(MODEL_CONFIGS.keys()))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok", 
        "model": "Dynamic Model", 
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
            "model": "dynamic-model",
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
    """OpenAI-compatible chat completions endpoint with context management."""
    try:
        data = request.get_json()
        
        if not data or 'messages' not in data:
            return jsonify({"error": {"message": "Missing 'messages' in request", "type": "invalid_request_error"}}), 400
        
        messages = data['messages']
        max_tokens = data.get('max_tokens', 4096)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        stream = data.get('stream', False)
        
        logger.info(f"Processing chat completion with {len(messages)} messages")
        
        if messages:
            # Maintain context by including the last few messages
            context_messages = messages[-5:]  # Keep last 5 messages for context
            formatted_prompt = format_prompt(context_messages)
            
            if stream:
                # Handle streaming response with context management
                def generate():
                    response = ""
                    current_tokens = 0
                    max_attempts = 5
                    attempt = 0
                    
                    while current_tokens < max_tokens and attempt < max_attempts:
                        attempt += 1
                        
                        # Generate a chunk of response with context
                        chunk = safe_generate(
                            formatted_prompt + response,
                            max_tokens=min(1024, max_tokens - current_tokens),
                            temperature=temperature,
                            top_p=top_p
                        )
                        
                        # Check if we have a complete response
                        if chunk and not chunk.endswith(('}', ';', '.', '?', '!')):
                            # Try to get more context
                            additional = safe_generate(
                                chunk,
                                max_tokens=256,
                                temperature=temperature,
                                top_p=top_p
                            )
                            chunk += additional
                        
                        # Add to the response
                        response += chunk
                        current_tokens += len(chunk.split())
                        
                        # Check if we have a complete response
                        if response and any(response.endswith(x) for x in ['}', ';', '.', '?', '!']):
                            break
                        
                        # Add a small delay to prevent overwhelming the model
                        time.sleep(0.1)
                    
                    # Split response into chunks for streaming
                    chunk_size = 50
                    for i in range(0, len(response), chunk_size):
                        chunk = response[i:i + chunk_size]
                        
                        # Remove padding words from the final chunk
                        if i + chunk_size >= len(response):
                            chunk = re.sub(r'\s*</?think\s*>', ' ', chunk)
                            chunk = re.sub(r'\s*</?code\s*>', ' ', chunk)
                            chunk = re.sub(r'\s*</?pre\s*>', ' ', chunk)
                            chunk = re.sub(r'\s*\\n\s*', ' ', chunk)
                            chunk = re.sub(r'\s*\\t\s*', ' ', chunk)
                        
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}, 'index': 0, 'finish_reason': None if i + chunk_size < len(response) else 'stop'}]})}\n\n"
                
                return Response(
                    generate(),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no'
                    }
                )
            else:
                # Handle non-streaming response with context
                response_text = safe_generate(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # Ensure the response is complete
                while not response_text.strip().endswith(('{', '}', ';', '.', '?', '!')):
                    additional = safe_generate(
                        response_text,
                        max_tokens=512,
                        temperature=temperature,
                        top_p=top_p
                    )
                    response_text += additional
                    time.sleep(0.1)
                
                # Format response in OpenAI format
                response = {
                    "id": f"chatcmpl-{os.urandom(4).hex()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "dynamic-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(formatted_prompt.split()),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(formatted_prompt.split()) + len(response_text.split())
                    }
                }
                
                return jsonify(response)
        
        return jsonify({"error": "No messages provided"}), 400
        
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

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models and their status."""
    available_models = {name: config["name"] for name, config in MODEL_CONFIGS.items()}
    
    loaded_models = list(MODELS.keys())
    active_model = ACTIVE_MODEL
    
    return jsonify({
        "available_models": available_models,
        "loaded_models": loaded_models,
        "active_model": active_model
    })

@app.route('/v1/models/<model_name>', methods=['POST'])
def switch_model(model_name):
    """Switch to a different model."""
    try:
        model = get_model(model_name)
        return jsonify({
            "status": "success",
            "message": f"Switched to model: {model_name}",
            "active_model": ACTIVE_MODEL
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/v1/models/<model_name>', methods=['DELETE'])
def remove_model(model_name):
    """Unload a specific model."""
    try:
        unload_model(model_name)
        return jsonify({
            "status": "success",
            "message": f"Model {model_name} unloaded",
            "active_model": ACTIVE_MODEL
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

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
        formatted_prompt = format_prompt(messages)
        
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
    logger.info("Starting Dynamic Inference Server on port 5020")
    # Pre-load the default model at startup
    try:
        model = get_model()
        logger.info(f"Model {ACTIVE_MODEL} loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load default model: {str(e)}")
    
    app.run(host='0.0.0.0', port=5020)
