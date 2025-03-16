docker-compose up --build -d

docker compose up

run this cmd once docker is running
docker exec -it $(docker ps -qf "name=ollama") ollama pull codellama:7b-instruct

add to startup.sh
ollama pull codellama:7b-instruct


Pull a new model on demand - 
```docker exec -it ollama ollama pull llama2```



docker build -t ollama-nodejs-combined .
docker run -d --name ollama-combined -p 3000:3000 -p 11434:11434 ollama-nodejs-combined



Once the container is running, you can:

Access the Node.js API Gateway:

Health check: http://localhost:3000/health
List models: http://localhost:3000/models
Generate text: POST to http://localhost:3000/generate
Chat: POST to http://localhost:3000/chat
Access Ollama directly if needed:

http://localhost:11434/api/tags
Testing with Postman
In Postman, you can now make requests to:

Generate text:

POST http://localhost:3000/generate
Body (raw JSON):
{
  "model": "llama3.2:latest",
  "prompt": "Hello world"
}



Chat:

POST http://localhost:3000/chat
Body (raw JSON):
{
  "model": "llama3.2:latest",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ]
}




