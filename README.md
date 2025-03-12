docker compose up

run this cmd once docker is running
docker exec -it $(docker ps -qf "name=ollama") ollama pull codellama:7b-instruct

add to startup.sh
ollama pull codellama:7b-instruct
