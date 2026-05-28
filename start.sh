#!/bin/bash
source .env

if [ "$OLLAMA_MODE" = "docker" ]; then
  echo "Docker mode: Ollama container is starting..."
  export OLLAMA_BASE_URL=http://ollama:11434
  docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build
else
  echo "Local mode: Ollama from host is starting..."
  export OLLAMA_BASE_URL=http://host.docker.internal:11434
  docker compose up --build
fi
