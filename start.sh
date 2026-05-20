#!/bin/bash
source .env

if [ "$OLLAMA_MODE" = "docker" ]; then
  echo "Modo Docker: levantando contenedor Ollama..."
  export OLLAMA_BASE_URL=http://ollama:11434
  docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build
else
  echo "Modo local: usando Ollama del host..."
  export OLLAMA_BASE_URL=http://host.docker.internal:11434
  docker compose up --build
fi
