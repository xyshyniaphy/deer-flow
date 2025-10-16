#!/bin/bash
set -e

echo "Building Docker images..."

# Build backend image
docker build -t xyshyniaphy/deer-flow-backend:latest -f Dockerfile .

# Build frontend image
docker build --build-arg NEXT_PUBLIC_API_URL=https://df.198066.xyz/api -t xyshyniaphy/deer-flow-frontend:latest -f web/Dockerfile ./web

echo "Images built successfully."
echo "Next, you can push them to Docker Hub with push.sh"
