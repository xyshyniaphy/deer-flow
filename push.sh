#!/bin/bash
set -e

# Uncomment the following line and run it once to log in to Docker Hub:
# docker login -u xyshyniaphy

echo "Pushing images to Docker Hub..."

docker push xyshyniaphy/deer-flow-backend:latest
docker push xyshyniaphy/deer-flow-frontend:latest

echo "Images pushed successfully."
