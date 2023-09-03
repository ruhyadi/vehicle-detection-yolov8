# Execute train docker container
# usage: bash scripts/exec_docker.sh
echo "Execute docker container"
docker compose -f docker-compose.train.yaml down -t 0
docker compose -f docker-compose.train.yaml up -d \
    && docker exec -it vehicle-yolov8 bash \
    &&
echo "Docker container executed"