# Start API server

echo "Starting API server..."
docker compose -f docker-compose.api.yaml down -t 0
docker compose -f docker-compose.api.yaml up
echo "API server started"