Run from folder /deployment.

## Online

```
docker build -f ./deployment/online/Dockerfile.online -t module-5-online:latest .
docker run -p 5000:5000 --name module-5-online module-5-online
```

Powershell: 
```
Invoke-RestMethod -Uri "http://localhost:5000/api/predict" ` \
                  -Method POST ` \
                  -Headers @{ "Content-Type" = "application/json" } ` \
                  -Body '{ "text": "I love this product very much!" }'
```
Bash: 
```
curl -X POST "http://localhost:5000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{ "text": "I love this product very much!" }'
```

## Batch (Airflow)
```
docker-compose -f ./batch/docker-compose.yaml build \
docker-compose -f ./batch/docker-compose.yaml up
```
In the Airflow GUI run batch_pipeline.
