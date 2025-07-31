# Deployment Guide

This guide covers deployment strategies for the Multimodal Counterfactual Lab across different environments.

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- GPU with CUDA support (recommended for model inference)
- 8GB+ RAM (16GB+ recommended)
- 50GB+ free disk space for models and data

## 🚀 Quick Start

### Local Development

```bash
# Clone and setup
git clone <repository-url>
cd multimodal-counterfactual-lab
cp .env.example .env

# Install dependencies
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e ".[dev]"

# Run tests
make test

# Start development server
streamlit run app.py
```

### Docker Development

```bash
# Start development environment
docker-compose --profile dev up

# Or with all services (monitoring, databases)
docker-compose --profile full up
```

## 🏗️ Production Deployment

### Docker Production

```bash
# Build production image
docker build -t counterfactual-lab:latest .

# Run with production compose
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: counterfactual-lab
spec:
  replicas: 3
  selector:
    matchLabels:
      app: counterfactual-lab
  template:
    metadata:
      labels:
        app: counterfactual-lab
    spec:
      containers:
      - name: counterfactual-lab
        image: ghcr.io/your-org/counterfactual-lab:latest
        ports:
        - containerPort: 8501
        env:
        - name: COUNTERFACTUAL_LAB_ENV
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
```

### Cloud Platforms

#### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t counterfactual-lab .
docker tag counterfactual-lab:latest <account>.dkr.ecr.us-east-1.amazonaws.com/counterfactual-lab:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/counterfactual-lab:latest
```

#### Google Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy counterfactual-lab \
  --image gcr.io/PROJECT-ID/counterfactual-lab \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 2
```

#### Azure Container Instances

```bash
# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name counterfactual-lab \
  --image myregistry.azurecr.io/counterfactual-lab:latest \
  --cpu 2 \
  --memory 8 \
  --ports 8501
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COUNTERFACTUAL_LAB_ENV` | Environment (development/production) | development |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | 0 |
| `MODEL_CACHE_DIR` | Model storage directory | ./models |
| `OUTPUT_DIR` | Output directory | ./outputs |

### Scaling Configuration

#### Horizontal Scaling

```yaml
# docker-compose.prod.yml
services:
  counterfactual-lab:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 8G
          cpus: '2'
        reservations:
          memory: 4G
          cpus: '1'
```

#### Vertical Scaling

- **CPU**: 2-4 cores per instance
- **Memory**: 8-16GB per instance
- **GPU**: 1 GPU per instance (optional but recommended)
- **Storage**: 50-100GB for models and cache

### Load Balancing

#### Nginx Configuration

```nginx
upstream counterfactual_lab {
    server counterfactual-lab-1:8501;
    server counterfactual-lab-2:8501;
    server counterfactual-lab-3:8501;
}

server {
    listen 80;
    server_name counterfactual-lab.example.com;
    
    location / {
        proxy_pass http://counterfactual_lab;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🔧 Performance Optimization

### Model Optimization

```python
# Enable compilation for PyTorch 2.0+
import torch
model = torch.compile(model)

# Use half precision for inference
model = model.half()

# Enable memory-efficient attention
torch.backends.cuda.enable_flash_sdp(True)
```

### Caching Strategy

```yaml
# Redis configuration for model caching
redis:
  image: redis:7-alpine
  command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

### Resource Limits

```yaml
services:
  counterfactual-lab:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '4'
        reservations:
          memory: 8G
          cpus: '2'
```

## 📊 Monitoring

### Health Checks

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Metrics Collection

```python
# Enable Prometheus metrics
from prometheus_client import start_http_server, Counter, Histogram

generation_counter = Counter('generations_total', 'Total generations')
generation_duration = Histogram('generation_duration_seconds', 'Generation duration')

start_http_server(8080)  # Metrics endpoint
```

### Log Aggregation

```yaml
# Centralized logging with Fluentd
logging:
  driver: fluentd
  options:
    fluentd-address: localhost:24224
    tag: counterfactual-lab
```

## 🔒 Security

### Network Security

```yaml
# Restrict network access
networks:
  internal:
    driver: bridge
    internal: true
  external:
    driver: bridge
```

### Secrets Management

```bash
# Use Docker secrets for sensitive data
echo "your-api-key" | docker secret create huggingface-token -
```

### Image Security

```dockerfile
# Use non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Scan images regularly
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
```

## 🚨 Troubleshooting

### Common Issues

#### GPU Not Detected

```bash
# Check GPU availability
docker run --gpus all nvidia/cuda:11.8-runtime nvidia-smi

# Verify CUDA in container
python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues

```bash
# Monitor memory usage
docker stats counterfactual-lab

# Adjust memory limits
docker run -m 8g counterfactual-lab
```

#### Model Loading Failures

```bash
# Check model cache
ls -la ./models/

# Clear cache if corrupted
rm -rf ./models/.locks/
```

### Log Analysis

```bash
# View application logs
docker logs counterfactual-lab

# Follow logs in real-time
docker logs -f counterfactual-lab

# Filter error logs
docker logs counterfactual-lab 2>&1 | grep ERROR
```

## 📈 Scaling Guidelines

### Load Testing

```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8501
```

### Auto-scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: counterfactual-lab-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: counterfactual-lab
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Database Scaling

```yaml
# PostgreSQL with read replicas
postgres-primary:
  image: postgres:15
  environment:
    POSTGRES_REPLICATION_MODE: master

postgres-replica:
  image: postgres:15
  environment:
    POSTGRES_REPLICATION_MODE: slave
    POSTGRES_MASTER_HOST: postgres-primary
```

## 🔄 CI/CD Integration

### GitHub Actions Deployment

```yaml
- name: Deploy to production
  uses: azure/webapps-deploy@v2
  with:
    app-name: counterfactual-lab
    images: ghcr.io/your-org/counterfactual-lab:${{ github.sha }}
```

### Rolling Updates

```bash
# Zero-downtime deployment
docker service update --image counterfactual-lab:new-version counterfactual-lab
```

## 📋 Maintenance

### Regular Tasks

```bash
# Update dependencies monthly
pip-review --local --auto

# Clean up old images weekly
docker system prune -f

# Rotate logs daily
logrotate /etc/logrotate.d/counterfactual-lab
```

### Backup Strategy

```bash
# Backup models and data
tar -czf backup-$(date +%Y%m%d).tar.gz models/ data/ outputs/

# Upload to cloud storage
aws s3 cp backup-$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
```

For more deployment patterns and configurations, see the [Infrastructure as Code examples](./deployment/README.md).