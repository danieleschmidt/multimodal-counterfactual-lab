# Production Deployment Guide

## Overview

This guide covers production deployment strategies for the Multimodal Counterfactual Lab.

## Deployment Architecture

### Recommended Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Interface │    │  API Gateway    │
│   (Nginx/ALB)   │───▶│   (Frontend)    │───▶│  (FastAPI)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 ▼                                 │
            ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
            │   Generation    │    │    Evaluation   │    │     Database    │    │     Redis       │
            │   Workers       │    │    Service      │    │   (PostgreSQL)  │    │   (Cache/Queue) │
            │   (GPU-enabled) │    │                 │    └─────────────────┘    └─────────────────┘
            └─────────────────┘    └─────────────────┘
                       │                                 │
            ┌─────────────────┐    ┌─────────────────┐    │
            │   Model Store   │    │   File Storage  │    │
            │   (S3/MinIO)    │    │   (S3/MinIO)    │    │
            └─────────────────┘    └─────────────────┘    │
                                                          │
                       ┌─────────────────────────────────────────────────────────┘
                       │
            ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
            │   Monitoring    │    │     Logging     │    │    Security     │
            │ (Prometheus)    │    │ (ELK/Loki)      │    │   (Vault/SSM)   │
            └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Container Deployment

### Docker Production Build

```dockerfile
# Dockerfile.prod
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set ownership and permissions
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["gunicorn", "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", "--bind", "0.0.0.0:8000", "counterfactual_lab.api:app"]
```

### Multi-stage Build for GPU Support

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as gpu-base

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link
RUN ln -s /usr/bin/python3.11 /usr/bin/python

FROM gpu-base as production

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install application
COPY pyproject.toml ./
RUN pip install -e .

COPY src/ ./src/

# Create non-root user
RUN groupadd -r app && useradd -r -g app app
RUN chown -R app:app /app
USER app

EXPOSE 8000

CMD ["python", "-m", "counterfactual_lab.server"]
```

## Kubernetes Deployment

### Namespace Configuration

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: counterfactual-lab
  labels:
    name: counterfactual-lab
    environment: production
```

### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: counterfactual-lab-config
  namespace: counterfactual-lab
data:
  DATABASE_URL: "postgresql://user:password@postgres:5432/counterfactual_lab"
  REDIS_URL: "redis://redis:6379/0"
  MODEL_CACHE_DIR: "/app/models"
  LOG_LEVEL: "INFO"
  PROMETHEUS_PORT: "9090"
```

### Secret Management

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: counterfactual-lab-secrets
  namespace: counterfactual-lab
type: Opaque
data:
  DATABASE_PASSWORD: <base64-encoded-password>
  HUGGINGFACE_TOKEN: <base64-encoded-token>
  SECRET_KEY: <base64-encoded-key>
```

### API Deployment

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: counterfactual-lab-api
  namespace: counterfactual-lab
spec:
  replicas: 3
  selector:
    matchLabels:
      app: counterfactual-lab-api
  template:
    metadata:
      labels:
        app: counterfactual-lab-api
    spec:
      containers:
      - name: api
        image: counterfactual-lab:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: counterfactual-lab-config
              key: DATABASE_URL
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: counterfactual-lab-secrets
              key: DATABASE_PASSWORD
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
```

### GPU Worker Deployment

```yaml
# k8s/gpu-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: counterfactual-lab-gpu-worker
  namespace: counterfactual-lab
spec:
  replicas: 2
  selector:
    matchLabels:
      app: counterfactual-lab-gpu-worker
  template:
    metadata:
      labels:
        app: counterfactual-lab-gpu-worker
    spec:
      containers:
      - name: gpu-worker
        image: counterfactual-lab:gpu-latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2000m"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4000m"
        env:
        - name: WORKER_TYPE
          value: "gpu"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      nodeSelector:
        accelerator: nvidia-tesla-v100
```

### Service Configuration

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: counterfactual-lab-api-service
  namespace: counterfactual-lab
spec:
  selector:
    app: counterfactual-lab-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: counterfactual-lab-ingress
  namespace: counterfactual-lab
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.counterfactual-lab.com
    secretName: counterfactual-lab-tls
  rules:
  - host: api.counterfactual-lab.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: counterfactual-lab-api-service
            port:
              number: 80
```

## Database Setup

### PostgreSQL Configuration

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: counterfactual-lab
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: counterfactual_lab
        - name: POSTGRES_USER
          value: user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: counterfactual-lab-secrets
              key: DATABASE_PASSWORD
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
```

### Database Migration Job

```yaml
# k8s/migration-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: counterfactual-lab-migration
  namespace: counterfactual-lab
spec:
  template:
    spec:
      containers:
      - name: migration
        image: counterfactual-lab:latest
        command: ["python", "-m", "alembic", "upgrade", "head"]
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: counterfactual-lab-config
              key: DATABASE_URL
      restartPolicy: OnFailure
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# k8s/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: counterfactual-lab
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'counterfactual-lab-api'
      static_configs:
      - targets: ['counterfactual-lab-api-service:80']
      metrics_path: /metrics
    - job_name: 'counterfactual-lab-workers'
      static_configs:
      - targets: ['counterfactual-lab-gpu-worker:9090']
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: counterfactual-lab
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-storage
          mountPath: /prometheus
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
      - name: prometheus-storage
        persistentVolumeClaim:
          claimName: prometheus-pvc
```

## Security Configuration

### Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: counterfactual-lab-network-policy
  namespace: counterfactual-lab
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - podSelector:
        matchLabels:
          app: counterfactual-lab-api
    ports:
    - protocol: TCP
      port: 5432
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 80   # HTTP
    - protocol: UDP
      port: 53   # DNS
```

### Pod Security Policy

```yaml
# k8s/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: counterfactual-lab-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## Scaling Configuration

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: counterfactual-lab-api-hpa
  namespace: counterfactual-lab
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: counterfactual-lab-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaler

```yaml
# k8s/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: counterfactual-lab-api-vpa
  namespace: counterfactual-lab
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: counterfactual-lab-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: api
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 128Mi
```

## Backup and Recovery

### Database Backup CronJob

```yaml
# k8s/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: counterfactual-lab
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:15
            command:
            - /bin/bash
            - -c
            - |
              export PGPASSWORD="$DATABASE_PASSWORD"
              pg_dump -h postgres -U user counterfactual_lab | \
              gzip > /backup/counterfactual_lab_$(date +%Y%m%d_%H%M%S).sql.gz
              
              # Keep only last 7 days of backups
              find /backup -name "*.sql.gz" -mtime +7 -delete
            env:
            - name: DATABASE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: counterfactual-lab-secrets
                  key: DATABASE_PASSWORD
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

## Load Testing

### K6 Load Test Configuration

```javascript
// load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 10 },
    { duration: '5m', target: 10 },
    { duration: '2m', target: 20 },
    { duration: '5m', target: 20 },
    { duration: '2m', target: 0 },
  ],
};

export default function() {
  let response = http.get('https://api.counterfactual-lab.com/health');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  // Test generation endpoint
  let payload = JSON.stringify({
    method: 'modicf',
    attributes: ['gender', 'age'],
    num_samples: 1
  });
  
  let params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  response = http.post('https://api.counterfactual-lab.com/generate', payload, params);
  check(response, {
    'generation status is 202': (r) => r.status === 202,
  });
  
  sleep(1);
}
```

## Deployment Checklist

### Pre-deployment

- [ ] Run full test suite
- [ ] Security scan passed
- [ ] Database migrations ready
- [ ] Environment variables configured
- [ ] Secrets properly encrypted
- [ ] Resource limits defined
- [ ] Monitoring configured
- [ ] Backup strategy in place

### Deployment

- [ ] Blue-green or canary deployment
- [ ] Database migration executed
- [ ] Health checks passing
- [ ] Load balancer configured
- [ ] SSL certificates valid
- [ ] Monitoring alerts active
- [ ] Rollback plan ready

### Post-deployment

- [ ] Application health verified
- [ ] Performance metrics normal
- [ ] Error rates acceptable
- [ ] User acceptance testing
- [ ] Documentation updated
- [ ] Team notified

## Troubleshooting Production Issues

### Common Issues

#### Pod Restart Loops
```bash
# Check pod logs
kubectl logs -f deployment/counterfactual-lab-api -n counterfactual-lab

# Check events
kubectl get events -n counterfactual-lab --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n counterfactual-lab
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it deployment/counterfactual-lab-api -n counterfactual-lab -- \
  python -c "import psycopg2; conn = psycopg2.connect('postgresql://...'); print('Connected')"

# Check database logs
kubectl logs statefulset/postgres -n counterfactual-lab
```

#### Performance Issues
```bash
# Check resource utilization
kubectl top nodes
kubectl top pods -n counterfactual-lab

# Scale up if needed
kubectl scale deployment/counterfactual-lab-api --replicas=5 -n counterfactual-lab
```

### Emergency Procedures

#### Quick Rollback
```bash
# Rollback to previous version
kubectl rollout undo deployment/counterfactual-lab-api -n counterfactual-lab

# Check rollback status
kubectl rollout status deployment/counterfactual-lab-api -n counterfactual-lab
```

#### Scale Down
```bash
# Scale down in emergency
kubectl scale deployment/counterfactual-lab-api --replicas=0 -n counterfactual-lab
```

This comprehensive production deployment guide ensures robust, scalable, and secure deployment of the Multimodal Counterfactual Lab.