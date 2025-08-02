# Deployment Guide

This guide covers deployment options for the Multimodal Counterfactual Lab, from local development to production environments.

## Overview

The application supports multiple deployment methods:
- **Local Development**: Docker Compose for development
- **Production**: Docker Compose with security hardening
- **Kubernetes**: Cloud-native deployment (see kubernetes/)
- **Cloud Platforms**: AWS, Azure, GCP specific configurations

## Quick Start

### Development Deployment

```bash
# Clone repository
git clone https://github.com/terragon-labs/multimodal-counterfactual-lab.git
cd multimodal-counterfactual-lab

# Start development environment
docker-compose up -d

# Access application
open http://localhost:8501
```

### Production Deployment

```bash
# Build production image
./scripts/build.sh --type production

# Deploy with production configuration
docker-compose -f docker-compose.prod.yml up -d

# Access application
open https://localhost
```

## Build System

### Build Script

The `scripts/build.sh` script provides comprehensive build automation:

```bash
# Development build
./scripts/build.sh

# Production build with push
./scripts/build.sh --type production --push

# Security-hardened build with SBOM
./scripts/build.sh --type security --sbom --security-scan

# Clean build without cache
./scripts/build.sh --clean --no-cache
```

### Build Options

| Option | Description | Example |
|--------|-------------|---------|
| `--type` | Build type (development, production, security) | `--type production` |
| `--version` | Version tag | `--version v1.0.0` |
| `--push` | Push to registry | `--push` |
| `--registry` | Container registry | `--registry ghcr.io/myorg` |
| `--sbom` | Generate SBOM | `--sbom` |
| `--security-scan` | Run security scans | `--security-scan` |
| `--clean` | Clean before build | `--clean` |
| `--no-cache` | Build without cache | `--no-cache` |
| `--no-tests` | Skip tests | `--no-tests` |

## Container Images

### Image Variants

1. **Development Image** (`Dockerfile`, target: `builder`)
   - Includes development tools
   - Jupyter notebook support
   - Hot reloading enabled
   - Larger size (~2GB)

2. **Production Image** (`Dockerfile`, target: `production`)
   - Minimal runtime dependencies
   - Security optimizations
   - Smaller size (~800MB)

3. **Security-Hardened Image** (`Dockerfile.security`)
   - Maximum security hardening
   - Non-root user
   - Read-only filesystem
   - Distroless base (~600MB)

### Image Security Features

- **Non-root user**: All images run as non-privileged user
- **Minimal attack surface**: Only necessary packages installed
- **Security scanning**: Integrated vulnerability scanning
- **SBOM generation**: Software Bill of Materials included
- **Multi-stage builds**: Separate build and runtime environments

## Docker Compose Configurations

### Development (docker-compose.yml)

Features:
- Hot reloading
- Development tools
- Jupyter notebook access
- Volume mounts for development

```bash
# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f

# Stop environment
docker-compose down
```

### Production (docker-compose.prod.yml)

Features:
- Security hardening
- Resource limits
- Monitoring stack (Prometheus, Grafana)
- Load balancing (Nginx)
- Redis caching

```bash
# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# Scale application
docker-compose -f docker-compose.prod.yml up -d --scale counterfactual-lab-prod=3

# Stop production environment
docker-compose -f docker-compose.prod.yml down
```

## Environment Configuration

### Environment Variables

Core application settings:

```bash
# Application
COUNTERFACTUAL_LAB_ENV=production
ENABLE_TELEMETRY=true
LOG_LEVEL=INFO

# Models
HUGGINGFACE_TOKEN=your_token_here
DEFAULT_DIFFUSION_MODEL=stabilityai/stable-diffusion-2-1

# Performance
WORKERS=2
MAX_REQUESTS=1000
TIMEOUT=30

# Security
SECRET_KEY=your-secret-key
API_RATE_LIMIT=100

# Database
DATABASE_URL=postgresql://user:pass@db:5432/cflab
REDIS_URL=redis://redis:6379/0

# Monitoring
PROMETHEUS_ENDPOINT=http://prometheus:9090
GRAFANA_ENDPOINT=http://grafana:3000
```

### Secrets Management

For production, use proper secrets management:

```bash
# Docker Swarm
echo "your-secret" | docker secret create db_password -

# Kubernetes
kubectl create secret generic app-secrets --from-literal=db-password=your-secret

# Cloud providers
# AWS: AWS Secrets Manager
# Azure: Azure Key Vault
# GCP: Google Secret Manager
```

## Security Considerations

### Container Security

1. **Non-root execution**: All containers run as non-privileged users
2. **Read-only filesystems**: Containers use read-only root filesystems
3. **Security contexts**: Proper security contexts applied
4. **Resource limits**: CPU and memory limits enforced
5. **Network policies**: Restricted network access

### Image Security

1. **Base image scanning**: Regular security scans of base images
2. **Dependency scanning**: Automated scanning of Python dependencies
3. **Vulnerability patching**: Regular updates for security patches
4. **SBOM generation**: Software Bill of Materials for compliance

### Runtime Security

1. **TLS encryption**: All communications encrypted in transit
2. **Authentication**: Proper authentication mechanisms
3. **Authorization**: Role-based access control
4. **Audit logging**: Comprehensive audit trails
5. **Monitoring**: Security monitoring and alerting

## Performance Optimization

### Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Application | 1-2 cores | 2-4GB | 10GB |
| Redis | 0.5 cores | 512MB | 1GB |
| Prometheus | 0.5 cores | 1GB | 20GB |
| Grafana | 0.2 cores | 256MB | 1GB |

### Scaling Strategies

1. **Horizontal scaling**: Multiple application instances
2. **Load balancing**: Nginx or cloud load balancers
3. **Caching**: Redis for model and data caching
4. **Database optimization**: Connection pooling and indexing

### Performance Monitoring

Monitor these key metrics:
- Response time
- Throughput (requests/second)
- Error rate
- Resource utilization (CPU, memory, disk)
- Model inference time

## Troubleshooting

### Common Issues

1. **Out of memory errors**
   ```bash
   # Increase memory limits in docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 8G
   ```

2. **Model download failures**
   ```bash
   # Check HuggingFace token
   docker-compose logs counterfactual-lab | grep "authentication"
   ```

3. **Permission errors**
   ```bash
   # Check file permissions
   docker-compose exec counterfactual-lab ls -la /app
   ```

4. **Network connectivity issues**
   ```bash
   # Check network configuration
   docker network ls
   docker network inspect multimodal-counterfactual-lab_default
   ```

### Logging

Application logs are available through Docker:

```bash
# View all logs
docker-compose logs

# Follow logs for specific service
docker-compose logs -f counterfactual-lab

# View logs with timestamps
docker-compose logs -t counterfactual-lab
```

### Health Checks

Health check endpoints:

```bash
# Application health
curl http://localhost:8501/health

# Metrics endpoint
curl http://localhost:8501/metrics

# Ready check
curl http://localhost:8501/ready
```

## Backup and Recovery

### Data Backup

Important data to backup:
- Model cache (`/app/models`)
- Application data (`/app/data`)
- User outputs (`/app/outputs`)
- Configuration files
- Secrets and certificates

```bash
# Create backup
docker run --rm -v $(pwd):/backup -v counterfactual-lab_outputs:/data alpine tar czf /backup/data-backup-$(date +%Y%m%d).tar.gz /data

# Restore backup
docker run --rm -v $(pwd):/backup -v counterfactual-lab_outputs:/data alpine tar xzf /backup/data-backup-20241201.tar.gz -C /
```

### Disaster Recovery

1. **Regular backups**: Automated daily backups
2. **Multi-region deployment**: Deploy in multiple regions
3. **Database replication**: Master-slave database setup
4. **Infrastructure as Code**: Version controlled infrastructure
5. **Recovery testing**: Regular recovery procedure testing

## Production Checklist

Before deploying to production:

- [ ] Security scan passed
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Secrets properly managed
- [ ] SSL certificates configured
- [ ] Load balancing configured
- [ ] Auto-scaling policies set
- [ ] Disaster recovery plan tested
- [ ] Documentation updated
- [ ] Team trained on operations

## Support

For deployment support:
- Check the [troubleshooting guide](troubleshooting.md)
- Review [monitoring documentation](monitoring.md)
- Open an issue on [GitHub](https://github.com/terragon-labs/multimodal-counterfactual-lab/issues)
- Contact the team at support@terragon.ai