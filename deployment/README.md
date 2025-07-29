# Deployment Guide

This directory contains deployment configurations for the Multimodal Counterfactual Lab.

## Production Deployment

### Prerequisites

- Docker and Docker Compose installed
- SSL certificates for HTTPS (recommended)
- Domain name configured (for production)

### Quick Start

1. **Prepare Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Build and Deploy**
   ```bash
   # Build the application image
   make docker-build
   
   # Start production services
   docker-compose -f deployment/docker-compose.prod.yml up -d
   ```

3. **Verify Deployment**
   ```bash
   # Check service health
   curl http://localhost/healthz
   
   # View logs
   docker-compose -f deployment/docker-compose.prod.yml logs -f
   ```

### Configuration Files

#### docker-compose.prod.yml
Production-ready Docker Compose configuration with:
- Application container with resource limits
- Nginx reverse proxy with SSL termination
- Redis for caching (optional)
- Monitoring with Prometheus/Grafana (optional)

#### nginx.conf
Nginx configuration providing:
- SSL/TLS termination
- Rate limiting
- Security headers
- WebSocket support for Streamlit
- Static file caching
- Health check endpoints

### SSL/TLS Setup

1. **Using Let's Encrypt (Recommended)**
   ```bash
   # Install certbot
   sudo apt-get install certbot
   
   # Obtain certificate
   sudo certbot certonly --standalone -d your-domain.com
   
   # Copy certificates
   sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem deployment/ssl/cert.pem
   sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem deployment/ssl/key.pem
   ```

2. **Using Self-Signed Certificates (Development)**
   ```bash
   mkdir -p deployment/ssl
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout deployment/ssl/key.pem \
     -out deployment/ssl/cert.pem
   ```

### Monitoring Setup

Enable monitoring services:
```bash
docker-compose -f deployment/docker-compose.prod.yml --profile monitoring up -d
```

Access monitoring:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/changeme)

### Performance Tuning

#### Resource Limits
Adjust in `docker-compose.prod.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Adjust based on available CPU
      memory: 8G       # Adjust based on available RAM
```

#### Nginx Tuning
Modify `nginx.conf` for high traffic:
```nginx
worker_processes auto;
worker_connections 2048;
```

#### Application Scaling
Scale horizontally:
```bash
docker-compose -f deployment/docker-compose.prod.yml up -d --scale counterfactual-lab-prod=3
```

### Security Considerations

1. **Network Security**
   - Use a firewall to restrict access
   - Enable fail2ban for brute force protection
   - Regularly update Docker images

2. **Application Security**
   - Change default passwords
   - Enable audit logging
   - Implement proper authentication
   - Regular security scans with `make security-scan`

3. **Data Protection**
   - Encrypt data at rest
   - Use secure communication channels
   - Implement proper backup strategies
   - Follow data retention policies

### Backup and Recovery

#### Database Backup (if using)
```bash
# Backup Redis data
docker exec redis redis-cli BGSAVE
docker cp redis:/data/dump.rdb ./backups/redis-$(date +%Y%m%d).rdb
```

#### Application Data Backup
```bash
# Backup outputs and logs
tar -czf backup-$(date +%Y%m%d).tar.gz outputs/ logs/
```

#### Recovery Process
1. Stop services: `docker-compose down`
2. Restore data from backups
3. Restart services: `docker-compose up -d`
4. Verify functionality

### Troubleshooting

#### Common Issues

1. **SSL Certificate Errors**
   ```bash
   # Verify certificate validity
   openssl x509 -in deployment/ssl/cert.pem -text -noout
   ```

2. **High Memory Usage**
   ```bash
   # Monitor resource usage
   docker stats
   
   # Adjust memory limits in docker-compose.prod.yml
   ```

3. **Connection Issues**
   ```bash
   # Check service connectivity
   docker-compose -f deployment/docker-compose.prod.yml ps
   docker-compose -f deployment/docker-compose.prod.yml logs nginx
   ```

#### Log Analysis
```bash
# Application logs
docker-compose -f deployment/docker-compose.prod.yml logs counterfactual-lab-prod

# Nginx access logs
docker exec nginx tail -f /var/log/nginx/access.log

# System resource monitoring
docker exec counterfactual-lab-prod top
```

### Maintenance

#### Regular Tasks
- Update Docker images: `docker-compose pull && docker-compose up -d`
- Monitor disk usage: `df -h`
- Check logs for errors: `docker-compose logs --tail=100`
- Backup data: Run backup scripts
- Security updates: `make security-scan`

#### Health Checks
```bash
# Application health
curl -f http://localhost/healthz || echo "Service unhealthy"

# Docker health status
docker-compose -f deployment/docker-compose.prod.yml ps
```

### Support

For deployment issues:
- Check the troubleshooting section above
- Review application logs
- Consult the main documentation
- Open an issue on GitHub with deployment details