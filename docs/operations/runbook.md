# Operational Runbook

## Overview

This runbook provides operational procedures for maintaining, monitoring, and troubleshooting the Multimodal Counterfactual Lab in production environments.

## Service Architecture

### Core Components
- **API Server**: Main application serving counterfactual generation requests
- **Model Engine**: ML model inference and processing
- **Data Pipeline**: Input preprocessing and output postprocessing  
- **Monitoring Stack**: Prometheus, Grafana, Alertmanager
- **Security Layer**: Authentication, authorization, input validation

### Dependencies
- **External APIs**: Hugging Face Hub, cloud storage services
- **Databases**: Redis for caching, PostgreSQL for metadata
- **Infrastructure**: Docker, Kubernetes, cloud provider services

## Deployment Procedures

### Production Deployment Checklist

- [ ] **Pre-deployment**
  - [ ] Code review completed and approved
  - [ ] All tests passing (unit, integration, e2e)
  - [ ] Security scan results reviewed
  - [ ] Performance benchmarks validated
  - [ ] Database migrations prepared
  - [ ] Rollback plan documented

- [ ] **Deployment**  
  - [ ] Blue-green deployment initiated
  - [ ] Health checks passing
  - [ ] Monitoring dashboards updated
  - [ ] Load balancer configured
  - [ ] SSL certificates validated

- [ ] **Post-deployment**
  - [ ] Smoke tests executed
  - [ ] Performance metrics within acceptable ranges
  - [ ] Error rates < 0.1%
  - [ ] Response times < 95th percentile SLA
  - [ ] Security monitoring active

### Rollback Procedures

1. **Immediate Rollback** (< 5 minutes)
   ```bash
   # Switch load balancer to previous version
   kubectl patch service counterfactual-lab -p '{"spec":{"selector":{"version":"v1.0"}}}'
   
   # Verify health
   kubectl get pods -l app=counterfactual-lab
   ```

2. **Database Rollback** (if needed)
   ```bash
   # Run rollback migration
   python manage.py migrate app_name 0001_previous_migration
   ```

3. **Monitoring Verification**
   - Check error rates return to baseline
   - Verify response times improve
   - Confirm no data corruption

## Monitoring and Alerting

### Key Metrics

#### Application Metrics
- **Request Rate**: `rate(http_requests_total[5m])`
- **Response Time**: `histogram_quantile(0.95, http_request_duration_seconds_bucket)`
- **Error Rate**: `rate(http_requests_total{status=~"5.."}[5m])`
- **Model Inference Time**: `histogram_quantile(0.95, model_inference_duration_seconds_bucket)`

#### Infrastructure Metrics
- **CPU Usage**: `rate(container_cpu_usage_seconds_total[5m]) * 100`
- **Memory Usage**: `container_memory_usage_bytes / container_spec_memory_limit_bytes * 100`
- **Disk Usage**: `node_filesystem_avail_bytes / node_filesystem_size_bytes * 100`
- **Network I/O**: `rate(container_network_receive_bytes_total[5m])`

#### Business Metrics
- **Bias Detection Rate**: `rate(bias_threshold_exceeded_total[1h])`
- **Counterfactual Quality Score**: `avg(counterfactual_quality_score)`
- **User Satisfaction**: `avg(user_rating_score)`

### Alert Response Procedures

#### Critical Alerts (P0 - Response within 15 minutes)

**Service Down**
1. Check load balancer and DNS resolution
2. Verify pod status: `kubectl get pods -l app=counterfactual-lab`
3. Check application logs: `kubectl logs -f deployment/counterfactual-lab`
4. If persistent, initiate rollback procedure

**High Error Rate (>5%)**
1. Identify error patterns in logs
2. Check downstream dependencies
3. Verify model loading and GPU availability
4. Scale up resources if needed: `kubectl scale deployment counterfactual-lab --replicas=5`

**Security Breach**
1. Immediately isolate affected components
2. Notify security team
3. Enable enhanced logging
4. Prepare incident report

#### Warning Alerts (P1 - Response within 1 hour)

**High Response Time**
1. Check resource utilization
2. Analyze slow query logs
3. Consider scaling or optimization
4. Monitor for improvement

**Memory Usage High**
1. Check for memory leaks in application
2. Verify model memory usage
3. Consider increasing resource limits
4. Monitor garbage collection

## Troubleshooting Guide

### Common Issues

#### Model Loading Failures
**Symptoms**: 500 errors, "Model not found" messages
**Resolution**:
```bash
# Check model files
ls -la /app/models/
# Verify model permissions
chown app:app /app/models/*
# Restart model service
kubectl rollout restart deployment/counterfactual-lab
```

#### High Inference Latency
**Symptoms**: >10s response times, timeout errors
**Investigation**:
```bash
# Check GPU utilization
nvidia-smi
# Monitor model performance
curl -s http://localhost:8001/model-metrics | grep inference_time
# Check batch processing
tail -f /app/logs/inference.log
```

#### Memory Leaks
**Symptoms**: Gradually increasing memory usage, OOM kills
**Resolution**:
```bash
# Enable memory profiling
export PYTHONMALLOC=debug
# Analyze memory usage
python -m memory_profiler src/counterfactual_lab/core.py
# Restart pods on schedule
kubectl create cronjob pod-restart --schedule="0 2 * * *" --image=bitnami/kubectl
```

### Performance Optimization

#### Model Optimization
- **Quantization**: Reduce model precision for faster inference
- **Batch Processing**: Group requests for efficient GPU utilization
- **Model Caching**: Cache frequent model outputs
- **GPU Optimization**: Use CUDA streams and memory pools

#### Infrastructure Optimization
- **Auto-scaling**: Configure HPA based on CPU/memory/custom metrics
- **Resource Requests**: Set appropriate CPU/memory requests and limits
- **Node Affinity**: Schedule GPU workloads on appropriate nodes
- **Caching Layer**: Implement Redis for frequent queries

## Security Operations

### Security Monitoring

#### Real-time Monitoring
- **Intrusion Detection**: Monitor for unusual access patterns
- **Vulnerability Scanning**: Daily scans of dependencies and containers
- **Anomaly Detection**: ML-based detection of unusual behavior
- **Compliance Checking**: Automated policy validation

#### Incident Response
1. **Detection**: Automated alerts and manual reporting
2. **Classification**: Severity assessment and team notification
3. **Containment**: Isolate affected systems
4. **Investigation**: Root cause analysis and evidence collection
5. **Recovery**: System restoration and validation
6. **Post-incident**: Documentation and process improvement

### Backup and Recovery

#### Data Backup Strategy
- **Model Artifacts**: Daily backup to cloud storage
- **Configuration**: Version-controlled in Git
- **Logs**: 30-day retention in centralized logging
- **Metrics**: 1-year retention in time-series database

#### Recovery Procedures
```bash
# Restore from backup
aws s3 sync s3://backup-bucket/models/ /app/models/
# Verify model integrity
python scripts/verify_models.py
# Restart services
docker-compose up -d
```

## Capacity Planning

### Resource Forecasting
- **Traffic Growth**: Plan for 20% monthly growth
- **Model Size**: Account for larger models and ensemble methods
- **Data Storage**: Plan for dataset growth and archival
- **Compute Resources**: GPU scaling based on inference demand

### Scaling Guidelines

#### Horizontal Scaling
- **CPU-bound**: Scale API server replicas
- **Memory-bound**: Use larger instance types
- **GPU-bound**: Add GPU nodes to cluster
- **Storage-bound**: Implement distributed storage

#### Vertical Scaling
- **Memory**: Increase per-pod memory limits
- **CPU**: Upgrade to higher-performance instances
- **Network**: Upgrade network bandwidth
- **Storage**: Use faster SSD storage

## Maintenance Windows

### Scheduled Maintenance
- **Weekly**: Dependency updates and security patches
- **Monthly**: OS updates and configuration review
- **Quarterly**: Performance optimization and capacity review
- **Annually**: Full security audit and disaster recovery test

### Maintenance Procedures
1. **Pre-maintenance**: Notify users, prepare rollback plan
2. **Maintenance**: Execute changes in staging first
3. **Validation**: Run comprehensive test suite
4. **Post-maintenance**: Monitor for issues, update documentation

## Contact Information

### On-call Rotation
- **Primary**: daniel@terragon.ai (24/7)
- **Secondary**: team@terragon.ai
- **Escalation**: security@terragon.ai

### Emergency Contacts
- **Critical Issues**: +1-xxx-xxx-xxxx
- **Security Incidents**: security-emergency@terragon.ai
- **Vendor Support**: Available in team documentation

## Runbook Maintenance

This runbook should be reviewed and updated:
- After every incident
- Monthly during team meetings  
- When significant system changes are made
- Based on operational lessons learned

**Last Updated**: 2025-01-01  
**Next Review**: 2025-02-01  
**Owner**: Platform Team