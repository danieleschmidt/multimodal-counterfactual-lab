# Monitoring and Observability

This document covers the comprehensive monitoring and observability setup for the Multimodal Counterfactual Lab.

## Overview

Our monitoring stack provides complete visibility into:
- **Application Performance**: Request latency, throughput, error rates
- **Business Metrics**: User activity, generation rates, quality scores
- **Infrastructure Health**: CPU, memory, disk, network utilization
- **Model Performance**: Inference time, cache hit rates, quality drift
- **Security**: SSL certificates, vulnerability scanning, access patterns

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Stack                         │
├─────────────────────────────────────────────────────────────┤
│  Grafana Dashboards  │  AlertManager  │  External Systems   │
├─────────────────────────────────────────────────────────────┤
│                    Prometheus                               │
├─────────────────────────────────────────────────────────────┤
│  App Metrics  │  System Metrics  │  Custom Exporters       │
│  ├─ HTTP       │  ├─ Node         │  ├─ Blackbox           │
│  ├─ Generation│  ├─ Container     │  ├─ Redis              │
│  ├─ Models    │  ├─ GPU          │  └─ Postgres           │
│  └─ Business  │  └─ Network      │                         │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Prometheus
- **Purpose**: Metrics collection and storage
- **Configuration**: `monitoring/prometheus.yml`
- **Port**: 9090
- **Retention**: 30 days

### Grafana
- **Purpose**: Visualization and dashboards
- **Configuration**: `monitoring/grafana/`
- **Port**: 3000
- **Default credentials**: admin/admin (change in production)

### AlertManager
- **Purpose**: Alert routing and notification
- **Configuration**: `monitoring/alertmanager.yml`
- **Port**: 9093

### Exporters
- **Node Exporter**: System metrics (port 9100)
- **cAdvisor**: Container metrics (port 8080)
- **Blackbox Exporter**: External monitoring (port 9115)
- **Redis Exporter**: Redis metrics (port 9121)
- **Postgres Exporter**: Database metrics (port 9187)

## Key Metrics

### Application Metrics

#### Performance Metrics
```promql
# Request rate
sum(rate(http_requests_total[5m])) by (method, status)

# Average response time
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
```

#### Generation Metrics
```promql
# Generation rate
sum(rate(counterfactual_generation_total[5m])) by (method)

# Average generation time
rate(counterfactual_generation_duration_seconds_sum[5m]) / rate(counterfactual_generation_duration_seconds_count[5m])

# Success rate
sum(rate(counterfactual_generation_total{status="success"}[5m])) / sum(rate(counterfactual_generation_total[5m]))
```

#### Quality Metrics
```promql
# Average quality score
avg_over_time(counterfactual_quality_score[5m])

# Bias score
avg_over_time(bias_evaluation_score[5m])

# Model drift detection
abs(avg_over_time(counterfactual_quality_score[5m]) - avg_over_time(counterfactual_quality_score[5m] offset 24h))
```

### Infrastructure Metrics

#### System Resources
```promql
# CPU utilization
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory utilization
100 - ((node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)

# Disk utilization
100 - ((node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100)
```

#### Container Resources
```promql
# Container CPU usage
rate(container_cpu_usage_seconds_total[5m])

# Container memory usage
container_memory_usage_bytes

# Container restart count
increase(container_restart_count[1h])
```

### Business Metrics

#### User Activity
```promql
# Daily active users
count(increase(user_sessions_total[24h]))

# API usage rate
rate(api_requests_total[1h])

# Unique users per day
count by (day) (increase(user_sessions_total[24h]))
```

#### Financial Metrics
```promql
# API costs
sum(api_costs_total) by (service)

# Storage costs
sum(storage_costs_total) by (type)

# Compute costs
sum(compute_costs_total) by (instance_type)
```

## Dashboards

### Main Application Dashboard
- **File**: `monitoring/grafana/dashboards/application.json`
- **Panels**:
  - Request rate and latency
  - Error rate trends
  - Generation performance
  - Quality scores
  - Resource utilization

### Infrastructure Dashboard
- **File**: `monitoring/grafana/dashboards/infrastructure.json`
- **Panels**:
  - System resource usage
  - Container metrics
  - Network traffic
  - Disk I/O

### Business Dashboard
- **File**: `monitoring/grafana/dashboards/business.json`
- **Panels**:
  - User activity
  - API usage
  - Cost tracking
  - Feature adoption

### Model Performance Dashboard
- **File**: `monitoring/grafana/dashboards/models.json`
- **Panels**:
  - Inference latency
  - Model accuracy
  - Cache performance
  - GPU utilization

## Alerting

### Alert Severity Levels

1. **Critical**: Immediate action required (paging)
   - Service completely down
   - Data loss risk
   - Security breaches

2. **Warning**: Action required within hours
   - Performance degradation
   - Resource exhaustion risk
   - Quality degradation

3. **Info**: Awareness alerts
   - Unusual but not critical patterns
   - Capacity planning alerts

### Alert Rules

#### Critical Alerts
```yaml
- alert: ServiceDown
  expr: up{job="counterfactual-lab"} == 0
  for: 1m
  labels:
    severity: critical

- alert: HighErrorRate
  expr: cf_lab:error_rate:rate5m > 0.1
  for: 2m
  labels:
    severity: critical
```

#### Warning Alerts
```yaml
- alert: HighMemoryUsage
  expr: cf_lab:memory_usage_bytes / 1024 / 1024 / 1024 > 3
  for: 5m
  labels:
    severity: warning

- alert: ModelGenerationLatency
  expr: cf_lab:generation_duration_seconds:rate5m > 30
  for: 5m
  labels:
    severity: warning
```

### Alert Routing

Configure AlertManager for different notification channels:

```yaml
# Slack for team notifications
- name: 'slack-alerts'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK'
    channel: '#alerts'
    title: 'Alert: {{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

# PagerDuty for critical alerts
- name: 'pagerduty'
  pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_KEY'
    description: '{{ .GroupLabels.alertname }}: {{ .GroupLabels.instance }}'
```

## Service Level Objectives (SLOs)

### Availability SLO: 99.9%
- **Error Budget**: 43.2 minutes/month
- **Monitoring**: `cf_lab:availability:rate5m`
- **Alerting**: Error budget burn rate

### Latency SLO: 95% < 2s, 99% < 5s
- **Monitoring**: 
  - `cf_lab:latency_95th:5m`
  - `cf_lab:latency_99th:5m`
- **Alerting**: SLO violation for > 5 minutes

### Generation Success Rate SLO: 95%
- **Monitoring**: `cf_lab:generation_success_rate:rate5m`
- **Alerting**: Below 95% for > 5 minutes

### Quality Score SLO: Average > 0.8
- **Monitoring**: `cf_lab:quality_score:avg5m`
- **Alerting**: Below 0.8 for > 10 minutes

## Operational Procedures

### Daily Monitoring Checklist
- [ ] Check overnight alerts
- [ ] Verify all services healthy
- [ ] Review performance trends
- [ ] Check error rate patterns
- [ ] Validate backup completion
- [ ] Review security alerts

### Weekly Monitoring Tasks
- [ ] Review SLO compliance
- [ ] Analyze capacity trends
- [ ] Update alert thresholds
- [ ] Review dashboard effectiveness
- [ ] Plan capacity upgrades
- [ ] Test disaster recovery

### Monthly Monitoring Tasks
- [ ] Review and update SLOs
- [ ] Analyze cost trends
- [ ] Update monitoring documentation
- [ ] Review alert fatigue
- [ ] Plan monitoring improvements
- [ ] Security review

## Troubleshooting

### Common Issues

#### High Memory Usage
1. Check application metrics for memory leaks
2. Review model cache settings
3. Analyze garbage collection patterns
4. Consider scaling horizontally

#### High Error Rate
1. Check application logs for error patterns
2. Verify database connectivity
3. Check external service dependencies
4. Review recent deployments

#### Slow Response Times
1. Analyze request patterns
2. Check database query performance
3. Review model inference times
4. Validate cache hit rates

### Monitoring the Monitoring

#### Prometheus Health
```bash
# Check Prometheus status
curl http://prometheus:9090/-/healthy

# Check configuration
curl http://prometheus:9090/api/v1/status/config

# Check targets
curl http://prometheus:9090/api/v1/targets
```

#### Grafana Health
```bash
# Check Grafana status
curl http://grafana:3000/api/health

# Check datasource connectivity
curl -u admin:admin http://grafana:3000/api/datasources/proxy/1/api/v1/query?query=up
```

## Performance Optimization

### Metrics Cardinality
- Limit label values to prevent high cardinality
- Use recording rules for expensive queries
- Set appropriate retention policies

### Query Optimization
- Use recording rules for frequently accessed metrics
- Implement proper time range selection
- Cache dashboard queries where possible

### Storage Optimization
- Configure appropriate retention periods
- Use downsampling for long-term storage
- Monitor Prometheus storage usage

## Security Considerations

### Access Control
- Implement authentication for Grafana
- Restrict Prometheus access to internal networks
- Use HTTPS for all monitoring endpoints

### Data Privacy
- Avoid collecting sensitive user data in metrics
- Implement data retention policies
- Use metric relabeling to remove sensitive labels

### Network Security
- Segment monitoring network
- Use TLS for inter-service communication
- Implement firewall rules

## Integration

### External Systems

#### AWS CloudWatch
```yaml
# Send metrics to CloudWatch
remote_write:
  - url: https://aps-workspaces.us-east-1.amazonaws.com/workspaces/ws-xxx/api/v1/remote_write
    sigv4:
      region: us-east-1
```

#### DataDog
```yaml
# Send metrics to DataDog
remote_write:
  - url: https://api.datadoghq.com/api/v1/series
    basic_auth:
      password: YOUR_API_KEY
```

### CI/CD Integration
- Monitor deployment metrics
- Track deployment frequency
- Measure change failure rate
- Monitor mean time to recovery

## Best Practices

### Metric Design
1. **Use consistent naming**: Follow Prometheus naming conventions
2. **Implement proper labels**: Use labels for dimensions, not values
3. **Avoid high cardinality**: Limit unique label combinations
4. **Use counters for totals**: Use gauges for current values

### Dashboard Design
1. **Start with USE method**: Utilization, Saturation, Errors
2. **Include SLI/SLO tracking**: Show compliance with objectives
3. **Add context**: Include annotations and links to runbooks
4. **Optimize for different audiences**: Separate technical and business views

### Alert Design
1. **Alert on symptoms**: Focus on user-impacting issues
2. **Include actionable information**: What to do when alert fires
3. **Set appropriate thresholds**: Balance sensitivity and noise
4. **Test alert rules**: Verify alerts fire when expected

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [SRE Workbook](https://sre.google/workbook/)
- [Monitoring Best Practices](https://prometheus.io/docs/practices/)

For monitoring support, contact the SRE team or open an issue in the monitoring repository.