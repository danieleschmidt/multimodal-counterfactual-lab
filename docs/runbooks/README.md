# Operational Runbooks

This directory contains step-by-step operational procedures for common scenarios in the Multimodal Counterfactual Lab.

## Runbook Structure

Each runbook follows this template:

```markdown
# [Issue Name] Runbook

## Summary
Brief description of the issue and its impact.

## Symptoms
- What users experience
- What monitoring shows
- Common error messages

## Investigation
Step-by-step troubleshooting process.

## Resolution
Steps to fix the issue.

## Prevention
How to prevent this issue in the future.

## Escalation
When and how to escalate.
```

## Available Runbooks

### Critical Issues
- [Service Down](service-down.md) - Complete service outage
- [High Error Rate](high-error-rate.md) - Elevated error rates
- [Database Issues](database-issues.md) - Database connectivity problems
- [Security Incidents](security-incidents.md) - Security breach response

### Performance Issues
- [High Memory Usage](high-memory.md) - Memory exhaustion
- [Slow Generation](slow-generation.md) - Model generation latency
- [Quality Degradation](quality-degradation.md) - Output quality decline
- [Bias Detection](bias-detection.md) - Fairness issues

### Infrastructure Issues
- [Container Restart Loop](container-restart.md) - Frequent container restarts
- [Storage Full](storage-full.md) - Disk space exhaustion
- [Network Issues](network-issues.md) - Connectivity problems
- [SSL Certificate Expiry](ssl-cert-expiry.md) - Certificate renewal

### Business Issues
- [Low User Activity](low-users.md) - Declining user engagement
- [API Rate Limiting](api-limits.md) - Rate limit exceeded
- [Cost Overruns](cost-overruns.md) - Budget threshold exceeded

## On-Call Procedures

### Alert Response Process

1. **Acknowledge Alert** (within 5 minutes)
   - Check monitoring dashboards
   - Assess impact and severity
   - Acknowledge in alerting system

2. **Initial Assessment** (within 10 minutes)
   - Check service status
   - Review recent changes
   - Identify affected components

3. **Communicate** (within 15 minutes)
   - Notify team if severity is high
   - Update status page if needed
   - Document actions taken

4. **Investigate and Resolve** (time varies)
   - Follow relevant runbook
   - Implement fix or workaround
   - Monitor for resolution

5. **Post-Incident** (within 24 hours)
   - Write incident report
   - Identify root cause
   - Plan preventive measures

### Escalation Matrix

| Severity | Response Time | Escalation Path |
|----------|---------------|-----------------|
| Critical | 5 minutes | Primary → Secondary → Manager |
| High | 15 minutes | Primary → Secondary |
| Medium | 1 hour | Primary on-call |
| Low | Next business day | Assign to team |

### Contact Information

```yaml
# Primary On-Call
- Name: [Current Primary]
- Phone: [Phone Number]
- Slack: @[username]

# Secondary On-Call
- Name: [Current Secondary]
- Phone: [Phone Number]
- Slack: @[username]

# Manager
- Name: [Manager Name]
- Phone: [Phone Number]
- Email: [manager@company.com]
```

## Common Tools and Commands

### Health Checks
```bash
# Application health
curl -f http://localhost:8080/health

# Database connectivity
pg_isready -h localhost -p 5432

# Redis connectivity
redis-cli ping

# Container status
docker ps --filter "status=running"
```

### Log Analysis
```bash
# Application logs
docker logs counterfactual-lab-prod --tail 100 -f

# System logs
journalctl -u docker.service -f

# Nginx logs
tail -f /var/log/nginx/access.log
```

### Monitoring Queries
```bash
# High-level service health
curl 'http://prometheus:9090/api/v1/query?query=up{job="counterfactual-lab"}'

# Error rate
curl 'http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])'

# Memory usage
curl 'http://prometheus:9090/api/v1/query?query=process_resident_memory_bytes'
```

### Emergency Procedures

#### Service Restart
```bash
# Graceful restart
docker-compose restart counterfactual-lab-prod

# Force restart if unresponsive
docker-compose stop counterfactual-lab-prod
docker-compose up -d counterfactual-lab-prod
```

#### Rollback Deployment
```bash
# Check recent deployments
docker images counterfactual-lab --format "table {{.Tag}}\t{{.CreatedAt}}"

# Rollback to previous version
docker tag counterfactual-lab:previous counterfactual-lab:latest
docker-compose up -d counterfactual-lab-prod
```

#### Emergency Scaling
```bash
# Scale up application instances
docker-compose up -d --scale counterfactual-lab-prod=3

# Scale down if resource constrained
docker-compose up -d --scale counterfactual-lab-prod=1
```

## Incident Management

### Incident Categories

**SEV-1 (Critical)**
- Complete service outage
- Data loss or corruption
- Security breach
- Response: Immediate (5 minutes)

**SEV-2 (High)**
- Significant performance degradation
- Feature unavailable
- High error rates
- Response: 15 minutes

**SEV-3 (Medium)**
- Minor performance issues
- Non-critical feature issues
- Monitoring alerts
- Response: 1 hour

**SEV-4 (Low)**
- Cosmetic issues
- Enhancement requests
- Documentation updates
- Response: Next business day

### Communication Templates

#### Initial Response
```
🚨 INCIDENT ALERT 🚨

Severity: [SEV-X]
Service: Counterfactual Lab
Issue: [Brief description]
Impact: [User impact]
Status: Investigating
Owner: [On-call engineer]
Started: [Timestamp]
```

#### Update Template
```
📊 INCIDENT UPDATE 📊

Severity: [SEV-X]
Issue: [Brief description]
Update: [What was discovered/attempted]
Next Step: [What's being tried next]
ETA: [Expected resolution time]
Owner: [Current owner]
```

#### Resolution Template
```
✅ INCIDENT RESOLVED ✅

Severity: [SEV-X]
Issue: [Brief description]
Resolution: [How it was fixed]
Duration: [Total time]
Root Cause: [If known]
Follow-up: [Tracking item for post-mortem]
```

## Monitoring Dashboard Links

- [Application Overview](http://grafana:3000/d/app-overview)
- [Infrastructure Health](http://grafana:3000/d/infra-health)
- [Model Performance](http://grafana:3000/d/model-perf)
- [Business Metrics](http://grafana:3000/d/business)
- [Security Dashboard](http://grafana:3000/d/security)

## External Resources

- [Status Page](https://status.counterfactual-lab.com)
- [Documentation](https://docs.counterfactual-lab.com)
- [Support Portal](https://support.counterfactual-lab.com)
- [Incident Management System](https://incident.counterfactual-lab.com)

## Runbook Maintenance

### Review Schedule
- Monthly: Review all runbooks for accuracy
- Quarterly: Update contact information
- After incidents: Update relevant runbooks
- After major changes: Review affected runbooks

### Contributing
1. Use the standard template
2. Test procedures in staging
3. Get peer review
4. Update index when adding new runbooks

For questions about runbooks or to report issues, contact the SRE team.