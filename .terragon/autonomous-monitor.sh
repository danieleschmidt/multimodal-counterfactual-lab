#!/bin/bash

# Terragon Autonomous SDLC Monitoring Script
# Continuously monitors repository health and triggers value discovery

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TERRAGON_DIR="$REPO_ROOT/.terragon"
LOG_FILE="$TERRAGON_DIR/monitor.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if running in CI environment
is_ci() {
    [[ "${CI:-false}" == "true" ]] || [[ -n "${GITHUB_ACTIONS:-}" ]]
}

# Monitor repository health metrics
monitor_repository_health() {
    log "🔍 Monitoring repository health..."
    
    local health_score=0
    local max_score=100
    
    # Check CI/CD workflows (20 points)
    if [[ -d "$REPO_ROOT/.github/workflows" ]] && [[ $(find "$REPO_ROOT/.github/workflows" -name "*.yml" | wc -l) -gt 0 ]]; then
        health_score=$((health_score + 20))
        log "✅ CI/CD workflows present (+20 points)"
    else
        log "⚠️  No CI/CD workflows found"
    fi
    
    # Check test coverage (20 points)
    if [[ -f "$REPO_ROOT/pytest.ini" ]] && [[ -d "$REPO_ROOT/tests" ]]; then
        health_score=$((health_score + 20))
        log "✅ Test infrastructure present (+20 points)"
    else
        log "⚠️  Test infrastructure incomplete"
    fi
    
    # Check security configuration (20 points)
    if [[ -f "$REPO_ROOT/pyproject.toml" ]] && grep -q "security" "$REPO_ROOT/pyproject.toml"; then
        health_score=$((health_score + 20))
        log "✅ Security tooling configured (+20 points)"
    else
        log "⚠️  Security tooling not configured"
    fi
    
    # Check documentation (20 points)
    if [[ -f "$REPO_ROOT/README.md" ]] && [[ -d "$REPO_ROOT/docs" ]]; then
        health_score=$((health_score + 20))
        log "✅ Documentation structure present (+20 points)"
    else
        log "⚠️  Documentation incomplete"
    fi
    
    # Check dependency management (20 points)
    if [[ -f "$REPO_ROOT/requirements.txt" ]] || [[ -f "$REPO_ROOT/Pipfile.lock" ]] || [[ -f "$REPO_ROOT/poetry.lock" ]]; then
        health_score=$((health_score + 20))
        log "✅ Dependency locking present (+20 points)"
    else
        log "⚠️  No dependency locking found"
    fi
    
    local health_percentage=$((health_score * 100 / max_score))
    log "📊 Repository health score: $health_score/$max_score ($health_percentage%)"
    
    # Update metrics file with health score
    if [[ -f "$TERRAGON_DIR/value-metrics.json" ]]; then
        python3 -c "
import json
import sys
from datetime import datetime, timezone

try:
    with open('$TERRAGON_DIR/value-metrics.json', 'r') as f:
        metrics = json.load(f)
    
    metrics['repository']['currentScore'] = $health_percentage
    metrics['repository']['lastHealthCheck'] = datetime.now(timezone.utc).isoformat()
    
    with open('$TERRAGON_DIR/value-metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print('Health score updated successfully')
except Exception as e:
    print(f'Error updating health score: {e}', file=sys.stderr)
"
    fi
    
    return $health_percentage
}

# Check for security vulnerabilities
check_security_status() {
    log "🔒 Checking security status..."
    
    local security_issues=0
    
    # Check for security scan results
    if [[ -f "$REPO_ROOT/bandit-report.json" ]]; then
        security_issues=$(python3 -c "
import json
try:
    with open('$REPO_ROOT/bandit-report.json', 'r') as f:
        data = json.load(f)
    print(len(data.get('results', [])))
except:
    print(0)
")
        log "📊 Bandit security issues: $security_issues"
    fi
    
    # Check for vulnerable dependencies (if safety report exists)
    if [[ -f "$REPO_ROOT/safety-report.json" ]]; then
        local vuln_deps=$(python3 -c "
import json
try:
    with open('$REPO_ROOT/safety-report.json', 'r') as f:
        data = json.load(f)
    print(len(data.get('vulnerabilities', [])))
except:
    print(0)
")
        log "📊 Vulnerable dependencies: $vuln_deps"
        security_issues=$((security_issues + vuln_deps * 2))  # Weight dependency vulns higher
    fi
    
    if [[ $security_issues -gt 0 ]]; then
        log "⚠️  Security issues found: $security_issues"
        return 1
    else
        log "✅ No security issues detected"
        return 0
    fi
}

# Trigger value discovery if conditions are met
trigger_value_discovery() {
    log "🎯 Evaluating value discovery triggers..."
    
    local should_discover=false
    local discovery_reasons=()
    
    # Check if it's been more than 1 hour since last discovery
    if [[ -f "$TERRAGON_DIR/value-metrics.json" ]]; then
        local last_discovery=$(python3 -c "
import json
from datetime import datetime, timezone
try:
    with open('$TERRAGON_DIR/value-metrics.json', 'r') as f:
        metrics = json.load(f)
    last_run = metrics.get('continuousMetrics', {}).get('lastDiscoveryRun', '')
    if last_run:
        last_dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
        hours_ago = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
        print(f'{hours_ago:.1f}')
    else:
        print('999')  # Force discovery if never run
except:
    print('999')
")
        
        if (( $(echo "$last_discovery > 1.0" | bc -l) )); then
            should_discover=true
            discovery_reasons+=("Hourly discovery cycle")
        fi
    else
        should_discover=true
        discovery_reasons+=("Initial discovery run")
    fi
    
    # Check for new commits (if not in CI)
    if ! is_ci; then
        local commits_since_last_discovery=$(git log --oneline --since="1 hour ago" 2>/dev/null | wc -l || echo "0")
        if [[ $commits_since_last_discovery -gt 0 ]]; then
            should_discover=true
            discovery_reasons+=("New commits detected: $commits_since_last_discovery")
        fi
    fi
    
    # Check for security issues
    if ! check_security_status; then
        should_discover=true
        discovery_reasons+=("Security issues detected")
    fi
    
    if [[ "$should_discover" == "true" ]]; then
        log "🚀 Triggering value discovery. Reasons: ${discovery_reasons[*]}"
        
        # Run value discovery
        if python3 "$TERRAGON_DIR/discover-value.py"; then
            log "✅ Value discovery completed successfully"
        else
            log "❌ Value discovery failed"
            return 1
        fi
    else
        log "⏸️  No triggers for value discovery"
    fi
}

# Monitor continuous integration status  
monitor_ci_status() {
    if is_ci; then
        log "🏗️  Running in CI environment"
        
        # Monitor CI metrics
        local workflow_name="${GITHUB_WORKFLOW:-unknown}"
        local run_number="${GITHUB_RUN_NUMBER:-0}"
        
        log "📊 CI Metrics - Workflow: $workflow_name, Run: $run_number"
        
        # Update CI metrics in value-metrics.json
        if [[ -f "$TERRAGON_DIR/value-metrics.json" ]]; then
            python3 -c "
import json
from datetime import datetime, timezone

try:
    with open('$TERRAGON_DIR/value-metrics.json', 'r') as f:
        metrics = json.load(f)
    
    if 'ciMetrics' not in metrics:
        metrics['ciMetrics'] = {}
    
    metrics['ciMetrics']['lastRun'] = datetime.now(timezone.utc).isoformat()
    metrics['ciMetrics']['workflow'] = '$workflow_name'
    metrics['ciMetrics']['runNumber'] = $run_number
    
    with open('$TERRAGON_DIR/value-metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
except Exception as e:
    print(f'Error updating CI metrics: {e}')
"
        fi
    else
        log "💻 Running in local development environment"
    fi
}

# Main monitoring function
main() {
    log "🤖 Starting Terragon Autonomous SDLC Monitor"
    
    # Ensure terragon directory exists
    mkdir -p "$TERRAGON_DIR"
    
    # Monitor repository health
    local health_score
    health_score=$(monitor_repository_health)
    
    # Monitor CI status
    monitor_ci_status
    
    # Trigger value discovery based on conditions
    trigger_value_discovery
    
    # Generate summary
    log "📋 Monitoring cycle complete"
    log "   Health Score: $health_score%"
    log "   Next check: $(date -d '+1 hour' '+%Y-%m-%d %H:%M:%S')"
    
    # Exit with non-zero if health is critically low
    if [[ $health_score -lt 50 ]]; then
        log "⚠️  Critical: Repository health below 50%"
        exit 1
    fi
}

# Run main function
main "$@"