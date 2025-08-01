#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuously discovers and prioritizes the highest-value work items.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ValueDiscoveryEngine:
    """Main engine for discovering and scoring value opportunities."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
    def discover_items(self) -> List[Dict]:
        """Discover new value items from multiple sources."""
        items = []
        
        # Source 1: Git history analysis
        items.extend(self._discover_from_git())
        
        # Source 2: Static code analysis  
        items.extend(self._discover_from_static_analysis())
        
        # Source 3: Security vulnerability scanning
        items.extend(self._discover_from_security())
        
        # Source 4: Dependency analysis
        items.extend(self._discover_from_dependencies())
        
        # Source 5: Documentation gaps
        items.extend(self._discover_from_documentation())
        
        # Source 6: ML/AI specific analysis
        items.extend(self._discover_from_ml_analysis())
        
        return items
    
    def _discover_from_git(self) -> List[Dict]:
        """Discover items from git history analysis."""
        items = []
        
        try:
            # Find TODO/FIXME/HACK comments
            result = subprocess.run([
                "grep", "-r", "-n", "-E", 
                "(TODO|FIXME|XXX|HACK|DEPRECATED)", 
                "src/", "tests/"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                items.append({
                    "id": "tech-debt-markers",
                    "title": "Address technical debt markers in code",
                    "category": "technical-debt",
                    "priority": "medium",
                    "effort": 4,
                    "source": "git-analysis",
                    "details": f"Found {len(result.stdout.splitlines())} debt markers"
                })
        except Exception:
            pass
            
        return items
    
    def _discover_from_static_analysis(self) -> List[Dict]:
        """Discover items from static code analysis."""
        items = []
        
        try:
            # Run ruff to find code quality issues
            result = subprocess.run([
                "ruff", "check", "src/", "--output-format=json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                if issues:
                    items.append({
                        "id": "code-quality-issues",
                        "title": f"Fix {len(issues)} code quality issues",
                        "category": "quality",
                        "priority": "low" if len(issues) < 10 else "medium",
                        "effort": min(len(issues) // 5 + 1, 6),
                        "source": "static-analysis"
                    })
        except Exception:
            pass
            
        return items
    
    def _discover_from_security(self) -> List[Dict]:
        """Discover security-related items."""
        items = []
        
        # Check if security scanning script exists and suggest running it
        security_script = self.repo_path / "scripts" / "security_scan.sh"
        if security_script.exists():
            items.append({
                "id": "security-scan-execution",
                "title": "Run comprehensive security scan",
                "category": "security",
                "priority": "high",
                "effort": 1,
                "source": "security-analysis"
            })
            
        return items
    
    def _discover_from_dependencies(self) -> List[Dict]:
        """Discover dependency-related improvements."""
        items = []
        
        # Check for missing lock files
        pyproject_path = self.repo_path / "pyproject.toml"
        requirements_path = self.repo_path / "requirements.txt"
        
        if pyproject_path.exists() and not requirements_path.exists():
            items.append({
                "id": "dependency-locks",
                "title": "Generate dependency lock files for reproducible builds",
                "category": "infrastructure",
                "priority": "medium",
                "effort": 1,
                "source": "dependency-analysis"
            })
            
        return items
    
    def _discover_from_documentation(self) -> List[Dict]:
        """Discover documentation improvements."""
        items = []
        
        # Check for missing API documentation
        docs_path = self.repo_path / "docs"
        if not (docs_path / "api").exists() and (self.repo_path / "src").exists():
            items.append({
                "id": "api-documentation",
                "title": "Generate comprehensive API documentation",
                "category": "documentation",
                "priority": "medium",
                "effort": 3,
                "source": "documentation-analysis"
            })
            
        return items
    
    def _discover_from_ml_analysis(self) -> List[Dict]:
        """Discover ML/AI specific improvements."""
        items = []
        
        # Check for core ML functionality implementation
        core_path = self.repo_path / "src" / "counterfactual_lab" / "core.py"
        if core_path.exists():
            try:
                content = core_path.read_text()
                if "NotImplementedError" in content:
                    items.append({
                        "id": "ml-core-implementation",
                        "title": "Implement core ML functionality (CounterfactualGenerator, BiasEvaluator)",
                        "category": "feature",
                        "priority": "critical",
                        "effort": 8,
                        "source": "ml-analysis"
                    })
            except Exception:
                pass
                
        # Check for missing ML monitoring
        monitoring_path = self.repo_path / "monitoring"
        if not (monitoring_path / "ml_metrics.yml").exists():
            items.append({
                "id": "ml-monitoring",
                "title": "Add ML model monitoring and drift detection",
                "category": "monitoring",
                "priority": "medium",
                "effort": 6,
                "source": "ml-analysis"
            })
            
        return items
    
    def score_items(self, items: List[Dict]) -> List[Dict]:
        """Score items using WSJF + ICE + Technical Debt scoring."""
        scored_items = []
        
        for item in items:
            # WSJF Components
            user_value = self._score_user_value(item)
            time_criticality = self._score_time_criticality(item)
            risk_reduction = self._score_risk_reduction(item)
            opportunity = self._score_opportunity(item)
            
            cost_of_delay = user_value + time_criticality + risk_reduction + opportunity
            job_size = item.get("effort", 1)
            wsjf = cost_of_delay / job_size if job_size > 0 else 0
            
            # ICE Components
            impact = self._score_impact(item)
            confidence = self._score_confidence(item)
            ease = 10 - min(job_size, 10)  # Inverse of effort
            
            ice = impact * confidence * ease
            
            # Technical Debt Score
            tech_debt = self._score_technical_debt(item)
            
            # Composite Score
            composite = (
                0.6 * self._normalize_score(wsjf, 0, 50) +
                0.1 * self._normalize_score(ice, 0, 1000) +
                0.2 * self._normalize_score(tech_debt, 0, 100) +
                0.1 * self._apply_category_boost(item)
            ) * 100
            
            # Apply priority boosts
            if item.get("priority") == "critical":
                composite *= 1.5
            elif item.get("priority") == "high":
                composite *= 1.2
                
            scored_item = item.copy()
            scored_item.update({
                "scores": {
                    "wsjf": round(wsjf, 1),
                    "ice": round(ice, 1),
                    "technicalDebt": round(tech_debt, 1),
                    "composite": round(composite, 1)
                },
                "discoveredAt": datetime.now(timezone.utc).isoformat()
            })
            
            scored_items.append(scored_item)
            
        return sorted(scored_items, key=lambda x: x["scores"]["composite"], reverse=True)
    
    def _score_user_value(self, item: Dict) -> float:
        """Score user/business value impact."""
        category = item.get("category", "")
        if category in ["feature", "ml-core"]:
            return 8.0
        elif category in ["security", "reliability"]:
            return 7.0
        elif category in ["performance", "monitoring"]:
            return 6.0
        else:
            return 4.0
    
    def _score_time_criticality(self, item: Dict) -> float:
        """Score time sensitivity."""
        priority = item.get("priority", "medium")
        if priority == "critical":
            return 8.0
        elif priority == "high":
            return 6.0
        elif priority == "medium":
            return 4.0
        else:
            return 2.0
    
    def _score_risk_reduction(self, item: Dict) -> float:
        """Score risk mitigation value."""
        category = item.get("category", "")
        if category == "security":
            return 9.0
        elif category == "reliability":
            return 7.0
        elif category == "infrastructure":
            return 5.0
        else:
            return 3.0
    
    def _score_opportunity(self, item: Dict) -> float:
        """Score opportunity enablement."""
        category = item.get("category", "")
        if category in ["feature", "infrastructure"]:
            return 6.0
        else:
            return 3.0
    
    def _score_impact(self, item: Dict) -> float:
        """Score business impact (1-10)."""
        return self._score_user_value(item)
    
    def _score_confidence(self, item: Dict) -> float:
        """Score execution confidence (1-10)."""
        effort = item.get("effort", 1)
        if effort <= 2:
            return 9.0
        elif effort <= 4:
            return 7.0
        elif effort <= 8:
            return 5.0
        else:
            return 3.0
    
    def _score_technical_debt(self, item: Dict) -> float:
        """Score technical debt impact."""
        if item.get("category") == "technical-debt":
            return 80.0
        elif "debt" in item.get("title", "").lower():
            return 60.0
        else:
            return 20.0
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-1 range."""
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
    
    def _apply_category_boost(self, item: Dict) -> float:
        """Apply category-specific boosts."""
        category = item.get("category", "")
        if category == "security":
            return 10.0
        elif category == "feature":
            return 8.0
        elif category == "infrastructure":
            return 6.0
        else:
            return 5.0
    
    def update_metrics(self, new_items: List[Dict]) -> Dict:
        """Update the value metrics file with new discoveries."""
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                metrics = json.load(f)
        else:
            metrics = {"executionHistory": [], "discoveredItems": []}
        
        # Add new items to discovered items
        existing_ids = {item["id"] for item in metrics.get("discoveredItems", [])}
        for item in new_items:
            if item["id"] not in existing_ids:
                metrics.setdefault("discoveredItems", []).append(item)
        
        # Update metrics
        metrics["repository"]["lastUpdated"] = datetime.now(timezone.utc).isoformat()
        metrics["continuousMetrics"]["lastDiscoveryRun"] = datetime.now(timezone.utc).isoformat()
        
        # Update backlog metrics
        total_items = len(metrics.get("discoveredItems", []))
        estimated_effort = sum(
            item.get("effort", 1) if isinstance(item.get("effort", 1), (int, float)) else 1 
            for item in metrics.get("discoveredItems", [])
        )
        potential_value = sum(
            item.get("scores", {}).get("composite", 0) if isinstance(item.get("scores", {}), dict) else 0
            for item in metrics.get("discoveredItems", [])
        )
        
        metrics["backlogMetrics"] = {
            "totalItems": total_items,
            "estimatedTotalEffort": estimated_effort,
            "potentialValueDelivery": round(potential_value, 1),
            "averageAge": 0  # Would calculate based on discovery dates
        }
        
        # Save updated metrics
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics
    
    def update_backlog(self, metrics: Dict) -> None:
        """Update the BACKLOG.md file with latest discoveries."""
        items = metrics.get("discoveredItems", [])
        sorted_items = sorted(items, key=lambda x: x.get("scores", {}).get("composite", 0), reverse=True)
        
        # Get the next best value item
        next_item = sorted_items[0] if sorted_items else None
        
        backlog_content = f"""# 📊 Autonomous Value Backlog

**Repository**: multimodal-counterfactual-lab  
**Maturity Level**: {metrics.get('repository', {}).get('currentScore', 68.5)}%  
**Last Updated**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Next Discovery**: {(datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S UTC")}  

## 🎯 Next Best Value Item

"""
        
        if next_item:
            backlog_content += f"""**[{next_item['id'].upper()}] {next_item['title']}**
- **Composite Score**: {next_item.get('scores', {}).get('composite', 0):.1f} ⭐
- **WSJF**: {next_item.get('scores', {}).get('wsjf', 0):.1f} | **ICE**: {next_item.get('scores', {}).get('ice', 0):.1f} | **Tech Debt**: {next_item.get('scores', {}).get('technicalDebt', 0):.1f}
- **Estimated Effort**: {next_item.get('effort', 1)} hours
- **Category**: {next_item.get('category', 'unknown').title()} | **Priority**: {next_item.get('priority', 'medium').upper()}

"""
        
        backlog_content += """## 📋 Prioritized Backlog

| Rank | ID | Title | Score | Category | Effort | Priority |
|------|-----|--------|---------|----------|---------|----------|
"""
        
        for i, item in enumerate(sorted_items[:10], 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
            backlog_content += f"| {rank_emoji} | {item['id'].upper()} | {item['title'][:50]}{'...' if len(item['title']) > 50 else ''} | {item.get('scores', {}).get('composite', 0):.1f} | {item.get('category', 'unknown').title()} | {item.get('effort', 1)}h | {item.get('priority', 'medium').upper()} |\n"
        
        backlog_content += f"""

## 📈 Value Metrics Dashboard

### Backlog Statistics
- **Total Items**: {len(sorted_items)}
- **Estimated Total Effort**: {sum(item.get('effort', 1) if isinstance(item.get('effort', 1), (int, float)) else 1 for item in sorted_items)} hours
- **Potential Value Delivery**: {sum(item.get('scores', {}).get('composite', 0) if isinstance(item.get('scores', {}), dict) else 0 for item in sorted_items):.1f} points

### Discovery Status
- **Last Discovery Run**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
- **New Items Found**: {len([item for item in sorted_items if (datetime.now(timezone.utc) - datetime.fromisoformat(item.get('discoveredAt', datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))).days < 1])}
- **Sources**: Git Analysis, Static Analysis, Security Scan, Dependencies, Documentation, ML Analysis

---
*🤖 Generated by Terragon Autonomous SDLC Agent*  
*Value-driven development for maximum impact*
"""
        
        with open(self.backlog_path, "w") as f:
            f.write(backlog_content)
    
    def run_discovery(self) -> None:
        """Run the complete value discovery cycle."""
        print("🔍 Starting value discovery cycle...")
        
        # Discover new items
        items = self.discover_items()
        print(f"📦 Discovered {len(items)} potential value items")
        
        # Score items
        scored_items = self.score_items(items)
        print(f"📊 Scored and prioritized {len(scored_items)} items")
        
        # Update metrics
        metrics = self.update_metrics(scored_items)
        print(f"💾 Updated value metrics")
        
        # Update backlog
        self.update_backlog(metrics)
        print(f"📋 Updated backlog with {len(scored_items)} items")
        
        # Print top 3 items
        if scored_items:
            print("\n🎯 Top 3 Value Items:")
            for i, item in enumerate(scored_items[:3], 1):
                print(f"{i}. {item['title']} (Score: {item['scores']['composite']:.1f})")
        
        print("✅ Value discovery cycle complete!")


if __name__ == "__main__":
    engine = ValueDiscoveryEngine()
    engine.run_discovery()