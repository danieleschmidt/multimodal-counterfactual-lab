"""Global compliance and regulatory framework for counterfactual generation."""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import re

logger = logging.getLogger(__name__)


@dataclass
class ComplianceAuditRecord:
    """Record of a compliance audit."""
    audit_id: str
    regulation: str
    audit_date: str
    compliance_status: str
    findings: List[str]
    remediation_actions: List[str]
    next_audit_due: str
    auditor: str
    scope: List[str]


@dataclass
class PrivacyProtectionConfig:
    """Configuration for privacy protection."""
    data_minimization: bool = True
    anonymization_required: bool = True
    consent_tracking: bool = True
    retention_period_days: int = 365
    cross_border_transfer_allowed: bool = False
    approved_regions: List[str] = None
    
    def __post_init__(self):
        if self.approved_regions is None:
            self.approved_regions = ["EU", "US", "CA"]


class RegionalComplianceManager:
    """Manages compliance with regional regulations."""
    
    def __init__(self, primary_region: str = "EU"):
        """Initialize compliance manager.
        
        Args:
            primary_region: Primary regulatory region (EU, US, CA, etc.)
        """
        self.primary_region = primary_region
        self.compliance_frameworks = self._initialize_frameworks()
        self.audit_records: List[ComplianceAuditRecord] = []
        self.privacy_config = PrivacyProtectionConfig()
        
        # Data processing logs for compliance
        self.processing_logs: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized compliance manager for region: {primary_region}")
    
    def _initialize_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize regulatory compliance frameworks."""
        return {
            "EU_AI_ACT": {
                "name": "EU AI Act",
                "scope": ["AI systems with potential bias impact"],
                "requirements": [
                    "Bias testing and documentation",
                    "Human oversight mechanisms", 
                    "Transparency and explainability",
                    "Risk assessment and mitigation",
                    "Quality management system"
                ],
                "audit_frequency": 180,  # days
                "penalties": "Up to 6% of global revenue",
                "data_protection": True
            },
            "GDPR": {
                "name": "General Data Protection Regulation",
                "scope": ["Personal data processing"],
                "requirements": [
                    "Lawful basis for processing",
                    "Data minimization",
                    "Right to erasure",
                    "Data portability",
                    "Privacy by design"
                ],
                "audit_frequency": 365,  # days
                "penalties": "Up to 4% of global revenue",
                "data_protection": True
            },
            "US_ALGORITHMIC_ACCOUNTABILITY": {
                "name": "US Algorithmic Accountability Act",
                "scope": ["Automated decision systems"],
                "requirements": [
                    "Impact assessments",
                    "Bias testing",
                    "Public documentation",
                    "Consumer notification"
                ],
                "audit_frequency": 365,
                "penalties": "Varies by jurisdiction",
                "data_protection": False
            },
            "CCPA": {
                "name": "California Consumer Privacy Act",
                "scope": ["California residents' data"],
                "requirements": [
                    "Disclosure of data collection",
                    "Right to delete",
                    "Right to opt-out",
                    "Non-discrimination"
                ],
                "audit_frequency": 365,
                "penalties": "Up to $7,500 per violation",
                "data_protection": True
            }
        }
    
    def assess_compliance_requirements(self, 
                                     data_types: List[str],
                                     processing_purposes: List[str],
                                     affected_regions: List[str]) -> Dict[str, Any]:
        """Assess compliance requirements for specific processing.
        
        Args:
            data_types: Types of data being processed
            processing_purposes: Purposes of processing
            affected_regions: Regions where data subjects are located
            
        Returns:
            Compliance assessment results
        """
        applicable_frameworks = []
        requirements = set()
        max_audit_frequency = 0
        
        # Determine applicable frameworks
        for region in affected_regions:
            if region == "EU" or region == "EEA":
                applicable_frameworks.extend(["EU_AI_ACT", "GDPR"])
            elif region == "US":
                applicable_frameworks.append("US_ALGORITHMIC_ACCOUNTABILITY")
            elif region == "CA":
                applicable_frameworks.extend(["US_ALGORITHMIC_ACCOUNTABILITY", "CCPA"])
        
        # Remove duplicates
        applicable_frameworks = list(set(applicable_frameworks))
        
        # Collect requirements
        compliance_details = {}
        for framework_id in applicable_frameworks:
            framework = self.compliance_frameworks[framework_id]
            compliance_details[framework_id] = framework
            requirements.update(framework["requirements"])
            max_audit_frequency = max(max_audit_frequency, framework["audit_frequency"])
        
        # Assess specific requirements
        personal_data_involved = any("personal" in dt.lower() or "biometric" in dt.lower() 
                                   for dt in data_types)
        ai_decision_making = any("generation" in purpose or "evaluation" in purpose 
                               for purpose in processing_purposes)
        
        assessment = {
            "applicable_frameworks": applicable_frameworks,
            "compliance_details": compliance_details,
            "unified_requirements": list(requirements),
            "personal_data_involved": personal_data_involved,
            "ai_decision_making": ai_decision_making,
            "audit_frequency_days": max_audit_frequency,
            "high_risk_processing": personal_data_involved and ai_decision_making,
            "recommendations": self._generate_compliance_recommendations(
                applicable_frameworks, personal_data_involved, ai_decision_making
            )
        }
        
        return assessment
    
    def _generate_compliance_recommendations(self,
                                           frameworks: List[str],
                                           personal_data: bool,
                                           ai_decisions: bool) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        if personal_data:
            recommendations.extend([
                "Implement data minimization practices",
                "Obtain explicit consent where required",
                "Establish data retention and deletion policies",
                "Implement privacy by design principles"
            ])
        
        if ai_decisions:
            recommendations.extend([
                "Conduct algorithmic impact assessments",
                "Implement bias testing and monitoring",
                "Provide model explainability mechanisms",
                "Establish human oversight procedures"
            ])
        
        if "EU_AI_ACT" in frameworks:
            recommendations.extend([
                "Register as high-risk AI system if applicable",
                "Implement CE marking compliance",
                "Establish quality management system"
            ])
        
        if "GDPR" in frameworks:
            recommendations.extend([
                "Appoint Data Protection Officer if required",
                "Conduct Data Protection Impact Assessment",
                "Implement appropriate technical and organizational measures"
            ])
        
        return recommendations
    
    def log_processing_activity(self,
                              activity_type: str,
                              data_categories: List[str],
                              purposes: List[str],
                              legal_basis: str,
                              data_subjects: Optional[str] = None,
                              retention_period: Optional[int] = None) -> str:
        """Log data processing activity for compliance.
        
        Args:
            activity_type: Type of processing activity
            data_categories: Categories of data processed
            purposes: Purposes of processing
            legal_basis: Legal basis for processing
            data_subjects: Categories of data subjects
            retention_period: Data retention period in days
            
        Returns:
            Activity log ID
        """
        activity_id = f"activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.processing_logs)}"
        
        log_entry = {
            "activity_id": activity_id,
            "timestamp": datetime.now().isoformat(),
            "activity_type": activity_type,
            "data_categories": data_categories,
            "purposes": purposes,
            "legal_basis": legal_basis,
            "data_subjects": data_subjects or "research participants",
            "retention_period_days": retention_period or self.privacy_config.retention_period_days,
            "processing_region": self.primary_region,
            "anonymized": "anonymization" in [p.lower() for p in purposes],
            "consent_obtained": legal_basis.lower() == "consent"
        }
        
        self.processing_logs.append(log_entry)
        
        logger.info(f"Logged processing activity: {activity_id}")
        return activity_id
    
    def validate_cross_border_transfer(self,
                                     source_region: str,
                                     destination_region: str,
                                     data_types: List[str]) -> Dict[str, Any]:
        """Validate cross-border data transfer compliance.
        
        Args:
            source_region: Source region code
            destination_region: Destination region code  
            data_types: Types of data being transferred
            
        Returns:
            Transfer validation results
        """
        validation_result = {
            "transfer_allowed": False,
            "legal_mechanism": None,
            "additional_safeguards": [],
            "risks_identified": [],
            "recommendations": []
        }
        
        # Check if personal data is involved
        personal_data = any("personal" in dt.lower() or "biometric" in dt.lower() 
                          for dt in data_types)
        
        if not personal_data:
            validation_result["transfer_allowed"] = True
            validation_result["legal_mechanism"] = "No personal data - transfer permitted"
            return validation_result
        
        # EU to non-EU transfers (GDPR Article 44-49)
        if source_region in ["EU", "EEA"] and destination_region not in ["EU", "EEA"]:
            adequacy_decisions = ["CA", "UK", "IL", "JP", "KR", "NZ", "CH", "UY"]
            
            if destination_region in adequacy_decisions:
                validation_result["transfer_allowed"] = True
                validation_result["legal_mechanism"] = "European Commission adequacy decision"
            else:
                validation_result["transfer_allowed"] = True  # With safeguards
                validation_result["legal_mechanism"] = "Standard Contractual Clauses required"
                validation_result["additional_safeguards"] = [
                    "Implement Standard Contractual Clauses (SCCs)",
                    "Conduct Transfer Impact Assessment",
                    "Implement additional technical measures if required"
                ]
        
        # US state-specific requirements
        elif source_region == "CA":  # California
            validation_result["transfer_allowed"] = True
            validation_result["legal_mechanism"] = "CCPA disclosure requirements"
            validation_result["additional_safeguards"] = [
                "Provide clear disclosure to California residents",
                "Honor opt-out requests",
                "Ensure service provider agreements include CCPA requirements"
            ]
        
        # General risk assessment
        high_risk_destinations = ["CN", "RU", "XX"]  # Example high-risk regions
        if destination_region in high_risk_destinations:
            validation_result["risks_identified"] = [
                "Government surveillance laws",
                "Limited data protection framework",
                "Potential for data localization requirements"
            ]
            validation_result["additional_safeguards"].extend([
                "Enhanced encryption in transit and at rest",
                "Regular security assessments",
                "Data minimization measures"
            ])
        
        return validation_result
    
    def conduct_bias_audit(self,
                          model_id: str,
                          test_results: Dict[str, Any],
                          protected_attributes: List[str]) -> ComplianceAuditRecord:
        """Conduct bias audit for compliance.
        
        Args:
            model_id: Identifier of the model being audited
            test_results: Results from bias testing
            protected_attributes: Protected attributes tested
            
        Returns:
            Audit record
        """
        audit_id = f"bias_audit_{model_id}_{datetime.now().strftime('%Y%m%d')}"
        findings = []
        remediation_actions = []
        
        # Analyze test results
        if "fairness_score" in test_results:
            fairness_score = test_results["fairness_score"]
            if fairness_score < 0.7:
                findings.append(f"Low fairness score: {fairness_score:.3f}")
                remediation_actions.append("Implement bias mitigation techniques")
        
        if "demographic_parity" in test_results:
            dp_results = test_results["demographic_parity"]
            if isinstance(dp_results, dict) and "overall_score" in dp_results:
                if dp_results["overall_score"] > 0.1:
                    findings.append(f"Demographic parity violation: {dp_results['overall_score']:.3f}")
                    remediation_actions.append("Balance representation across protected attributes")
        
        # Check for specific attribute biases
        for attr in protected_attributes:
            if f"{attr}_bias_detected" in test_results:
                if test_results[f"{attr}_bias_detected"]:
                    findings.append(f"Bias detected for attribute: {attr}")
                    remediation_actions.append(f"Implement {attr}-specific bias correction")
        
        # Determine compliance status
        compliance_status = "COMPLIANT" if not findings else "NON_COMPLIANT"
        if len(findings) == 1 and "Low fairness score" in findings[0]:
            # Minor issue
            compliance_status = "CONDITIONAL_COMPLIANCE"
        
        # Create audit record
        audit_record = ComplianceAuditRecord(
            audit_id=audit_id,
            regulation="EU_AI_ACT",
            audit_date=datetime.now().isoformat(),
            compliance_status=compliance_status,
            findings=findings or ["No significant bias detected"],
            remediation_actions=remediation_actions or ["Continue regular monitoring"],
            next_audit_due=(datetime.now() + timedelta(days=180)).isoformat(),
            auditor="Automated Bias Audit System",
            scope=[f"Model: {model_id}", f"Attributes: {', '.join(protected_attributes)}"]
        )
        
        self.audit_records.append(audit_record)
        
        logger.info(f"Completed bias audit {audit_id}: {compliance_status}")
        return audit_record
    
    def generate_compliance_report(self,
                                 report_type: str = "comprehensive",
                                 timeframe_days: int = 365) -> Dict[str, Any]:
        """Generate compliance report.
        
        Args:
            report_type: Type of report (comprehensive, executive, technical)
            timeframe_days: Timeframe for the report in days
            
        Returns:
            Compliance report
        """
        cutoff_date = datetime.now() - timedelta(days=timeframe_days)
        
        # Filter recent activities and audits
        recent_activities = [
            log for log in self.processing_logs
            if datetime.fromisoformat(log["timestamp"]) > cutoff_date
        ]
        
        recent_audits = [
            audit for audit in self.audit_records
            if datetime.fromisoformat(audit.audit_date) > cutoff_date
        ]
        
        # Calculate compliance metrics
        total_audits = len(recent_audits)
        compliant_audits = len([a for a in recent_audits if a.compliance_status == "COMPLIANT"])
        compliance_rate = (compliant_audits / total_audits * 100) if total_audits > 0 else 100
        
        # Identify high-risk activities
        high_risk_activities = [
            log for log in recent_activities
            if "personal" in str(log.get("data_categories", [])).lower() and
               "generation" in str(log.get("purposes", [])).lower()
        ]
        
        report = {
            "report_metadata": {
                "report_type": report_type,
                "generation_date": datetime.now().isoformat(),
                "timeframe_days": timeframe_days,
                "reporting_region": self.primary_region,
                "report_id": f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "executive_summary": {
                "compliance_rate": compliance_rate,
                "total_audits_conducted": total_audits,
                "high_risk_activities": len(high_risk_activities),
                "data_processing_activities": len(recent_activities),
                "regulatory_frameworks": list(self.compliance_frameworks.keys()),
                "overall_status": "GOOD" if compliance_rate >= 90 else "NEEDS_IMPROVEMENT"
            },
            "detailed_findings": {
                "audit_summary": [asdict(audit) for audit in recent_audits],
                "processing_activities": recent_activities,
                "cross_border_transfers": self._analyze_cross_border_activities(recent_activities),
                "privacy_compliance": self._assess_privacy_compliance(recent_activities)
            },
            "recommendations": self._generate_report_recommendations(compliance_rate, recent_audits),
            "next_actions": {
                "upcoming_audits": self._get_upcoming_audits(),
                "remediation_items": self._get_pending_remediations(recent_audits)
            }
        }
        
        if report_type == "executive":
            # Simplified executive report
            return {
                "report_metadata": report["report_metadata"],
                "executive_summary": report["executive_summary"],
                "key_recommendations": report["recommendations"][:3],
                "next_actions": report["next_actions"]
            }
        
        return report
    
    def _analyze_cross_border_activities(self, activities: List[Dict]) -> Dict[str, Any]:
        """Analyze cross-border data transfer activities."""
        transfers = []
        for activity in activities:
            if activity.get("processing_region") != self.primary_region:
                transfers.append({
                    "activity_id": activity["activity_id"],
                    "source_region": self.primary_region,
                    "destination_region": activity.get("processing_region"),
                    "data_categories": activity.get("data_categories", []),
                    "legal_basis": activity.get("legal_basis")
                })
        
        return {
            "total_transfers": len(transfers),
            "transfer_details": transfers,
            "compliance_status": "COMPLIANT" if all(
                t.get("legal_basis") in ["legitimate_interest", "consent", "contract"]
                for t in transfers
            ) else "REVIEW_REQUIRED"
        }
    
    def _assess_privacy_compliance(self, activities: List[Dict]) -> Dict[str, Any]:
        """Assess privacy compliance of activities."""
        personal_data_activities = [
            a for a in activities
            if any("personal" in str(cat).lower() for cat in a.get("data_categories", []))
        ]
        
        consent_based = len([a for a in personal_data_activities 
                           if a.get("legal_basis") == "consent"])
        anonymized = len([a for a in personal_data_activities 
                         if a.get("anonymized", False)])
        
        return {
            "personal_data_activities": len(personal_data_activities),
            "consent_based_processing": consent_based,
            "anonymized_processing": anonymized,
            "privacy_score": ((consent_based + anonymized) / max(len(personal_data_activities), 1)) * 100,
            "compliance_level": "HIGH" if len(personal_data_activities) == 0 or 
                               ((consent_based + anonymized) / len(personal_data_activities)) > 0.8 else "MEDIUM"
        }
    
    def _generate_report_recommendations(self, compliance_rate: float, audits: List) -> List[str]:
        """Generate recommendations based on compliance analysis."""
        recommendations = []
        
        if compliance_rate < 90:
            recommendations.append("Increase bias testing frequency and improve mitigation strategies")
        
        if any(audit.compliance_status == "NON_COMPLIANT" for audit in audits):
            recommendations.append("Address non-compliant findings immediately")
        
        recommendations.extend([
            "Continue regular compliance monitoring",
            "Update privacy policies to reflect current processing activities",
            "Conduct staff training on data protection requirements",
            "Review and update technical and organizational measures"
        ])
        
        return recommendations
    
    def _get_upcoming_audits(self) -> List[Dict[str, str]]:
        """Get upcoming audit requirements."""
        upcoming = []
        current_date = datetime.now()
        
        for audit in self.audit_records:
            next_due = datetime.fromisoformat(audit.next_audit_due)
            if next_due <= current_date + timedelta(days=30):  # Due within 30 days
                upcoming.append({
                    "audit_type": audit.regulation,
                    "due_date": audit.next_audit_due,
                    "scope": audit.scope
                })
        
        return upcoming
    
    def _get_pending_remediations(self, audits: List) -> List[str]:
        """Get pending remediation actions."""
        actions = []
        for audit in audits:
            if audit.compliance_status != "COMPLIANT":
                actions.extend(audit.remediation_actions)
        
        return list(set(actions))  # Remove duplicates
    
    def export_compliance_data(self, file_path: str):
        """Export compliance data for external auditing.
        
        Args:
            file_path: Path to export file
        """
        export_data = {
            "export_metadata": {
                "export_date": datetime.now().isoformat(),
                "region": self.primary_region,
                "data_version": "1.0"
            },
            "compliance_frameworks": self.compliance_frameworks,
            "audit_records": [asdict(audit) for audit in self.audit_records],
            "processing_logs": self.processing_logs,
            "privacy_configuration": asdict(self.privacy_config)
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Compliance data exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export compliance data: {e}")
            raise


# Global compliance manager instance
_global_compliance_manager = None

def get_global_compliance_manager(region: str = "EU") -> RegionalComplianceManager:
    """Get or create global compliance manager."""
    global _global_compliance_manager
    if _global_compliance_manager is None:
        _global_compliance_manager = RegionalComplianceManager(region)
    return _global_compliance_manager

def initialize_compliance(region: str = "EU",
                         privacy_config: Optional[PrivacyProtectionConfig] = None) -> RegionalComplianceManager:
    """Initialize global compliance management.
    
    Args:
        region: Primary regulatory region
        privacy_config: Privacy protection configuration
        
    Returns:
        Global compliance manager instance
    """
    global _global_compliance_manager
    _global_compliance_manager = RegionalComplianceManager(region)
    
    if privacy_config:
        _global_compliance_manager.privacy_config = privacy_config
    
    logger.info(f"Global compliance management initialized for region: {region}")
    return _global_compliance_manager