# ============================================================================
# IMPORTS
# ============================================================================

from datetime import datetime
import json
from typing import Dict, Any

# ============================================================================
# EVALUATOR CLASS
# ============================================================================

class NegotifyEvaluator:
    """Evaluation metrics for Negotify agent performance"""
    
    def __init__(self):
        self.metrics = {
            "contracts_analyzed": 0,
            "risks_identified": 0,
            "high_risks_caught": 0,
            "negotiation_success_rate": 0.0,
            "avg_response_time": 0.0,
            "user_satisfaction": 0.0
        }
        self.evaluations = []
    
    def evaluate_analysis(self, contract_text: str, analysis_result: dict) -> dict:
        """Evaluate contract analysis quality"""
        
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "contract_length": len(contract_text),
            "risks_found": len(analysis_result.get("high_risks", [])) + 
                          len(analysis_result.get("medium_risks", [])),
            "risk_score": analysis_result.get("overall_risk_score", 0),
            "completeness": self._check_completeness(analysis_result),
            "accuracy_indicators": self._check_accuracy(analysis_result)
        }
        
        self.evaluations.append(evaluation)
        self._update_metrics(evaluation)
        
        return evaluation
    
    def _check_completeness(self, result: dict) -> float:
        """Check if all required fields are present"""
        required_fields = ["overall_risk_score", "high_risks", "medium_risks", "low_risks"]
        present = sum(1 for f in required_fields if f in result and result[f] is not None)
        return present / len(required_fields)
    
    def _check_accuracy(self, result: dict) -> dict:
        """Check accuracy indicators"""
        return {
            "has_specific_clauses": any(r.get("clause_id") for r in result.get("high_risks", [])),
            "has_financial_impact": any(r.get("financial_impact") for r in result.get("high_risks", [])),
            "has_recommendations": any(r.get("recommended_changes") for r in result.get("high_risks", []))
        }
    
    def _update_metrics(self, evaluation: dict):
        """Update aggregate metrics"""
        self.metrics["contracts_analyzed"] += 1
        self.metrics["risks_identified"] += evaluation["risks_found"]
    
    def get_summary(self) -> str:
        """Get evaluation summary for display"""
        return f"""
📊 NEGOTIFY EVALUATION METRICS
================================
Contracts Analyzed: {self.metrics['contracts_analyzed']}
Total Risks Identified: {self.metrics['risks_identified']}
Avg Completeness: {sum(e['completeness'] for e in self.evaluations) / max(len(self.evaluations), 1):.1%}
"""