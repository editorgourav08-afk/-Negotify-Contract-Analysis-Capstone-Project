# ============================================================================
# IMPORTS
# ============================================================================

from datetime import datetime
import json
from typing import Dict, Any, List, Union

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
    
    def evaluate_analysis(self, contract_text: str, analysis_result: Union[dict, list, Any]) -> dict:
        """Evaluate contract analysis quality
        
        Args:
            contract_text: The original contract text
            analysis_result: Can be dict, list, or Pydantic model
        """
        
        # Normalize the analysis_result to a consistent format
        normalized = self._normalize_result(analysis_result)
        
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "contract_length": len(contract_text) if contract_text else 0,
            "risks_found": len(normalized.get("high_risks", [])) + 
                          len(normalized.get("medium_risks", [])) +
                          len(normalized.get("low_risks", [])),
            "high_risk_count": len(normalized.get("high_risks", [])),
            "medium_risk_count": len(normalized.get("medium_risks", [])),
            "low_risk_count": len(normalized.get("low_risks", [])),
            "risk_score": normalized.get("overall_risk_score", 0),
            "completeness": self._check_completeness(normalized),
            "accuracy_indicators": self._check_accuracy(normalized)
        }
        
        self.evaluations.append(evaluation)
        self._update_metrics(evaluation)
        
        return evaluation
    
    def _normalize_result(self, analysis_result: Union[dict, list, Any]) -> dict:
        """Normalize different input types to a consistent dict format"""
        
        # Case 1: Already a dict
        if isinstance(analysis_result, dict):
            return analysis_result
        
        # Case 2: It's a list (e.g., list of risks)
        if isinstance(analysis_result, list):
            # Categorize risks by severity
            high_risks = []
            medium_risks = []
            low_risks = []
            
            for item in analysis_result:
                if isinstance(item, dict):
                    severity = item.get("severity", 5)
                    if severity >= 8:
                        high_risks.append(item)
                    elif severity >= 4:
                        medium_risks.append(item)
                    else:
                        low_risks.append(item)
                elif hasattr(item, 'severity'):
                    # Pydantic model in list
                    severity = item.severity
                    item_dict = item.model_dump() if hasattr(item, 'model_dump') else item.dict()
                    if severity >= 8:
                        high_risks.append(item_dict)
                    elif severity >= 4:
                        medium_risks.append(item_dict)
                    else:
                        low_risks.append(item_dict)
            
            # Calculate overall score
            total_risks = len(high_risks) + len(medium_risks) + len(low_risks)
            if total_risks > 0:
                overall_score = min(10, (len(high_risks) * 3 + len(medium_risks) * 2 + len(low_risks)) // max(total_risks, 1) + 3)
            else:
                overall_score = 1
            
            return {
                "high_risks": high_risks,
                "medium_risks": medium_risks,
                "low_risks": low_risks,
                "overall_risk_score": overall_score
            }
        
        # Case 3: Pydantic model
        if hasattr(analysis_result, 'model_dump'):
            return analysis_result.model_dump()
        
        # Case 4: Old Pydantic model (v1)
        if hasattr(analysis_result, 'dict'):
            return analysis_result.dict()
        
        # Case 5: Has attributes we expect
        if hasattr(analysis_result, 'high_risks'):
            return {
                "high_risks": list(analysis_result.high_risks) if analysis_result.high_risks else [],
                "medium_risks": list(analysis_result.medium_risks) if hasattr(analysis_result, 'medium_risks') and analysis_result.medium_risks else [],
                "low_risks": list(analysis_result.low_risks) if hasattr(analysis_result, 'low_risks') and analysis_result.low_risks else [],
                "overall_risk_score": analysis_result.overall_risk_score if hasattr(analysis_result, 'overall_risk_score') else 5
            }
        
        # Case 6: String (possibly JSON)
        if isinstance(analysis_result, str):
            try:
                return json.loads(analysis_result)
            except json.JSONDecodeError:
                pass
        
        # Fallback: Return empty structure
        return {
            "high_risks": [],
            "medium_risks": [],
            "low_risks": [],
            "overall_risk_score": 0
        }
    
    def _check_completeness(self, result: dict) -> float:
        """Check if all required fields are present"""
        required_fields = ["overall_risk_score", "high_risks", "medium_risks", "low_risks"]
        present = sum(1 for f in required_fields if f in result and result[f] is not None)
        return present / len(required_fields)
    
    def _check_accuracy(self, result: dict) -> dict:
        """Check accuracy indicators"""
        high_risks = result.get("high_risks", [])
        
        # Handle if high_risks items are dicts or objects
        def get_field(item, field):
            if isinstance(item, dict):
                return item.get(field)
            return getattr(item, field, None)
        
        return {
            "has_specific_clauses": any(get_field(r, "clause_id") for r in high_risks),
            "has_financial_impact": any(get_field(r, "financial_impact") for r in high_risks),
            "has_recommendations": any(get_field(r, "recommended_changes") for r in high_risks)
        }
    
    def _update_metrics(self, evaluation: dict):
        """Update aggregate metrics"""
        self.metrics["contracts_analyzed"] += 1
        self.metrics["risks_identified"] += evaluation["risks_found"]
        self.metrics["high_risks_caught"] += evaluation.get("high_risk_count", 0)
    
    def get_summary(self) -> str:
        """Get evaluation summary for display"""
        avg_completeness = sum(e['completeness'] for e in self.evaluations) / max(len(self.evaluations), 1)
        avg_risks = self.metrics['risks_identified'] / max(self.metrics['contracts_analyzed'], 1)
        
        return f"""
📊 NEGOTIFY EVALUATION METRICS
================================
Contracts Analyzed: {self.metrics['contracts_analyzed']}
Total Risks Identified: {self.metrics['risks_identified']}
High Risks Caught: {self.metrics['high_risks_caught']}
Avg Risks per Contract: {avg_risks:.1f}
Avg Completeness: {avg_completeness:.1%}
"""
    
    def to_dict(self) -> dict:
        """Export metrics as dictionary"""
        return {
            "metrics": self.metrics,
            "evaluations": self.evaluations,
            "summary": {
                "total_contracts": self.metrics["contracts_analyzed"],
                "total_risks": self.metrics["risks_identified"],
                "high_risks": self.metrics["high_risks_caught"]
            }
        }