# agents/negotify/tools/benchmark_tools.py
"""
Benchmark Tools for Negotify ADK Agents

These tools provide real data access for the benchmarking pipeline:
1. search_similar_clauses - Vector search in CUAD database
2. get_industry_benchmark - Freelancer/industry standard lookup
3. get_clause_alternatives - Generate negotiation alternatives

All tools are wrapped as ADK FunctionTools for agent use.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional

from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# =========================================================================
# LAZY LOADING - Initialize on first use
# =========================================================================

_vector_search = None
_freelancer_db = None


def _get_vector_search():
    """Lazy load vector search instance"""
    global _vector_search
    if _vector_search is None:
        try:
            from ..benchmarks.vector_search import NegotifyVectorSearch
            _vector_search = NegotifyVectorSearch(use_vertex_ai=False)
            logger.info("✅ Vector search initialized")
        except Exception as e:
            logger.warning(f"Could not initialize vector search: {e}")
            _vector_search = "disabled"
    return _vector_search if _vector_search != "disabled" else None


def _get_freelancer_db():
    """Lazy load freelancer benchmark database"""
    global _freelancer_db
    if _freelancer_db is None:
        try:
            from ..benchmarks.freelancer_benchmarks import FreelancerBenchmarkDB
            _freelancer_db = FreelancerBenchmarkDB()
            logger.info("✅ Freelancer benchmark DB initialized")
        except Exception as e:
            logger.warning(f"Could not initialize freelancer DB: {e}")
            _freelancer_db = "disabled"
    return _freelancer_db if _freelancer_db != "disabled" else None


# =========================================================================
# TOOL FUNCTIONS
# =========================================================================

def search_similar_clauses(
    clause_text: str,
    clause_category: Optional[str] = None,
    num_results: int = 5
) -> Dict[str, Any]:
    """
    Search for similar contract clauses in the CUAD benchmark database.
    
    This tool searches 7,500+ real contract clauses from the Contract Understanding
    Atticus Dataset (CUAD) to find similar language and provide risk context.
    
    Args:
        clause_text: The contract clause text to find similar matches for.
        clause_category: Optional category filter. One of:
            - "liability" (indemnification, caps, insurance)
            - "payment_terms" (payment timing, fees, commitments)  
            - "ip_ownership" (IP assignment, licensing, work-for-hire)
            - "termination" (notice periods, convenience termination)
            - "non_compete" (exclusivity, non-solicitation)
        num_results: Number of similar clauses to return (1-10).
    
    Returns:
        A dictionary containing:
        - similar_clauses: List of similar clauses with scores
        - risk_assessment: Overall risk assessment based on matches
        - category_distribution: Distribution of matched categories
        - benchmark_insights: Key insights from the comparison
    """
    vs = _get_vector_search()
    
    if not vs:
        return {
            "status": "error",
            "message": "Vector search not available. Using fallback analysis.",
            "similar_clauses": [],
            "risk_assessment": "Unable to benchmark - analyze clause text directly"
        }
    
    try:
        # Perform vector search
        results = vs.search(
            query=clause_text,
            category=clause_category,
            top_k=min(num_results, 10)
        )
        
        # Process results
        similar_clauses = []
        risk_levels = {"high": 0, "medium": 0, "low": 0}
        
        for r in results:
            similar_clauses.append({
                "text": r.text[:500],  # Truncate for response size
                "similarity_score": round(r.score, 4),
                "category": r.category,
                "risk_level": r.risk_level,
                "source": r.metadata.get("source_dataset", "CUAD")
            })
            risk_levels[r.risk_level] = risk_levels.get(r.risk_level, 0) + 1
        
        # Generate risk assessment
        if risk_levels["high"] >= len(results) / 2:
            risk_assessment = "HIGH - Similar clauses are frequently problematic"
        elif risk_levels["medium"] >= len(results) / 2:
            risk_assessment = "MEDIUM - Mixed risk profile in similar clauses"
        else:
            risk_assessment = "LOW - Similar clauses are typically acceptable"
        
        # Generate insights
        insights = []
        if similar_clauses:
            avg_score = sum(c["similarity_score"] for c in similar_clauses) / len(similar_clauses)
            if avg_score > 0.8:
                insights.append("This clause uses very common language - negotiation leverage may be limited")
            elif avg_score < 0.5:
                insights.append("This clause is unusual - request clarification on intent")
            
            if risk_levels["high"] > 0:
                insights.append(f"{risk_levels['high']} similar clauses flagged as high risk")
        
        return {
            "status": "success",
            "query_category": clause_category or "all",
            "similar_clauses": similar_clauses,
            "total_matches": len(similar_clauses),
            "risk_assessment": risk_assessment,
            "risk_distribution": risk_levels,
            "benchmark_insights": insights,
            "data_source": "CUAD (Contract Understanding Atticus Dataset)"
        }
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "similar_clauses": []
        }


def get_industry_benchmark(
    industry: str,
    contract_type: str = "service_agreement",
    clause_categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get industry benchmark data for contract terms.
    
    Returns real benchmark data from Freelancers Union surveys, MBO Partners reports,
    and industry-specific standards.
    
    Args:
        industry: The industry vertical. One of:
            - "technology"
            - "creative" 
            - "consulting"
            - "marketing"
            - "legal"
            - "healthcare"
            - "general"
        contract_type: Type of contract:
            - "service_agreement"
            - "consulting"
            - "creative"
            - "nda"
        clause_categories: List of categories to get benchmarks for.
            If None, returns all available benchmarks.
            Options: "payment_terms", "liability_cap", "ip_ownership",
                     "termination", "non_compete"
    
    Returns:
        A dictionary containing benchmark data for each requested category,
        including percentile rankings and recommendations.
    """
    db = _get_freelancer_db()
    
    if not db:
        return {
            "status": "error",
            "message": "Benchmark database not available",
            "benchmarks": _get_fallback_benchmarks(industry)
        }
    
    try:
        # Default to all categories
        if not clause_categories:
            clause_categories = [
                "payment_terms", "liability_cap", "ip_ownership",
                "termination", "non_compete"
            ]
        
        benchmarks = {
            "industry": industry,
            "contract_type": contract_type,
            "data_sources": [
                "Freelancers Union Annual Survey 2024",
                "MBO Partners State of Independence 2024",
                "Industry-specific contract databases"
            ]
        }
        
        # Get each requested benchmark
        if "payment_terms" in clause_categories:
            benchmarks["payment_terms"] = db.get_payment_benchmark(industry)
        
        if "liability_cap" in clause_categories:
            benchmarks["liability_cap"] = db.get_liability_benchmark(industry)
        
        if "ip_ownership" in clause_categories:
            benchmarks["ip_ownership"] = db.get_ip_benchmark(contract_type)
        
        if "termination" in clause_categories:
            benchmarks["termination"] = db.get_termination_benchmark(industry)
        
        if "non_compete" in clause_categories:
            benchmarks["non_compete"] = db.get_noncompete_benchmark(industry)
        
        return benchmarks
        
    except Exception as e:
        logger.error(f"Benchmark lookup error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "benchmarks": _get_fallback_benchmarks(industry)
        }


def get_clause_alternatives(
    original_clause: str,
    clause_category: str,
    risk_level: str = "medium",
    negotiation_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate alternative clause language for negotiation.
    
    Uses the benchmark database to suggest better alternatives based on
    industry standards and lower-risk language from the CUAD corpus.
    
    Args:
        original_clause: The problematic clause text to improve.
        clause_category: Category of the clause:
            - "liability"
            - "payment_terms"
            - "ip_ownership"
            - "termination"
            - "non_compete"
        risk_level: Current risk level of the clause (low/medium/high).
        negotiation_context: Optional context about the negotiation
            (e.g., "client is large enterprise", "long-term relationship").
    
    Returns:
        A dictionary containing:
        - alternatives: List of alternative clause suggestions
        - negotiation_tips: Specific tips for this negotiation
        - success_probability: Estimated chance of acceptance
        - fallback_position: Minimum acceptable terms
    """
    vs = _get_vector_search()
    db = _get_freelancer_db()
    
    alternatives = []
    tips = []
    
    # Get lower-risk alternatives from vector search
    if vs:
        try:
            results = vs.search(
                query=original_clause,
                category=clause_category,
                risk_level="low",  # Only get low-risk matches
                top_k=5
            )
            
            for r in results:
                if r.risk_level == "low" and r.score > 0.5:
                    alternatives.append({
                        "text": r.text,
                        "source": "CUAD lower-risk alternative",
                        "similarity": round(r.score, 3),
                        "risk_reduction": f"From {risk_level} to {r.risk_level}"
                    })
        except Exception as e:
            logger.warning(f"Could not get vector alternatives: {e}")
    
    # Get benchmark-based recommendations
    if db:
        try:
            benchmark = None
            if clause_category == "liability":
                benchmark = db.get_liability_benchmark("general")
                tips.append(f"Industry standard liability cap: {benchmark['liability_cap']['recommended']}")
                tips.append(f"Push for mutual indemnification ({benchmark['mutual_vs_unilateral']})")
                
            elif clause_category == "payment_terms":
                benchmark = db.get_payment_benchmark("general")
                tips.append(f"Standard payment terms: Net-{benchmark['net_terms_days']['recommended']} days")
                tips.append(f"Request {benchmark['deposit_percentage']['recommended']*100:.0f}% deposit")
                
            elif clause_category == "ip_ownership":
                benchmark = db.get_ip_benchmark("service_agreement")
                tips.append(f"Retain pre-existing IP and general knowledge")
                tips.append(f"Negotiate portfolio rights: {benchmark['retention_rights']}")
                
            elif clause_category == "termination":
                benchmark = db.get_termination_benchmark("general")
                tips.append(f"Minimum notice period: {benchmark['notice_period_days']['recommended']} days")
                tips.append(f"Ensure payment for work completed on termination")
                
            elif clause_category == "non_compete":
                benchmark = db.get_noncompete_benchmark("general")
                tips.append(f"Maximum duration: {benchmark['typical_duration_months']['recommended']} months")
                tips.append(f"Limit geographic scope and request carve-outs")
        except Exception as e:
            logger.warning(f"Could not get benchmark tips: {e}")
    
    # Add standard alternatives based on category
    standard_alternatives = _get_standard_alternatives(clause_category, risk_level)
    alternatives.extend(standard_alternatives)
    
    # Calculate success probability
    success_prob = _estimate_success_probability(
        risk_level, clause_category, negotiation_context
    )
    
    # Determine fallback position
    fallback = _get_fallback_position(clause_category)
    
    return {
        "status": "success",
        "original_clause": original_clause[:300],
        "clause_category": clause_category,
        "current_risk": risk_level,
        "alternatives": alternatives[:5],  # Limit to 5 best
        "negotiation_tips": tips,
        "success_probability": success_prob,
        "fallback_position": fallback,
        "recommended_approach": _get_negotiation_approach(risk_level, success_prob)
    }


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def _get_fallback_benchmarks(industry: str) -> Dict[str, Any]:
    """Provide basic benchmarks when database unavailable"""
    return {
        "payment_terms": {
            "net_terms_days": {"recommended": 30, "p50": 30},
            "deposit_percentage": {"recommended": 0.25}
        },
        "liability_cap": {
            "recommended": "1x contract value or $50,000 minimum"
        },
        "ip_ownership": {
            "recommendation": "Assign upon full payment; retain pre-existing IP"
        },
        "termination": {
            "notice_period_days": {"recommended": 14}
        },
        "non_compete": {
            "typical_duration_months": {"recommended": 6}
        },
        "source": "Default benchmarks (database unavailable)"
    }


def _get_standard_alternatives(category: str, risk_level: str) -> List[Dict[str, Any]]:
    """Get pre-defined alternative language by category"""
    
    alternatives = {
        "liability": [
            {
                "text": "Contractor's total liability under this Agreement shall not exceed the total fees paid during the 12 months preceding the claim. Neither party shall be liable for indirect, consequential, or punitive damages.",
                "source": "Standard balanced liability clause",
                "risk_level": "low"
            },
            {
                "text": "Each party shall indemnify the other against third-party claims arising from that party's gross negligence or willful misconduct, subject to the liability cap stated herein.",
                "source": "Mutual indemnification template",
                "risk_level": "low"
            }
        ],
        "payment_terms": [
            {
                "text": "Client shall pay undisputed invoices within thirty (30) days of receipt. A deposit of 25% is due upon contract execution. Late payments accrue interest at 1.5% monthly.",
                "source": "Standard freelancer payment terms",
                "risk_level": "low"
            },
            {
                "text": "Fees shall be invoiced upon milestone completion. Client shall pay within 14 days. Work may be suspended if invoices remain unpaid for 30+ days.",
                "source": "Milestone-based payment template",
                "risk_level": "low"
            }
        ],
        "ip_ownership": [
            {
                "text": "Upon full payment, Client receives all rights to final deliverables. Contractor retains ownership of pre-existing tools, methodologies, and general knowledge, granting Client a license for project use.",
                "source": "Balanced IP assignment clause",
                "risk_level": "low"
            },
            {
                "text": "Work product is assigned to Client upon payment. Contractor retains the right to display work in portfolio and use anonymized case studies.",
                "source": "Creative industry standard",
                "risk_level": "low"
            }
        ],
        "termination": [
            {
                "text": "Either party may terminate with 14 days written notice. Upon termination, Client pays for all completed work. Contractor delivers all work product paid for.",
                "source": "Mutual termination clause",
                "risk_level": "low"
            },
            {
                "text": "Termination for convenience requires 30 days notice. Material breach triggers 10-day cure period. On termination, all paid deliverables transfer to Client.",
                "source": "Standard termination template",
                "risk_level": "low"
            }
        ],
        "non_compete": [
            {
                "text": "During the engagement, Contractor will not perform substantially similar services for Client's direct competitors in the same market segment.",
                "source": "Limited non-compete clause",
                "risk_level": "low"
            },
            {
                "text": "For 6 months post-termination, Contractor agrees not to solicit Client's employees. No restrictions on serving other clients.",
                "source": "Non-solicitation only template",
                "risk_level": "low"
            }
        ]
    }
    
    return alternatives.get(category, [])


def _estimate_success_probability(
    risk_level: str, 
    category: str, 
    context: Optional[str]
) -> Dict[str, Any]:
    """Estimate probability of successful negotiation"""
    
    base_prob = {
        "high": 0.55,
        "medium": 0.70,
        "low": 0.85
    }.get(risk_level, 0.65)
    
    # Category adjustments
    category_adj = {
        "liability": 0.05,  # Liability often negotiable
        "payment_terms": 0.10,  # Payment terms frequently adjusted
        "ip_ownership": -0.05,  # IP can be contentious
        "termination": 0.05,  # Usually negotiable
        "non_compete": -0.10  # Often non-negotiable for client
    }.get(category, 0)
    
    final_prob = min(0.95, max(0.30, base_prob + category_adj))
    
    return {
        "overall": round(final_prob, 2),
        "confidence": "medium",
        "factors": [
            f"Base probability for {risk_level} risk: {base_prob:.0%}",
            f"Category adjustment ({category}): {category_adj:+.0%}"
        ]
    }


def _get_fallback_position(category: str) -> str:
    """Get minimum acceptable terms for negotiation"""
    
    fallbacks = {
        "liability": "Liability cap of at least total fees paid; mutual indemnification",
        "payment_terms": "Net-45 maximum; 10% deposit minimum for new clients",
        "ip_ownership": "Assignment on full payment; retain portfolio rights",
        "termination": "7-day minimum notice; payment for work completed",
        "non_compete": "Limited to direct competitors; 12-month maximum duration"
    }
    
    return fallbacks.get(category, "Case-by-case evaluation required")


def _get_negotiation_approach(risk_level: str, success_prob: Dict) -> str:
    """Recommend negotiation approach"""
    
    prob = success_prob.get("overall", 0.5)
    
    if risk_level == "high" and prob < 0.5:
        return "ESCALATE - Consider walking away or requiring major concessions"
    elif risk_level == "high":
        return "FIRM STANCE - Present alternatives as requirements, not requests"
    elif prob > 0.7:
        return "COLLABORATIVE - Frame as mutual benefit; offer compromise"
    else:
        return "BALANCED - Present alternatives with rationale; be prepared to concede"


# =========================================================================
# ADK FUNCTION TOOL WRAPPERS
# =========================================================================

# Create FunctionTool instances for ADK agent use
benchmark_search_tool = FunctionTool(func=search_similar_clauses)
industry_benchmark_tool = FunctionTool(func=get_industry_benchmark)
clause_alternatives_tool = FunctionTool(func=get_clause_alternatives)


# =========================================================================
# STANDALONE TESTING
# =========================================================================

if __name__ == "__main__":
    print("🧪 Testing Benchmark Tools\n" + "="*50)
    
    # Test 1: Similar clause search
    print("\n1️⃣ Testing search_similar_clauses...")
    result = search_similar_clauses(
        clause_text="The contractor agrees to indemnify and hold harmless the client from all claims without limitation.",
        clause_category="liability",
        num_results=3
    )
    print(f"   Status: {result.get('status')}")
    print(f"   Matches: {result.get('total_matches', 0)}")
    print(f"   Risk: {result.get('risk_assessment')}")
    
    # Test 2: Industry benchmark
    print("\n2️⃣ Testing get_industry_benchmark...")
    result = get_industry_benchmark(
        industry="technology",
        contract_type="service_agreement",
        clause_categories=["payment_terms", "liability_cap"]
    )
    print(f"   Industry: {result.get('industry')}")
    if "payment_terms" in result:
        print(f"   Payment: Net-{result['payment_terms'].get('net_terms_days', {}).get('recommended', 'N/A')} days")
    
    # Test 3: Clause alternatives
    print("\n3️⃣ Testing get_clause_alternatives...")
    result = get_clause_alternatives(
        original_clause="Contractor shall indemnify Client from all claims without any limit.",
        clause_category="liability",
        risk_level="high"
    )
    print(f"   Alternatives: {len(result.get('alternatives', []))}")
    print(f"   Success prob: {result.get('success_probability', {}).get('overall', 'N/A')}")
    print(f"   Tips: {len(result.get('negotiation_tips', []))}")
    
    print("\n✅ All benchmark tool tests complete!")