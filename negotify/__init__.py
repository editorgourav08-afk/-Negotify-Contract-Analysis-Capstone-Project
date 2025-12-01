# agents/negotify/__init__.py
"""Negotify Agent Package - Main Entry Point"""

from .agent import (
    root_agent,
    analysis_pipeline,
    benchmarking_pipeline,
    NegotiationOrchestrator,
    initialize_negotiation_pipeline,
    analyze_contract,
)

# Export the root agent for ADK
def load_agent():
    """Agent loader function for the ADK Runner."""
    return root_agent

__all__ = [
    "root_agent",
    "analysis_pipeline", 
    "benchmarking_pipeline",
    "NegotiationOrchestrator",
    "initialize_negotiation_pipeline",
    "analyze_contract",
    "load_agent",
]