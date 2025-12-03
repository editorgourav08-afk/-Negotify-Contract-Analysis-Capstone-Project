# agents/negotify/tools/__init__.py
"""Negotify Custom Tools"""

from .custom_tools import (
    PDFExtractionTool,
    pdf_extraction_tool,
    GmailIntegrationTool,
    evaluate_contract_analysis,
)

#  Benchmark tools
from .benchmark_tools import (
    search_similar_clauses,
    get_industry_benchmark,
    get_clause_alternatives,
    benchmark_search_tool,
    industry_benchmark_tool,
    clause_alternatives_tool,
)

# Add these imports to tools/__init__.py

from .gmail_pro import (
    # Functions
    create_negotiation_draft,
    review_draft,
    edit_draft,
    rewrite_draft_with_feedback,
    approve_draft,
    send_approved_email,
    cancel_draft,
    get_draft_status,
    # ADK Tools
    create_draft_tool,
    review_draft_tool,
    edit_draft_tool,
    rewrite_draft_tool,
    approve_draft_tool,
    send_email_tool,
    cancel_draft_tool,
    draft_status_tool,
)
__all__ = [
    
    "PDFExtractionTool",
    "pdf_extraction_tool",
    "GmailIntegrationTool",
    "evaluate_contract_analysis",
    #  - Benchmark tools
    "search_similar_clauses",
    "get_industry_benchmark",
    "get_clause_alternatives",
    "benchmark_search_tool",
    "industry_benchmark_tool",
    "clause_alternatives_tool",
    
    "create_draft_tool",
    "review_draft_tool",
    "edit_draft_tool",
    "rewrite_draft_tool",
    "approve_draft_tool",
    "send_email_tool",
    "cancel_draft_tool",
    "draft_status_tool",
]