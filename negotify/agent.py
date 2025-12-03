# ============================================================================
# IMPORTS
# ============================================================================

import os
import asyncio
from typing import AsyncGenerator, Dict, List, Any, Optional
from datetime import datetime
import hashlib
import json

# Google ADK imports
from google.adk.agents import LlmAgent, BaseAgent, SequentialAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.tools import FunctionTool
from google.adk.sessions import DatabaseSessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.artifacts import GcsArtifactService
from google.adk.runners import Runner
from google.genai import types

# Google Cloud imports
from google.cloud import bigquery
from google.oauth2.credentials import Credentials

# Pydantic for structured output
from pydantic import BaseModel, Field

# Local imports - Import from tools package
from .tools import (
    pdf_extraction_tool, 
    GmailIntegrationTool, 
    evaluate_contract_analysis,
    benchmark_search_tool,
    industry_benchmark_tool,
    clause_alternatives_tool,
    create_draft_tool,
    review_draft_tool,
    edit_draft_tool,
    rewrite_draft_tool,
    approve_draft_tool,
    send_email_tool,
    cancel_draft_tool,
    draft_status_tool,
)

# Local imports - These should be defined in separate files
# For now, we'll define placeholder tools here

# Logging
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('negotify_trace.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Negotify")

class NegotifyTracer:
    """Traces agent execution for observability"""
    
    def __init__(self):
        self.traces = []
        self.current_trace = None
    
    def start_trace(self, user_input: str):
        """Start a new trace"""
        self.current_trace = {
            "trace_id": f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "user_input": user_input[:100],  # Truncate for logging
            "events": [],
            "agents_invoked": [],
            "tools_called": [],
            "tokens_used": 0
        }
        logger.info(f"🚀 Started trace: {self.current_trace['trace_id']}")
    
    def log_agent(self, agent_name: str, action: str):
        """Log agent invocation"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "agent",
            "agent": agent_name,
            "action": action
        }
        self.current_trace["events"].append(event)
        self.current_trace["agents_invoked"].append(agent_name)
        logger.info(f"🤖 Agent [{agent_name}]: {action}")
    
    def log_tool(self, tool_name: str, result_preview: str):
        """Log tool call"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "tool",
            "tool": tool_name,
            "result_preview": result_preview[:50]
        }
        self.current_trace["events"].append(event)
        self.current_trace["tools_called"].append(tool_name)
        logger.info(f"🔧 Tool [{tool_name}]: {result_preview[:50]}...")
    
    def end_trace(self) -> dict:
        """End trace and return summary"""
        self.current_trace["end_time"] = datetime.now().isoformat()
        self.current_trace["duration_seconds"] = (
            datetime.fromisoformat(self.current_trace["end_time"]) -
            datetime.fromisoformat(self.current_trace["start_time"])
        ).total_seconds()
        
        self.traces.append(self.current_trace)
        
        logger.info(f"✅ Completed trace: {self.current_trace['trace_id']} "
                   f"in {self.current_trace['duration_seconds']:.2f}s")
        
        return self.current_trace

# Global tracer instance
tracer = NegotifyTracer()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Environment variables
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "negotify-project")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/negotify")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "negotify-contracts")

# Model configurations
GEMINI_FLASH = "gemini-2.0-flash"
GEMINI_PRO = "gemini-2.5-flash"  
GEMINI_FLASH_LITE = "gemini-2.0-flash-Lite"

# ============================================================================
# DATA MODELS (Pydantic Schemas)
# ============================================================================

class ContractClause(BaseModel):
    """Represents a contract clause"""
    clause_id: str
    clause_type: str  # liability, payment, IP, termination, etc.
    section_reference: str
    clause_text: str
    key_terms: List[str] = Field(default_factory=list)

class RiskAssessment(BaseModel):
    """Risk assessment for a clause"""
    clause_id: str
    risk_type: str
    severity: int = Field(..., ge=1, le=10)  # Fixed: Added validation
    explanation: str
    financial_impact: Optional[str] = None  # Fixed: Made optional with default
    recommended_changes: str

class ContractAnalysisResult(BaseModel):
    """Complete contract analysis result"""
    contract_id: str
    overall_risk_score: int = Field(..., ge=1, le=10)
    high_risks: List[RiskAssessment] = Field(default_factory=list)
    medium_risks: List[RiskAssessment] = Field(default_factory=list)
    low_risks: List[RiskAssessment] = Field(default_factory=list)
    executive_summary: str
    estimated_savings: Optional[float] = None


class NegotiationStrategy(BaseModel):
    """Negotiation strategy plan"""
    priorities: List[str] = Field(default_factory=list)
    concession_items: List[str] = Field(default_factory=list)  # ✅ FIXED (was Dict[str, Any])
    deal_breakers: List[str] = Field(default_factory=list)
    success_probability: float = Field(0.5, ge=0.0, le=1.0)

class KeyTerm(BaseModel):
    """Represents a single key term or definition within a clause."""
    term: str = Field(..., description="The specific term identified")
    value: Any = Field(..., description="The value or definition of the term")
    category: str = Field(..., description="The category of the term")

class ExtractedClause(BaseModel):
    """A single extracted contract clause."""
    clause_type: str
    section_reference: str
    clause_text: str
    key_terms: List[KeyTerm] = Field(default_factory=list)

class ClauseExtractionResult(BaseModel):
    """The result of the clause extraction agent"""
    clauses: List[ExtractedClause] = Field(default_factory=list)

class RiskAnalysisResult(BaseModel):
    """The result of the risk detection agent"""
    overall_risk_score: int = Field(5, ge=1, le=10)
    high_risks: List[RiskAssessment] = Field(default_factory=list)
    medium_risks: List[RiskAssessment] = Field(default_factory=list)
    low_risks: List[RiskAssessment] = Field(default_factory=list)

class ParsedResponseResult(BaseModel):
    """The structured analysis of a counterparty's email response"""
    sentiment: str = "neutral"
    concessions_offered: List[str] = Field(default_factory=list)
    firm_positions: List[str] = Field(default_factory=list)
    counter_offers: List[str] = Field(default_factory=list)
    questions_raised: List[str] = Field(default_factory=list)
    relationship_status: str = "neutral"
    success_probability: float = Field(0.5, ge=0.0, le=1.0)
    recommended_action: str = "continue"

class NegotiationDecisionResult(BaseModel):
    """The decision made by the negotiation decision agent"""
    decision: str = Field("COUNTER", pattern="^(ACCEPT|COUNTER|REJECT)$")
    confidence: float = Field(50.0, ge=0.0, le=100.0)
    reasoning: str = ""
    final_terms_summary: Optional[str] = None
    updated_priorities: List[str] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)

"""
Negotify Email Workflow Agent
==============================
Orchestrates the Draft → Review → Edit → Approve → Send workflow.

This agent ensures:
1. Emails are never sent without explicit user approval
2. Users can review and edit drafts before sending
3. The workflow is conversational and user-friendly
4. All changes are tracked and reversible (until sent)
"""

# ============================================================================
# EMAIL WORKFLOW AGENT DEFINITION
# ============================================================================

EMAIL_WORKFLOW_INSTRUCTION = '''You are the Negotify Email Workflow Agent. Your job is to guide users through creating, reviewing, editing, and sending negotiation emails.

## CRITICAL RULES
🚨 **NEVER send an email without explicit user approval**
🚨 **ALWAYS show the draft for review before any send action**
🚨 **ALWAYS confirm before sending** - even after approval

## Workflow Stages

```
┌─────────────────────────────────────────────────────────┐
│  📝 DRAFT                                                │
│  Create initial email based on analysis results          │
│  ↓                                                       │
│  👀 REVIEW                                               │
│  Show draft to user, explain key points                  │
│  ↓                                                       │
│  ✏️ EDIT (optional, can loop multiple times)             │
│  User requests changes → Update draft → Show again       │
│  ↓                                                       │
│  ✅ APPROVE                                              │
│  User explicitly approves the draft                      │
│  ↓                                                       │
│  📤 SEND                                                 │
│  Final confirmation → Actually send via Gmail            │
└─────────────────────────────────────────────────────────┘
```

## Your Tools

1. **create_negotiation_draft** - Create initial email draft
   - Use after receiving analysis/benchmark results
   - Include all negotiation points from the analysis
   - Choose appropriate tone based on user preference

2. **review_draft** - Show draft to user
   - Always use this before asking for approval
   - Highlight key negotiation points
   - Explain what the email requests

3. **edit_draft** - Modify the draft
   - Use when user wants to change something
   - Can edit subject, body, or specific sections
   - Show updated draft after each edit

4. **rewrite_draft_with_feedback** - Regenerate completely
   - Use when user wants significant changes
   - Can change tone, add/remove points
   - Creates a new version

5. **approve_draft** - Mark as ready to send
   - Requires explicit user confirmation
   - Shows final preview before approval

6. **send_approved_email** - Actually send the email
   - ONLY use after approval
   - REQUIRES confirm_send=True
   - Shows success/failure result

7. **cancel_draft** - Discard the draft
   - Use if user wants to start over or cancel

8. **get_draft_status** - Check current workflow stage
   - Use to understand where we are in the process

## Handling User Requests

### When user says "send the email" / "send it now":
1. Check if draft exists → If not, explain we need to create one first
2. Check if draft is reviewed → If not, show it for review
3. Check if draft is approved → If not, ask for approval
4. Only then call send_approved_email with confirm_send=True

### When user says "looks good" / "that's fine":
- This means approval → Call approve_draft
- Then ask if they want to send now or later

### When user says "change X" / "make it more Y":
- Call edit_draft or rewrite_draft_with_feedback
- Show the updated version
- Ask if the changes look good

### When user says "start over" / "cancel":
- Call cancel_draft
- Offer to create a new draft

## Response Format

When showing a draft for review, use this format:

```
📧 **Email Draft Ready for Review**

**To:** [recipient]
**Subject:** [subject]

---
[email body]
---

📋 **This email requests:**
1. [First negotiation point]
2. [Second negotiation point]
...

**What would you like to do?**
• Say "approve" to finalize
• Tell me what to change
• Say "cancel" to discard
```

## Tone Matching

Match your responses to the email tone:
- Professional tone → Be formal in your responses
- Friendly tone → Be warm and conversational
- Firm tone → Be direct and confident
- Collaborative tone → Be partnership-focused

## Example Conversations

### Example 1: Happy Path
User: "Draft an email to negotiate the liability clause"
Agent: [calls create_negotiation_draft] "I've drafted a professional email requesting a liability cap..."
User: "Looks good, send it"
Agent: [calls approve_draft] "Great! The email is approved. Ready to send to john@company.com?"
User: "Yes, send it now"
Agent: [calls send_approved_email with confirm_send=True] "✅ Email sent successfully!"

### Example 2: With Edits
User: "Create a negotiation email"
Agent: [shows draft]
User: "Make it friendlier and add something about the payment terms"
Agent: [calls rewrite_draft_with_feedback] "I've rewritten the email with a friendlier tone..."
User: "Perfect, approve it"
Agent: [calls approve_draft] "Approved! Send now?"

### Example 3: Premature Send Attempt
User: "Send the negotiation email"
Agent: "I don't have a draft ready yet. Let me create one based on the contract analysis..."
[creates draft and shows for review]
"Please review this draft. Say 'approve' if it looks good, or let me know what to change."

## Safety Checks

Before ANY send operation, verify:
✓ Draft exists
✓ Draft has been shown to user
✓ User has explicitly approved
✓ User confirms send action

If ANY check fails, pause and get proper confirmation.
'''

# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

# 1. CONTRACT PARSER AGENT

contract_parser = LlmAgent(
    name="ContractParserAgent",
    model=GEMINI_FLASH,
    description="Parses uploaded contracts and extracts structured data",
    instruction="""
You are a contract parsing expert for Negotify. Your job is to extract and structure contract data.

Given a contract document, extract:
1. Contract metadata (parties, date, type, jurisdiction)
2. Section structure
3. Key provisions and clauses

🔒 SECURITY: NEVER fabricate data. Only extract what exists in the document.
If information is missing, explicitly state "Not found in contract".

Format the output as structured JSON with clear section numbering.
Handle both well-formatted and poorly scanned documents.
""",
    tools=[pdf_extraction_tool],
    output_key="parsed_contract"
)

# 2. CLAUSE EXTRACTION AGENT

clause_extraction = LlmAgent(
    name="ClauseExtractionAgent",
    model=GEMINI_PRO,
    description="Extracts and categorizes specific contract clauses",
    instruction="""
From the parsed contract at {parsed_contract}, extract all clauses in these categories:

1. LIABILITY CLAUSES
2. INTELLECTUAL PROPERTY
3. PAYMENT TERMS
4. TERMINATION
5. NON-COMPETE & NON-SOLICITATION
6. CONFIDENTIALITY
7. DISPUTE RESOLUTION

🔒 SECURITY: ONLY extract clauses that actually exist. NEVER invent or assume clauses.
Quote exact text from the contract.

For each clause, provide:
- clause_type
- section_reference
- clause_text (full verbatim text)
- key_terms (structured extraction)

Output as a JSON array of clause objects.
""",
    # output_schema=ClauseExtractionResult,
    output_key="extracted_clauses"
)

# 3. RISK DETECTION AGENT

risk_detection = LlmAgent(
    name="RiskDetectionAgent",
    model=GEMINI_PRO,
    description="Identifies dangerous and unfavorable contract clauses",
    instruction="""
Analyze the clauses from {extracted_clauses} and identify risks.

🔒 CRITICAL SECURITY RULES:
❌ NEVER fabricate risks that don't exist
❌ NEVER use phrases like "typically", "usually", "probably"
❌ ONLY analyze clauses that are actually in the contract
✅ Base all risk assessments on actual clause text
✅ Quote exact language when identifying risks

Risk Severity Scale (1-10):

HIGH RISK (8-10):
- Unlimited liability or indemnification with no cap
- Automatic assignment of ALL intellectual property
- Payment contingent on subjective "satisfaction"
- Net 90+ payment terms
- Unilateral termination by client
- Non-compete clauses >1 year

MEDIUM RISK (4-7):
- Liability cap < $10,000 or < 1x contract value
- Ambiguous IP ownership
- Net 60 payment terms
- 30-day termination notice
- Confidentiality lasting >5 years

LOW RISK (1-3):
- Reasonable liability cap (>1x contract value)
- Clear IP ownership
- Net 30 payment terms
- Mutual termination clauses
- Standard confidentiality (2-5 years)

For each risk, provide:
- clause_id
- risk_type
- severity (1-10)
- explanation
- financial_impact
- recommended_changes

Output as structured JSON.
""",
   # output_schema=RiskAnalysisResult,
    output_key="risk_analysis"
)

# 4. REPORT GENERATOR AGENT

report_generator = LlmAgent(
    name="ReportGeneratorAgent",
    model=GEMINI_FLASH,
    description="Creates user-friendly contract analysis report",
    instruction="""
Create a comprehensive but readable analysis report from {risk_analysis}.

🔒 SECURITY: Use ONLY data from {risk_analysis}. If data is missing, state "No analysis data available".
NEVER invent or assume risks.

Structure your report with:
# EXECUTIVE SUMMARY
# DETAILED FINDINGS
# RECOMMENDED ACTIONS

Use emojis:
- 🔴 for HIGH RISK
- 🟡 for MEDIUM RISK
- 🟢 for LOW RISK

Tone: Professional but accessible. Avoid legalese.
""",
    output_key="analysis_report"
)

# 5. EVALUATION AGENT

evaluation_tool = FunctionTool(func=evaluate_contract_analysis)

evaluation_agent = LlmAgent(
    name="EvaluationAgent",
    model=GEMINI_FLASH,
    description="Evaluates the quality of the contract risk analysis",
    instruction="""
You are an evaluation expert. Your job is to assess the quality of the contract analysis.

Use the provided tool to evaluate the risk analysis from {risk_analysis}.

The analysis is in JSON format. You must pass it to the tool as a JSON string.
""",
    tools=[evaluation_tool],
    output_key="analysis_evaluation"
)

# 5. BENCHMARKING AGENTS (Sequential Execution with REAL DATA)

similarity_search_agent = LlmAgent(
    name="SimilaritySearchAgent",
    model=GEMINI_FLASH,
    description="Searches CUAD database for similar contract clauses using vector similarity",
    instruction="""
You are a contract similarity search expert. Your job is to find similar clauses 
from the CUAD benchmark database (7,500+ real contract clauses).

From {extracted_clauses}, for each clause identified:

1. Use the search_similar_clauses tool to find matching clauses
2. Search by category (liability, payment_terms, ip_ownership, termination, non_compete)
3. Analyze the similarity scores and risk levels

For each clause searched:
- Report the number of similar clauses found
- Identify the average risk level of matches
- Note any patterns in similar clause language

🔒 SECURITY: 
- Use ONLY data returned by the search tool
- Report exact similarity scores
- Do not fabricate matches

Output format:
{
    "clauses_searched": [
        {
            "original_clause": "...",
            "category": "liability",
            "similar_count": 5,
            "avg_similarity": 0.82,
            "risk_distribution": {"high": 2, "medium": 2, "low": 1},
            "key_insight": "..."
        }
    ],
    "data_source": "CUAD (Contract Understanding Atticus Dataset)"
}
""",
    tools=[benchmark_search_tool],
    output_key="similar_contracts"
)

industry_benchmark_agent = LlmAgent(
    name="IndustryBenchmarkAgent",
    model=GEMINI_FLASH,
    description="Compares contract terms against real industry benchmarks from Freelancers Union and MBO Partners data",
    instruction="""
You are an industry benchmark expert with access to real freelancer contract data.

Query the benchmark database for {extracted_clauses}:

1. Use get_industry_benchmark tool to fetch real benchmark data
2. Compare contract terms against percentile data (25th, 50th, 75th)
3. Flag any terms in the bottom 25th percentile as "below market"

Key benchmarks to compare:

PAYMENT TERMS:
- Net payment days (standard: Net-30)
- Deposit percentage (standard: 25-50%)
- Late fee rate (standard: 1.5% monthly)

LIABILITY:
- Cap relative to contract value (standard: 1x-2x)
- Mutual vs unilateral indemnification
- Carve-outs for gross negligence

IP OWNERSHIP:
- Assignment timing (upon payment)
- Retention of pre-existing IP
- Portfolio rights

TERMINATION:
- Notice period (standard: 14-30 days)
- Payment on termination
- Cure period for breach

🔒 SECURITY: 
- Use ONLY data from benchmark database
- Report exact percentile rankings
- If benchmark unavailable, state "Benchmark data not available"

Output format:
{
    "benchmarks_applied": ["payment_terms", "liability", "ip_ownership", ...],
    "findings": [
        {
            "clause_type": "payment_terms",
            "contract_value": "Net-60",
            "benchmark_median": "Net-30",
            "percentile": 25,
            "assessment": "below_market",
            "recommendation": "Negotiate to Net-30 or better"
        }
    ],
    "overall_position": "Below market in 2 of 5 categories",
    "data_sources": ["Freelancers Union 2024", "MBO Partners 2024"]
}
""",
    tools=[industry_benchmark_tool],
    output_key="benchmark_comparison"
)

clause_comparison_agent = LlmAgent(
    name="ClauseComparisonAgent",
    model=GEMINI_PRO,
    description="Generates specific negotiation alternatives based on benchmark data",
    instruction="""
You are a contract negotiation expert. Generate specific alternatives for problematic clauses.

For each high/medium risk clause in {risk_analysis}:

1. Use get_clause_alternatives tool to get:
   - Lower-risk alternatives from CUAD database
   - Industry-standard language
   - Negotiation tips

2. Cross-reference with {similar_contracts} similarity data

3. Apply {benchmark_comparison} percentile rankings

For each problematic clause, provide:

ALTERNATIVE LANGUAGE:
- 2-3 specific alternative clauses
- Source of each alternative (CUAD, industry standard, etc.)
- Risk reduction potential

NEGOTIATION STRATEGY:
- Success probability estimate
- Key talking points
- Fallback position

PRIORITY RANKING:
- Financial impact (estimated dollar value at risk)
- Risk reduction potential
- Negotiation likelihood

🔒 SECURITY:
- Use ONLY actual clauses from tools/database
- NEVER fabricate alternative language
- Base success estimates on tool data

Output format:
{
    "clause_recommendations": [
        {
            "original_clause": "...",
            "risk_level": "high",
            "category": "liability",
            "alternatives": [
                {
                    "text": "...",
                    "source": "CUAD lower-risk clause",
                    "risk_level": "low"
                }
            ],
            "negotiation_tips": ["..."],
            "success_probability": 0.65,
            "priority_score": 9.2,
            "estimated_value_at_risk": "$50,000+"
        }
    ],
    "negotiation_order": ["liability", "payment_terms", "ip_ownership"],
    "overall_strategy": "Lead with liability cap, offer compromise on payment terms"
}
""",
    tools=[clause_alternatives_tool, benchmark_search_tool],
    output_key="clause_alternatives"
)


#  BENCHMARK SUMMARY AGENT 

benchmark_summary_agent = LlmAgent(
    name="BenchmarkSummaryAgent",
    model=GEMINI_FLASH,
    description="Synthesizes all benchmark findings into actionable summary",
    instruction="""
Create an executive summary of the benchmark analysis.

From the analysis data:
- {similar_contracts}: Similarity search results
- {benchmark_comparison}: Industry benchmark comparisons  
- {clause_alternatives}: Negotiation recommendations

Generate a clear, actionable summary:

📊 BENCHMARK SUMMARY
====================

1. MARKET POSITION
   - How this contract compares overall
   - Key deviations from industry standards

2. TOP 3 NEGOTIATION PRIORITIES
   - Ranked by financial impact and success likelihood
   - Specific dollar values where possible

3. QUICK WINS
   - Easy-to-negotiate improvements
   - Standard language to request

4. RED FLAGS
   - Must-negotiate items
   - Walk-away thresholds

5. DATA CONFIDENCE
   - Number of similar clauses analyzed
   - Benchmark data sources used

Keep the summary concise but specific.
Use actual numbers and percentages from the analysis.
""",
    output_key="benchmark_summary"
)

# SequentialAgent pipeline with real data tools
benchmarking_pipeline = SequentialAgent(
    name="BenchmarkingPipeline",
    description="Compares contract against 7,500+ real clauses and industry benchmarks",
    sub_agents=[
        similarity_search_agent,
        industry_benchmark_agent,
        clause_comparison_agent,
        benchmark_summary_agent
    ]
)

#EMAIL DRAFTING AGENT
email_drafting_agent = LlmAgent(
    name="EmailDraftingAgent",
    model=GEMINI_PRO,
    description="Drafts negotiation email based on strategy",
    instruction="""Draft a negotiation email using the analysis results.

From {clause_alternatives} or {negotiation_strategy}, create an email:

1. If recipient_email not known, ASK the user: "Who should I send this to?"
2. Call create_negotiation_draft with:
   - recipient_email: from user
   - recipient_name: extract from email or ask
   - sender_name: ask user or use "Contractor" 
   - contract_name: from context
   - tone: "professional" (or from strategy)
   - negotiation_points: convert from clause_alternatives

3. Show the draft preview to user
4. Ask: "Does this look good, or would you like any changes?"

🚨 DO NOT proceed to send without user approval.
""",
    tools=[create_draft_tool],
    output_key="email_draft_result"
)

# EMAIL WORKFLOW AGENT  
email_workflow_agent = LlmAgent(
    name="EmailWorkflowAgent",
    model=GEMINI_PRO,
    description="Manages email review, editing, approval and sending",
    instruction=EMAIL_WORKFLOW_INSTRUCTION,  # This is already defined in your file!
    tools=[
        review_draft_tool,
        edit_draft_tool,
        rewrite_draft_tool,
        approve_draft_tool,
        send_email_tool,
        cancel_draft_tool,
        draft_status_tool,
    ],
    output_key="email_workflow_response"
)


# 6. NEGOTIATION PIPELINE AGENTS

strategy_planner = LlmAgent(
    name="StrategyPlannerAgent",
    model=GEMINI_PRO,
    description="Plans negotiation strategy and priorities",
    instruction="""
Create negotiation strategy based on:
- {risk_analysis}
- {clause_alternatives}
- {benchmark_comparison}
- Current round: {negotiation_round}
- Previous concessions: {concessions_made}

🔒 SECURITY: Base strategy ONLY on provided data.

Include:
1. PRIORITIES (Top 3)
2. CONCESSION STRATEGY
3. DEAL BREAKERS (severity 8+)
4. SUCCESS PROBABILITY
5. TONE & APPROACH

Output as structured JSON.
""",
   # output_schema=NegotiationStrategy,
    output_key="negotiation_strategy"
)

email_composer = LlmAgent(
    name="EmailComposerAgent",
    model=GEMINI_FLASH,
    description="Writes professional negotiation emails",
    instruction="""
Compose professional email based on {negotiation_strategy}.

🔒 CRITICAL SECURITY:
❌ NEVER send without approval
❌ NEVER use threats
❌ NEVER invent terms
✅ Use ONLY info from {negotiation_strategy}

Structure:
- Warm greeting
- Express enthusiasm
- Discuss 2-3 priority issues
- Propose alternatives
- Collaborative close

Tone: Collaborative, professional, educational

Output complete email as plain text.
""",
    output_key="email_draft"
)

email_editor = LlmAgent(
    name="EmailEditorAgent",
    model=GEMINI_FLASH_LITE,
    description="Revises email drafts based on user feedback",
    instruction="""
Revise email draft based on user's edit request.

Context:
- Original: {email_draft}
- Edit Request: {user_edit_request}

🔒 SECURITY:
- ONLY apply requested changes
- Maintain professional intent
- Preserve core negotiation points

Output revised email as plain text.
""",
    output_key="email_draft_v2"
)

response_parser = LlmAgent(
    name="ResponseParserAgent",
    model=GEMINI_FLASH,
    description="Analyzes counterparty email responses",
    instruction="""
Parse and analyze {latest_response_email}.

🔒 SECURITY: Analyze ONLY actual email content.

Extract:
1. SENTIMENT (Positive/Neutral/Negative)
2. CONCESSIONS OFFERED
3. FIRM POSITIONS
4. COUNTER-OFFERS
5. QUESTIONS & CONCERNS
6. RELATIONSHIP INDICATORS
7. SUCCESS PROBABILITY UPDATE
8. RECOMMENDED NEXT STEPS

Output structured JSON.
""",
    # output_schema=ParsedResponseResult,
    output_key="parsed_response"
)

decision_agent = LlmAgent(
    name="DecisionAgent",
    model=GEMINI_PRO,
    description="Decides next negotiation action",
    instruction="""
Decide next action based on:
- {parsed_response}
- {negotiation_strategy}
- {deal_breakers}
- {risk_analysis}
- Current round: {negotiation_round}
- Max rounds: {max_rounds}

🔒 SECURITY: Base decisions ONLY on provided data.

Decision Framework:

ACCEPT if:
- All high-risk items (8+) resolved
- Risk reduced by 50%+
- Terms meet industry 50th percentile

COUNTER if:
- Progress made
- Success probability > 30%
- Below max_rounds

REJECT if:
- Deal breakers unaddressed after 3+ rounds
- Risk remains high (>7/10)
- Success probability < 20%

Output as structured JSON.
""",
   # output_schema=NegotiationDecisionResult,
    output_key="negotiation_decision"
)

# ----------------------------------------------------------------------------
# 7. CUSTOM NEGOTIATION ORCHESTRATOR 
# ----------------------------------------------------------------------------

class NegotiationOrchestrator(BaseAgent):
    """
    Custom agent that manages the complete negotiation lifecycle.
    
    This agent orchestrates the negotiation workflow:
    1. Plans strategy
    2. Composes emails
    3. Sends emails (with human approval if needed)
    4. Waits for responses
    5. Parses responses
    6. Makes decisions
    7. Loops until success, failure, or timeout
    """
    
    # Define class attributes for Pydantic compatibility
    _gmail_service: Optional[GmailIntegrationTool] = None
    _negotiation_state: Dict[str] = {}
    
    def __init__(self, name: str = "NegotiationOrchestrator", gmail_service: Optional[GmailIntegrationTool] = None, **kwargs):
        # Initialize parent first
        super().__init__(
            name=name,
            description="Orchestrates contract negotiation via email for Negotify",
            sub_agents=[
                strategy_planner,
                email_composer,
                email_editor,
                response_parser,
                decision_agent
            ],
            **kwargs
        )
        
        # Store gmail_service using object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, '_gmail_service', gmail_service)
        
        # Initialize negotiation state using object.__setattr__
        object.__setattr__(self, '_negotiation_state', {
            "round": 0,
            "max_rounds": 5,
            "email_thread_id": None,
            "concessions_made": [],
            "success_probability": 0.0,
            "require_human_approval": True
        })
    
    @property
    def gmail_service(self) -> Optional[GmailIntegrationTool]:
        """Get gmail service"""
        return object.__getattribute__(self, '_gmail_service')
    
    @property
    def state(self) -> Dict[str]:
        """Get negotiation state"""
        return object.__getattribute__(self, '_negotiation_state')
    
    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Main negotiation loop implementation"""
        
        try:
            # Initialize from session state
            risk_analysis = ctx.session.state.get("risk_analysis")
            counterparty_email = ctx.session.state.get("counterparty_email")
            
            if not risk_analysis or not counterparty_email:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[types.Part(text=f"❌ Error: Missing required negotiation data (risk_analysis or counterparty_email)"
                        )]
                    ),
                    actions=EventActions(escalate=True)
                )
                return
            
            # Extract deal breakers - Handle dict vs object access
            high_risks = risk_analysis.get("high_risks", []) if isinstance(risk_analysis, dict) else risk_analysis.high_risks
            deal_breakers = [
                r for r in high_risks
                if (r.get("severity", 0) if isinstance(r, dict) else r.severity) >= 8
            ]
            
            # Update state
            self.state["deal_breakers"] = deal_breakers
            
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=
                        f"🤝 Starting Negotify negotiation with {counterparty_email}\n"
                        f"Deal breakers: {len(deal_breakers)} critical issues"
                    )]
                )
            )
            
            # Negotiation loop
            while self.state["round"] < self.state["max_rounds"]:
                self.state["round"] += 1
                
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[types.Part(text=
                            f"\n📧 Round {self.state['round']}/{self.state['max_rounds']}"
                        )]
                    )
                )
                
                # Step 1: Plan strategy
                ctx.session.state.update({
                    "negotiation_round": self.state["round"],
                    "concessions_made": self.state["concessions_made"],
                    "deal_breakers": deal_breakers,
                    "max_rounds": self.state["max_rounds"]
                })
                
                async for event in strategy_planner.run_async(ctx):
                    pass
                
                # Step 2: Compose email
                async for event in email_composer.run_async(ctx):
                    pass
                
                email_draft = ctx.session.state.get("email_draft")
                
                # Step 3: Human Approval Loop
                is_approved = False
                if self.state["require_human_approval"]:
                    while not is_approved:
                        yield Event(
                            author=self.name,
                            content=types.Content(
                                parts=[types.Part(text=
                                    f"📝 **Draft Email for Round {self.state['round']}**\n\n"
                                    f"---\n{email_draft}\n---\n\n"
                                    "⚠️ Please review: Approve or provide edits"
                                )]
                            ),
                            actions=EventActions(escalate=True)
                        )

                        user_edit_request = ctx.session.state.pop("user_edit_request", None)
                        is_approved = ctx.session.state.pop("email_approved", False)

                        if user_edit_request:
                            ctx.session.state["user_edit_request"] = user_edit_request
                            async for event in email_editor.run_async(ctx):
                                pass
                            email_draft = ctx.session.state.get("email_draft")

                        elif is_approved:
                            yield Event(
                                author=self.name,
                                content=types.Content(
                                    parts=[types.Part(text="✅ Email approved!")]
                                )
                            )
                            break

                        if not is_approved and not user_edit_request:
                            yield Event(
                                author=self.name,
                                content=types.Content(
                                    parts=[types.Part(text="⏳ Waiting for approval...")]
                                )
                            )
                            return
                
                # Step 4: Send email
                if self.gmail_service:
                    result = await self.gmail_service.send_email(
                        to=counterparty_email,
                        subject=f"Re: Contract Review - Round {self.state['round']}",
                        body=email_draft,
                        thread_id=self.state.get("email_thread_id")
                    )
                    
                    self.state["email_thread_id"] = result["thread_id"]
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            parts=[types.Part(text=
                                f"✅ Email sent! Thread: {result['thread_id']}"
                            )]
                        )
                    )
                else:
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            parts=[types.Part(text=
                                "[SIMULATION] Email would be sent here"
                            )]
                        )
                    )
                
                # Step 5: Wait for response
                response_email = ctx.session.state.get("latest_response_email")
                
                if not response_email:
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            parts=[types.Part(text="⏳ Awaiting response...")]
                        )
                    )
                    break
                
                # Step 6: Parse response
                async for event in response_parser.run_async(ctx):
                    pass
                
                parsed_response = ctx.session.state.get("parsed_response")
                
                # Step 7: Make decision
                async for event in decision_agent.run_async(ctx):
                    pass
                
                decision = ctx.session.state.get("negotiation_decision")
                decision_type = decision.get("decision") if isinstance(decision, dict) else decision.decision
                
                if decision_type == "ACCEPT":
                    confidence = decision.get("confidence") if isinstance(decision, dict) else decision.confidence
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            parts=[types.Part(text=
                                f"✅ SUCCESS! Round {self.state['round']}\n"
                                f"Confidence: {confidence}%"
                            )]
                        )
                    )
                    ctx.session.state["negotiation_outcome"] = "success"
                    return
                
                elif decision_type == "REJECT":
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            parts=[types.Part(text="❌ Negotiation unsuccessful")]
                        )
                    )
                    ctx.session.state["negotiation_outcome"] = "failed"
                    return
                
                elif decision_type == "COUNTER":
                    success_prob = parsed_response.get("success_probability") if isinstance(parsed_response, dict) else parsed_response.success_probability
                    self.state["success_probability"] = success_prob
                    
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            parts=[types.Part(text=
                                f"🔄 Preparing round {self.state['round'] + 1}"
                            )]
                        )
                    )
                    continue
            
            # Max rounds reached
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=
                        f"⚠️ Max rounds ({self.state['max_rounds']}) reached"
                    )]
                ),
                actions=EventActions(escalate=True)
            )
            
            ctx.session.state["negotiation_outcome"] = "timeout"
            
        except Exception as e:
            logger.error(f"Negotiation error: {e}", exc_info=True)
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"❌ Error: {str(e)}")]
                ),
                actions=EventActions(escalate=True)
            )
# 8. PIPELINE ASSEMBLY

analysis_pipeline = SequentialAgent(
    name="AnalysisPipeline",
    description="Analyzes contracts for risks and generates reports",
    sub_agents=[
        contract_parser,
        clause_extraction,
        risk_detection,
        evaluation_agent,
        report_generator,
    ]
)

# ============================================================================
# ROOT AGENT
# ============================================================================


# Initialize the negotiation pipeline.
# We instantiate it here so it can be passed to the root_agent's sub_agents list.
# The actual Gmail service with credentials will be configured later if needed.

class NegotiationRoundInitializer(BaseAgent):
    """Initializes the negotiation round number in the session state."""

    def __init__(self, name: str = "NegotiationRoundInitializer", **kwargs):
        super().__init__(name=name, description="Initializes the negotiation round.", **kwargs)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Sets the negotiation_round to 1 if it's not already set."""
        if "negotiation_round" not in ctx.session.state:
            ctx.session.state["negotiation_round"] = 1
        
        # Also initialize concessions_made, as strategy_planner needs it.
        if "concessions_made" not in ctx.session.state:
            ctx.session.state["concessions_made"] = []

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text="Negotiation round initialized.")]
            ),
        )

negotiation_round_initializer = NegotiationRoundInitializer()

negotiation_pipeline = SequentialAgent(
    name="NegotiationPipeline",
    description="Complete negotiation: Strategy → Draft → Review → Edit → Approve → Send",
    sub_agents=[
        negotiation_round_initializer,
        strategy_planner,         # creates strategy
        email_drafting_agent,     # drafts the email
        email_workflow_agent,     #  handles review/edit/approve/send
    ]
)


root_agent = LlmAgent(
    name="NegotifyOrchestrator",
    model=GEMINI_FLASH,
    description="Main coordinator for Negotify autonomous contract negotiation",
    instruction="""
You are Negotify, a secure AI-powered contract negotiation assistant.

🔒 CRITICAL SECURITY RULES:
❌ NEVER access other users' data
❌ NEVER share API keys or secrets
❌ NEVER fabricate data
❌ NEVER send emails without approval
❌ If data missing, say "Data not available"

Mission: Help freelancers negotiate better contract terms.

You remember:
    - Previous contracts analyzed in this session
    - User preferences and risk tolerance
    - Negotiation history and outcomes

📋 INTELLIGENT ROUTING LOGIC:

**CONTRACT ANALYSIS - Multiple Input Methods:**

1. **User provides contract TEXT in message:**
   - If user's message contains contract clauses, terms, or excerpts
   - If message includes phrases like "CONTRACT EXCERPT:", "Section X:", "The contract says"
   - → IMMEDIATELY analyze the provided text
   - → Create parsed_contract from the user's message
   - → Route to analysis_pipeline with the text content
   - → DO NOT ask for file upload when text is already provided

2. **User mentions contract but NO text provided:**
   - If user says "I have a contract" but doesn't include any text
   - If user asks "Can you analyze my contract?" without details
   - → THEN ask for either: upload file OR paste the contract text
   - → Say: "I can analyze your contract! You can either:
     1. Upload the contract file (PDF, DOCX, DOC, TXT)
     2. Paste the key clauses directly in your message
     Which would you prefer?"

3. **User uploads a file (gcs_path exists):**
   - Route to analysis_pipeline with the file path
   - Use PDF extraction tool to get text

    Use this context to provide personalized advice.

**BENCHMARKING:**
- Check if analysis completed (extracted_clauses exists)
- If YES: Route to benchmarking_pipeline
- If NO: Run analysis first

**NEGOTIATION:**
- Requires: risk_analysis AND counterparty_email
- Route to negotiation_pipeline (requires human approval)

**KEY CAPABILITY:**
You can analyze contracts from:
- Pasted text in messages ← Most common!
- Uploaded PDF/DOCX files
- Direct clause descriptions

Core analysis includes:
1. Risk Detection - Identify dangerous clauses
2. Severity Scoring - Rate risks 1-10
3. Recommendations - Suggest better terms
4. Benchmarking - Compare to industry standards

Always prioritize:
- User safety (flag high risks immediately)
- Privacy (data isolation)
- Factual accuracy (no hallucination)
- Transparency (explain reasoning)



You are not a lawyer. Advise consulting legal counsel for complex situations.
""",
    sub_agents=[
        analysis_pipeline,
        benchmarking_pipeline,
        negotiation_pipeline,
    ]
)
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def initialize_negotiation_pipeline(
    user_gmail_credentials: Optional[Credentials] = None
) -> NegotiationOrchestrator:
    """Initialize negotiation pipeline"""
    gmail_service = GmailIntegrationTool(user_gmail_credentials)
    return NegotiationOrchestrator(gmail_service=gmail_service)

async def analyze_contract(
    contract_id: str,
    gcs_path: str,
    user_id: str,
    counterparty_email: Optional[str] = None
) -> ContractAnalysisResult:
    """
    Main entry point for contract analysis
    """
    # Initialize negotiation pipeline
    user_credentials = None  # Replace with actual credentials

    # Initialize services
    session_service = DatabaseSessionService(connection_string=DATABASE_URL)
    artifact_service = GcsArtifactService(
        bucket_name=GCS_BUCKET_NAME,
        project_id=GOOGLE_CLOUD_PROJECT
    )
    
    runner = Runner(
        agent=root_agent,
        app_name="negotify",
        session_service=session_service,
        artifact_service=artifact_service
    )
    
    # Run analysis
    session_id = f"{user_id}_{contract_id}"
    
    result_events = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        input_message="Analyze this contract",
        state={
            "contract_id": contract_id,
            "gcs_path": gcs_path,
            "counterparty_email": counterparty_email
        }
    ):
        result_events.append(event)
        logger.info(f"Event: {event.author}")
    
    # Extract results
    session = await session_service.get_session(
        app_name="negotify",
        user_id=user_id,
        session_id=session_id
    )
    
    risk_analysis = session.state.get("risk_analysis", {})
    
    return ContractAnalysisResult(
        contract_id=contract_id,
        overall_risk_score=risk_analysis.get("overall_risk_score", 0),
        high_risks=risk_analysis.get("high_risks", []),
        medium_risks=risk_analysis.get("medium_risks", []),
        low_risks=risk_analysis.get("low_risks", []),
        executive_summary=session.state.get("analysis_report", ""),
        estimated_savings=None
    )

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    async def main():
        result = await analyze_contract(
            contract_id="contract_123",
            gcs_path="gs://negotify-contracts/uploads/user_abc/contract.pdf",
            user_id="user_abc",
            counterparty_email="client@company.com"
        )
        
        print(f" Negotify Analysis Complete!")
        print(f"Risk Score: {result.overall_risk_score}/10")
        print(f"High Risks: {len(result.high_risks)}")
    
    asyncio.run(main())

# ============================================================================
# END OF IMPLEMENTATION
# ============================================================================
