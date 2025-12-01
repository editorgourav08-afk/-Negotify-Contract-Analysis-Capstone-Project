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
from .tools import pdf_extraction_tool, GmailIntegrationTool, evaluate_contract_analysis

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
GEMINI_FLASH = "gemini-2.0-flash-exp"
GEMINI_PRO = "gemini-2.0-flash-thinking-exp"  
GEMINI_FLASH_LITE = "gemini-2.0-flash-exp"

# ============================================================================
# DATA MODELS (Pydantic Schemas)
# ============================================================================

class ContractClause(BaseModel):
    """Represents a contract clause"""
    clause_id: str
    clause_type: str  # liability, payment, IP, termination, etc.
    section_reference: str
    clause_text: str
    key_terms: Dict[str, Any]

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
    high_risks: List[RiskAssessment]
    medium_risks: List[RiskAssessment]
    low_risks: List[RiskAssessment]
    executive_summary: str
    estimated_savings: Optional[float] = None

class NegotiationStrategy(BaseModel):
    """Negotiation strategy plan"""
    priorities: List[str]
    concession_strategy: Dict[str, Any]
    deal_breakers: List[str]
    success_probability: float = Field(..., ge=0.0, le=1.0)

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
    key_terms: List[KeyTerm]

class ClauseExtractionResult(BaseModel):
    """The result of the clause extraction agent."""
    clauses: List[ExtractedClause]

class RiskAnalysisResult(BaseModel):
    """The result of the risk detection agent."""
    overall_risk_score: int = Field(..., ge=1, le=10)
    high_risks: List[RiskAssessment]
    medium_risks: List[RiskAssessment]
    low_risks: List[RiskAssessment]

class ParsedResponseResult(BaseModel):
    """The structured analysis of a counterparty's email response."""
    sentiment: str
    concessions_offered: List[str]  # Fixed: Changed from List[Any]
    firm_positions: List[str]  # Fixed: Changed from List[Any]
    counter_offers: List[str]  # Fixed: Changed from List[Any]
    questions_raised: List[str]  # Fixed: Changed from List[Any]
    relationship_status: str
    success_probability: float = Field(..., ge=0.0, le=1.0)
    recommended_action: str

class NegotiationDecisionResult(BaseModel):
    """The decision made by the negotiation decision agent."""
    decision: str = Field(..., pattern="^(ACCEPT|COUNTER|REJECT)$")  # Fixed: Added validation
    confidence: float = Field(..., ge=0.0, le=100.0)
    reasoning: str
    final_terms_summary: Optional[str] = None
    updated_priorities: Optional[List[str]] = None
    rejection_reasons: Optional[List[str]] = None


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
    output_schema=ClauseExtractionResult,
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
    output_schema=RiskAnalysisResult,
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

# 5. BENCHMARKING AGENTS (Sequential Execution)

similarity_search_agent = LlmAgent(
    name="SimilaritySearchAgent",
    model=GEMINI_FLASH,
    description="Finds similar contracts from database",
    instruction="""
From {extracted_clauses}, identify key contract characteristics:
- Industry
- Contract type
- Contract value range
- Client size
- Geographic region

🔒 SECURITY: Use ONLY characteristics from the actual contract data.

Return:
- List of similar contract IDs
- Similarity scores
- Key matching characteristics
""",
    output_key="similar_contracts"
)

industry_benchmark_agent = LlmAgent(
    name="IndustryBenchmarkAgent",
    model=GEMINI_FLASH,
    description="Compares contract terms against industry benchmarks",
    instruction="""
Query benchmarking data for {extracted_clauses}:

1. Payment Terms (25th, 50th, 75th percentile)
2. Liability Caps
3. IP Ownership
4. Termination Notice

🔒 SECURITY: Use ONLY data from benchmark database.
If unavailable, state "Benchmark data not available".

Flag terms in bottom 25% as below market.
""",
    output_key="benchmark_comparison"
)

clause_comparison_agent = LlmAgent(
    name="ClauseComparisonAgent",
    model=GEMINI_PRO,
    description="Compares problematic clauses against alternatives",
    instruction="""
For each high/medium risk clause in {risk_analysis}:

1. Find corresponding clauses in {similar_contracts}
2. Identify better language
3. Calculate improvement potential
4. Provide 2-3 alternatives

🔒 SECURITY: Use ONLY actual clauses from database.
NEVER fabricate alternative language.

Prioritize by:
1. Financial impact
2. Risk reduction
3. Negotiation likelihood
""",
    output_key="clause_alternatives"
)

#  SequentialAgent pipeline
benchmarking_pipeline = SequentialAgent(
    name="BenchmarkingPipeline",
    description="Compares contract against industry standards",
    sub_agents=[
        similarity_search_agent,
        industry_benchmark_agent,
        clause_comparison_agent
    ]
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
    output_schema=NegotiationStrategy,
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
    output_schema=ParsedResponseResult,
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
    output_schema=NegotiationDecisionResult,
    output_key="negotiation_decision"
)

# ----------------------------------------------------------------------------
# 7. CUSTOM NEGOTIATION ORCHESTRATOR (Fixed)
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
    _negotiation_state: Dict[str, Any] = {}
    
    def __init__(self, gmail_service: Optional[GmailIntegrationTool] = None, **kwargs):
        # Initialize parent first
        super().__init__(
            name="NegotiationOrchestrator",
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
    def state(self) -> Dict[str, Any]:
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
                        parts=[types.Part.from_text(
                            "❌ Error: Missing required negotiation data (risk_analysis or counterparty_email)"
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
                    parts=[types.Part.from_text(
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
                        parts=[types.Part.from_text(
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
                                parts=[types.Part.from_text(
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
                                    parts=[types.Part.from_text("✅ Email approved!")]
                                )
                            )
                            break

                        if not is_approved and not user_edit_request:
                            yield Event(
                                author=self.name,
                                content=types.Content(
                                    parts=[types.Part.from_text("⏳ Waiting for approval...")]
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
                            parts=[types.Part.from_text(
                                f"✅ Email sent! Thread: {result['thread_id']}"
                            )]
                        )
                    )
                else:
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            parts=[types.Part.from_text(
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
                            parts=[types.Part.from_text("⏳ Awaiting response...")]
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
                            parts=[types.Part.from_text(
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
                            parts=[types.Part.from_text("❌ Negotiation unsuccessful")]
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
                            parts=[types.Part.from_text(
                                f"🔄 Preparing round {self.state['round'] + 1}"
                            )]
                        )
                    )
                    continue
            
            # Max rounds reached
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part.from_text(
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
                    parts=[types.Part.from_text(f"❌ Error: {str(e)}")]
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
negotiation_pipeline = None 


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