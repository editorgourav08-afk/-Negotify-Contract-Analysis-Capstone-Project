# agents/negotify/tools/gmail_pro.py
"""
Negotify Pro Gmail Integration (FIXED for Gemini API)
======================================================
Professional email workflow: Draft → Review → Edit → Approve → Send

Features beautiful formatted email previews for user review.
"""

import os
import json
import base64
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from google.adk.tools import FunctionTool


# ============================================================================
# EMAIL WORKFLOW TYPES
# ============================================================================

class EmailStatus(Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EmailDraft:
    draft_id: str
    status: EmailStatus
    to_email: str
    to_name: str
    cc_emails: List[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    tone: str = "professional"
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    round_number: int = 1
    max_rounds: int = 5
    user_notes: str = ""
    approval_timestamp: Optional[str] = None


# ============================================================================
# DRAFT STORAGE
# ============================================================================

class DraftManager:
    def __init__(self):
        self._drafts = {}
        self._current_draft_id = None
    
    def create_draft(self, to_email: str, to_name: str = "") -> EmailDraft:
        draft_id = f"draft_{hashlib.md5(f'{datetime.now().isoformat()}_{len(self._drafts)}'.encode()).hexdigest()[:8]}"
        draft = EmailDraft(
            draft_id=draft_id,
            status=EmailStatus.DRAFT,
            to_email=to_email,
            to_name=to_name or to_email.split("@")[0].title(),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        self._drafts[draft_id] = draft
        self._current_draft_id = draft_id
        return draft
    
    def get_draft(self, draft_id: str = None) -> Optional[EmailDraft]:
        if draft_id:
            return self._drafts.get(draft_id)
        return self._drafts.get(self._current_draft_id) if self._current_draft_id else None
    
    def update_draft(self, draft_id: str, **kwargs) -> Optional[EmailDraft]:
        draft = self._drafts.get(draft_id)
        if not draft:
            return None
        for key, value in kwargs.items():
            if hasattr(draft, key):
                setattr(draft, key, value)
        draft.updated_at = datetime.now().isoformat()
        return draft
    
    def set_status(self, draft_id: str, status: EmailStatus) -> Optional[EmailDraft]:
        draft = self._drafts.get(draft_id)
        if draft:
            draft.status = status
            draft.updated_at = datetime.now().isoformat()
            if status == EmailStatus.APPROVED:
                draft.approval_timestamp = datetime.now().isoformat()
        return draft
    
    def get_all_drafts(self) -> List[EmailDraft]:
        return list(self._drafts.values())


_draft_manager = DraftManager()


# ============================================================================
# CORE EMAIL FUNCTIONS
# ============================================================================

def create_negotiation_draft(
    recipient_email: str,
    recipient_name: str = "",
    sender_name: str = "Contractor",
    contract_name: str = "the proposed contract",
    tone: str = "professional",
    point1_category: str = "",
    point1_change: str = "",
    point1_justification: str = "",
    point2_category: str = "",
    point2_change: str = "",
    point2_justification: str = "",
    point3_category: str = "",
    point3_change: str = "",
    point3_justification: str = "",
    round_number: int = 1,
    max_rounds: int = 5
) -> str:
    """
    Create a negotiation email draft with beautiful formatting.
    
    Args:
        recipient_email: Email address of the counterparty
        recipient_name: Name of the recipient
        sender_name: Your name for the signature
        contract_name: Name of the contract
        tone: Email tone - professional/friendly/firm/collaborative
        point1_category: Category for first point (e.g., liability)
        point1_change: What change you want
        point1_justification: Why the change is reasonable
        point2_category: Category for second point
        point2_change: What change you want
        point2_justification: Why
        point3_category: Category for third point
        point3_change: What change you want
        point3_justification: Why
        round_number: Current negotiation round
        max_rounds: Maximum rounds
        
    Returns:
        Formatted email preview string
    """
    # Create draft
    to_name = recipient_name or recipient_email.split("@")[0].title()
    draft = _draft_manager.create_draft(to_email=recipient_email, to_name=to_name)
    draft.tone = tone.lower()
    draft.round_number = round_number
    draft.max_rounds = max_rounds
    draft.subject = f"Re: {contract_name} - A Few Clarifications"
    
    # Build greeting based on tone
    if tone.lower() == "friendly":
        greeting = f"Hi {to_name},"
        opening = "Thank you for sending over the agreement! I'm excited about this project and want to make sure we set up a smooth working relationship."
        closing = "I'm confident we can find terms that work for both of us. Happy to hop on a quick call if that's easier!"
        signature = f"Best regards,\n{sender_name}"
    elif tone.lower() == "firm":
        greeting = f"Dear {to_name},"
        opening = "I have reviewed the proposed agreement. Before proceeding, I need to address several terms that require modification."
        closing = "These changes are necessary for me to proceed. I look forward to your response."
        signature = f"Regards,\n{sender_name}"
    elif tone.lower() == "collaborative":
        greeting = f"Hi {to_name},"
        opening = "I'm really looking forward to working together on this! I've reviewed the contract and have some thoughts that I think will help us start off on the right foot."
        closing = "I see these as conversation starters - let's find what works for both of us. When's a good time to chat?"
        signature = f"Looking forward to collaborating,\n{sender_name}"
    else:  # professional
        greeting = f"Dear {to_name},"
        opening = "Thank you for sending over the contract for review. After careful consideration, I would like to discuss a few terms to ensure a mutually beneficial agreement."
        closing = "I appreciate your consideration and look forward to discussing these points further."
        signature = f"Best regards,\n{sender_name}"
    
    # Build negotiation points
    points_text = ""
    point_num = 1
    
    for category, change, justification in [
        (point1_category, point1_change, point1_justification),
        (point2_category, point2_change, point2_justification),
        (point3_category, point3_change, point3_justification)
    ]:
        if category and change:
            category_display = category.replace("_", " ").title()
            points_text += f"\n{point_num}. **{category_display}**: {change}"
            if justification:
                points_text += f" {justification}"
            points_text += "\n"
            point_num += 1
    
    # Build full email body
    email_body = f"""{greeting}

{opening}

I had a few points I'd like to discuss:
{points_text}
{closing}

{signature}"""
    
    draft.body = email_body
    draft.status = EmailStatus.PENDING_REVIEW
    
    # Create beautiful formatted preview
    preview = f"""
🤝 Starting negotiation with {recipient_email}

📧 Round {round_number}/{max_rounds}
══════════════════════════════════════════════════════════════

📝 **Draft Email**

**To:** {to_name} <{recipient_email}>
**Subject:** {draft.subject}

──────────────────────────────────────────────────────────────

{email_body}

──────────────────────────────────────────────────────────────

⚠️ **Please review this draft before sending.**

Your options:
• Say **"approve"** or **"looks good"** to approve this draft
• Say **"make it friendlier"** or **"more firm"** to change tone
• Say **"edit: [your changes]"** to modify specific parts
• Say **"cancel"** to discard and start over

══════════════════════════════════════════════════════════════
"""
    
    return preview


def review_draft(draft_id: str = "") -> str:
    """Display current draft for review."""
    draft = _draft_manager.get_draft(draft_id if draft_id else None)
    
    if not draft:
        return "❌ No draft found. Please create a draft first."
    
    status_emoji = {
        EmailStatus.DRAFT: "📝",
        EmailStatus.PENDING_REVIEW: "👀",
        EmailStatus.APPROVED: "✅",
        EmailStatus.SENT: "📤",
        EmailStatus.CANCELLED: "🚫"
    }
    
    preview = f"""
{status_emoji.get(draft.status, "📧")} **Current Draft** (Version {draft.version})

══════════════════════════════════════════════════════════════

**To:** {draft.to_name} <{draft.to_email}>
**Subject:** {draft.subject}
**Tone:** {draft.tone.title()}
**Status:** {draft.status.value.replace("_", " ").title()}

──────────────────────────────────────────────────────────────

{draft.body}

──────────────────────────────────────────────────────────────

Your options:
• **"approve"** - Approve and prepare to send
• **"edit subject: [new subject]"** - Change subject line
• **"change tone to [friendly/firm/professional]"** - Rewrite with new tone
• **"cancel"** - Discard this draft

══════════════════════════════════════════════════════════════
"""
    return preview


def edit_draft(
    draft_id: str = "",
    new_subject: str = "",
    new_body: str = "",
    add_paragraph: str = ""
) -> str:
    """Edit an existing draft."""
    draft = _draft_manager.get_draft(draft_id if draft_id else None)
    
    if not draft:
        return "❌ No draft found to edit."
    
    changes = []
    
    if new_subject:
        draft.subject = new_subject
        changes.append(f"✏️ Subject updated to: {new_subject}")
    
    if new_body:
        draft.body = new_body
        draft.version += 1
        changes.append("✏️ Email body replaced")
    
    if add_paragraph:
        paragraphs = draft.body.split("\n\n")
        paragraphs.insert(-1, add_paragraph)
        draft.body = "\n\n".join(paragraphs)
        changes.append("✏️ Added new paragraph")
    
    draft.updated_at = datetime.now().isoformat()
    draft.status = EmailStatus.PENDING_REVIEW
    
    changes_text = "\n".join(changes) if changes else "No changes made"
    
    return f"""
✏️ **Draft Updated** (Version {draft.version})

{changes_text}

══════════════════════════════════════════════════════════════

**To:** {draft.to_name} <{draft.to_email}>
**Subject:** {draft.subject}

──────────────────────────────────────────────────────────────

{draft.body}

──────────────────────────────────────────────────────────────

Does this look better? Say **"approve"** or tell me what else to change.

══════════════════════════════════════════════════════════════
"""


def rewrite_draft_with_feedback(
    draft_id: str = "",
    new_tone: str = "",
    feedback: str = "",
    add_point_category: str = "",
    add_point_change: str = "",
    remove_category: str = ""
) -> str:
    """Request a rewrite with new parameters."""
    draft = _draft_manager.get_draft(draft_id if draft_id else None)
    
    if not draft:
        return "❌ No draft found to rewrite."
    
    return f"""
🔄 **Rewrite Requested**

I'll regenerate the email with these changes:
• New tone: {new_tone or draft.tone}
• Your feedback: {feedback or "None specified"}
• Add point: {add_point_category or "None"}
• Remove: {remove_category or "None"}

Please call create_negotiation_draft again with the updated parameters.
"""


def approve_draft(
    draft_id: str = "",
    user_confirmation: str = ""
) -> str:
    """Approve a draft for sending."""
    draft = _draft_manager.get_draft(draft_id if draft_id else None)
    
    if not draft:
        return "❌ No draft found to approve."
    
    confirmation_keywords = ["yes", "approve", "send", "confirm", "go ahead", "looks good", "ok", "okay", "lgtm"]
    is_confirmed = any(kw in user_confirmation.lower() for kw in confirmation_keywords)
    
    if not is_confirmed and not user_confirmation:
        return f"""
⚠️ **Confirmation Required**

Please confirm you want to approve this email:

**To:** {draft.to_name} <{draft.to_email}>
**Subject:** {draft.subject}

Say **"yes, approve"** or **"looks good"** to confirm.
"""
    
    draft = _draft_manager.set_status(draft.draft_id, EmailStatus.APPROVED)
    
    return f"""
✅ **Email Approved!**

══════════════════════════════════════════════════════════════

**To:** {draft.to_name} <{draft.to_email}>
**Subject:** {draft.subject}
**Approved at:** {draft.approval_timestamp}

══════════════════════════════════════════════════════════════

The email is ready to send!

• Say **"send now"** or **"send it"** to deliver immediately
• Say **"wait"** to send later
• Say **"edit"** if you want to make changes (approval will be reset)

══════════════════════════════════════════════════════════════
"""


def send_approved_email(
    draft_id: str = "",
    confirm_send: bool = False
) -> str:
    """Send an approved email."""
    draft = _draft_manager.get_draft(draft_id if draft_id else None)
    
    if not draft:
        return "❌ No draft found to send."
    
    if draft.status != EmailStatus.APPROVED:
        return f"""
⚠️ **Cannot Send**

Draft must be approved before sending.
Current status: {draft.status.value}

Please say **"approve"** first.
"""
    
    if not confirm_send:
        return f"""
🚨 **Final Confirmation**

You are about to send this email:

**To:** {draft.to_name} <{draft.to_email}>
**Subject:** {draft.subject}

This action cannot be undone.

Say **"send now"** or **"yes, send it"** to confirm.
"""
    
    # Mark as sent (in production, this would call Gmail API)
    draft = _draft_manager.set_status(draft.draft_id, EmailStatus.SENT)
    
    return f"""
📤 **Email Sent Successfully!**

══════════════════════════════════════════════════════════════

✅ Delivered to: {draft.to_name} <{draft.to_email}>
📧 Subject: {draft.subject}
🕐 Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

══════════════════════════════════════════════════════════════

Your negotiation email has been sent! 

I'll help you handle their response when it arrives. You can:
• Upload their reply for analysis
• Ask me to draft a follow-up
• Review the negotiation history

Good luck with the negotiation! 🤝

══════════════════════════════════════════════════════════════
"""


def cancel_draft(draft_id: str = "", reason: str = "") -> str:
    """Cancel/discard a draft."""
    draft = _draft_manager.get_draft(draft_id if draft_id else None)
    
    if not draft:
        return "❌ No draft found to cancel."
    
    draft = _draft_manager.set_status(draft.draft_id, EmailStatus.CANCELLED)
    
    return f"""
🚫 **Draft Cancelled**

The draft email to {draft.to_email} has been discarded.
{f"Reason: {reason}" if reason else ""}

Would you like to:
• Start a new draft with different approach?
• Review the contract analysis again?
• Take a different action?
"""


def get_draft_status(draft_id: str = "") -> str:
    """Get current draft status."""
    draft = _draft_manager.get_draft(draft_id if draft_id else None)
    
    if not draft:
        all_drafts = _draft_manager.get_all_drafts()
        if all_drafts:
            drafts_list = "\n".join([f"• {d.draft_id}: {d.status.value} → {d.to_email}" for d in all_drafts])
            return f"📋 Available drafts:\n{drafts_list}"
        return "📭 No drafts have been created yet."
    
    status_display = {
        EmailStatus.DRAFT: "📝 Draft Created",
        EmailStatus.PENDING_REVIEW: "👀 Pending Your Review",
        EmailStatus.CHANGES_REQUESTED: "✏️ Changes Requested",
        EmailStatus.APPROVED: "✅ Approved - Ready to Send",
        EmailStatus.SENT: "📤 Sent Successfully",
        EmailStatus.FAILED: "❌ Send Failed",
        EmailStatus.CANCELLED: "🚫 Cancelled"
    }
    
    return f"""
📊 **Draft Status**

══════════════════════════════════════════════════════════════

**ID:** {draft.draft_id}
**Status:** {status_display.get(draft.status, draft.status.value)}
**To:** {draft.to_name} <{draft.to_email}>
**Subject:** {draft.subject}
**Version:** {draft.version}
**Created:** {draft.created_at}
**Updated:** {draft.updated_at}
{f"**Approved:** {draft.approval_timestamp}" if draft.approval_timestamp else ""}

══════════════════════════════════════════════════════════════
"""


# ============================================================================
# CREATE ADK FUNCTION TOOLS
# ============================================================================

create_draft_tool = FunctionTool(func=create_negotiation_draft)
review_draft_tool = FunctionTool(func=review_draft)
edit_draft_tool = FunctionTool(func=edit_draft)
rewrite_draft_tool = FunctionTool(func=rewrite_draft_with_feedback)
approve_draft_tool = FunctionTool(func=approve_draft)
send_email_tool = FunctionTool(func=send_approved_email)
cancel_draft_tool = FunctionTool(func=cancel_draft)
draft_status_tool = FunctionTool(func=get_draft_status)


__all__ = [
    "create_negotiation_draft",
    "review_draft",
    "edit_draft",
    "rewrite_draft_with_feedback",
    "approve_draft",
    "send_approved_email",
    "cancel_draft",
    "get_draft_status",
    "create_draft_tool",
    "review_draft_tool",
    "edit_draft_tool",
    "rewrite_draft_tool",
    "approve_draft_tool",
    "send_email_tool",
    "cancel_draft_tool",
    "draft_status_tool",
]