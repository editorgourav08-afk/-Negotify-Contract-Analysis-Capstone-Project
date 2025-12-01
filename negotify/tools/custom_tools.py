import base64
from datetime import datetime
from email.mime.text import MIMEText
from io import BytesIO
import logging
import json
from typing import Any, Dict, List, Optional

import PyPDF2
from google.api_core.client_options import ClientOptions
from google.auth.credentials import Credentials
from google.cloud import storage
from googleapiclient.discovery import build

from google.adk.tools import FunctionTool
from ..evaluation import NegotifyEvaluator

logger = logging.getLogger(__name__)


class PDFExtractionTool:
    """Tool to extract text from PDF contracts"""

    @staticmethod
    async def extract_pdf_text(contract_id: str, gcs_path: str) -> Dict[str, Any]:
        """Extracts text and structure from a PDF contract document.

        Args:
            contract_id: Unique contract identifier
            gcs_path: GCS path to PDF file

        Returns:
            A dictionary containing the extracted text, page count, and identified sections.
        """
        try:
            # Download from GCS
            storage_client = storage.Client()
            bucket_name = gcs_path.split("/")[2]
            blob_path = "/".join(gcs_path.split("/")[3:])

            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            pdf_bytes = blob.download_as_bytes()

            # Extract text (simplified - in production use Document AI)
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Structure the output
            sections = PDFExtractionTool._identify_sections(text)

            return {
                "contract_id": contract_id,
                "raw_text": text,
                "page_count": len(pdf_reader.pages),
                "sections": sections,
                "extraction_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise

    @staticmethod
    def _identify_sections(text: str) -> List[Dict]:
        """Identify contract sections from text"""
        # Simplified section detection
        # In production, use more sophisticated parsing
        sections = []
        common_headers = [
            "Scope of Work",
            "Payment Terms",
            "Intellectual Property",
            "Liability",
            "Indemnification",
            "Termination",
            "Confidentiality",
        ]

        for header in common_headers:
            if header.lower() in text.lower():
                # Find section text (simplified)
                start = text.lower().find(header.lower())
                end = start + 500  # Get next 500 chars
                sections.append({"title": header, "content": text[start:end]})

        return sections


# Create FunctionTool wrapper
pdf_extraction_tool = FunctionTool(func=PDFExtractionTool.extract_pdf_text)


class GmailIntegrationTool:
    """Tool to send and receive emails via Gmail API"""

    def __init__(self, user_credentials: Credentials):
        self.credentials = user_credentials
        self.service = build("gmail", "v1", credentials=self.credentials)

    async def send_email(
        self, to: str, subject: str, body: str, thread_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Send email via Gmail API"""
        try:
            message = MIMEText(body, "html")
            message["to"] = to
            message["subject"] = subject

            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

            send_payload = {"raw": raw_message}
            if thread_id:
                send_payload["threadId"] = thread_id

            result = (
                self.service.users().messages().send(userId="me", body=send_payload).execute()
            )

            logger.info(f"Email sent: {result['id']}")

            return {
                "message_id": result["id"],
                "thread_id": result.get("threadId"),
                "status": "sent",
            }

        except Exception as e:
            logger.error(f"Gmail send error: {e}")
            raise

    async def get_thread_messages(self, thread_id: str) -> List[Dict]:
        """Retrieve all messages in a thread"""
        try:
            thread = (
                self.service.users().threads().get(userId="me", id=thread_id, format="full").execute()
            )

            messages = []
            for msg in thread["messages"]:
                headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}

                # Get body
                if "parts" in msg["payload"]:
                    body_data = msg["payload"]["parts"][0]["body"]["data"]
                else:
                    body_data = msg["payload"]["body"]["data"]

                body_text = base64.urlsafe_b64decode(body_data).decode("utf-8")

                messages.append(
                    {
                        "id": msg["id"],
                        "from": headers.get("From"),
                        "subject": headers.get("Subject"),
                        "date": headers.get("Date"),
                        "body": body_text,
                    }
                )

            return messages

        except Exception as e:
            logger.error(f"Gmail fetch error: {e}")
            raise


def evaluate_contract_analysis(analysis_json: str, contract_text: str = "") -> str:
    """
    Evaluate the quality of a contract analysis.

    Args:
        analysis_json: JSON string of the analysis result.
        contract_text: The original contract text (optional).

    Returns:
        Evaluation metrics as a JSON string.
    """
    evaluator = NegotifyEvaluator()
    result = json.loads(analysis_json)
    # The contract_text is not strictly necessary for the current evaluation logic
    # but is part of the original function signature.
    evaluation = evaluator.evaluate_analysis(contract_text, result)
    return json.dumps(evaluation, indent=2)