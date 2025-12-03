# agents/negotify/benchmarks/cuad_processor.py
"""
CUAD Dataset Processor for Negotify

Downloads and processes the Contract Understanding Atticus Dataset (CUAD)
to extract real contract clauses for benchmarking.

CUAD contains:
- 510 commercial contracts from SEC EDGAR
- 13,101 expert-annotated clause spans
- 41 clause categories

This processor extracts clauses relevant to Negotify's 5 key categories:
1. Payment Terms
2. Liability Caps
3. IP Ownership
4. Termination
5. Non-Compete
"""

import json
import os
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import re


# Category mappings from CUAD question types to Negotify categories
CUAD_TO_NEGOTIFY_MAPPING = {
    # Payment Terms
    "payment_terms": [
        "Minimum Commitment",
        "Revenue/Profit Sharing",
        "Liquidated Damages",
        "Price Restrictions",
        "Most Favored Nation",
    ],
    
    # Liability
    "liability": [
        "Cap On Liability",
        "Limitation Of Liability",
        "Indemnification",
        "Insurance",
        "Uncapped Liability",
    ],
    
    # IP Ownership
    "ip_ownership": [
        "IP Ownership Assignment",
        "Joint IP Ownership", 
        "License Grant",
        "Non-Transferable License",
        "Affiliate License-Licensor",
        "Affiliate License-Licensee",
        "Unlimited/All-You-Can-Eat-License",
        "Irrevocable Or Perpetual License",
        "Source Code Escrow",
    ],
    
    # Termination
    "termination": [
        "Termination For Convenience",
        "Expiration Date",
        "Renewal Term",
        "Post-Termination Services",
        "Change Of Control",
        "Anti-Assignment",
    ],
    
    # Non-Compete
    "non_compete": [
        "Non-Compete",
        "Exclusivity",
        "No-Solicit Of Customers",
        "No-Solicit Of Employees",
        "Competitive Restriction Exception",
        "Non-Disparagement",
    ],
    
    # Additional useful categories
    "confidentiality": [
        "Confidentiality",
        "Third Party Beneficiary",
    ],
    
    "warranty": [
        "Warranty Duration",
    ],
    
    "dispute": [
        "Governing Law",
        "Dispute Resolution",
        "Audit Rights",
    ],
}

# Reverse mapping for quick lookup
QUESTION_TO_CATEGORY = {}
for category, questions in CUAD_TO_NEGOTIFY_MAPPING.items():
    for q in questions:
        QUESTION_TO_CATEGORY[q.lower()] = category


@dataclass
class ProcessedClause:
    """A processed contract clause ready for vector indexing"""
    chunk_id: str
    text: str
    clause_category: str
    cuad_category: str
    contract_id: str
    source_dataset: str = "CUAD"
    risk_level: str = "medium"
    char_count: int = 0
    word_count: int = 0
    extraction_date: str = ""
    
    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())
        if not self.extraction_date:
            self.extraction_date = datetime.now().isoformat()


class CUADProcessor:
    """
    Processes the CUAD dataset to extract contract clauses
    for Negotify's benchmark database.
    """
    
    def __init__(self, cache_dir: str = "benchmarks/data"):
        self.cache_dir = cache_dir
        self.processed_clauses: List[ProcessedClause] = []
        self.stats = {
            "total_samples": 0,
            "clauses_extracted": 0,
            "by_category": {},
            "contracts_processed": set()
        }
    
    def download_cuad(self) -> Any:
        """Download CUAD dataset from Hugging Face"""
        try:
            from datasets import load_dataset
            
            print("📥 Downloading CUAD dataset from Hugging Face...")
            print("   This may take a few minutes on first run...")
            
            # Load the CUAD dataset
            dataset = load_dataset("cuad")
            
            train_size = len(dataset['train'])
            test_size = len(dataset['test'])
            
            print(f"✅ Loaded CUAD dataset:")
            print(f"   - Train samples: {train_size:,}")
            print(f"   - Test samples: {test_size:,}")
            print(f"   - Total: {train_size + test_size:,}")
            
            return dataset
            
        except ImportError:
            print("❌ Error: 'datasets' library not installed")
            print("   Run: pip install datasets")
            raise
        except Exception as e:
            print(f"❌ Error downloading CUAD: {e}")
            raise
    
    def _extract_category_from_question(self, question: str) -> Optional[str]:
        """Map CUAD question to Negotify category"""
        question_lower = question.lower()
        
        # Direct mapping
        for cuad_q, category in QUESTION_TO_CATEGORY.items():
            if cuad_q in question_lower:
                return category
        
        # Keyword fallback
        keyword_mapping = {
            "liability": "liability",
            "indemnif": "liability",
            "cap on": "liability",
            "payment": "payment_terms",
            "revenue": "payment_terms",
            "price": "payment_terms",
            "ip": "ip_ownership",
            "intellectual": "ip_ownership",
            "license": "ip_ownership",
            "ownership": "ip_ownership",
            "terminat": "termination",
            "expir": "termination",
            "renewal": "termination",
            "non-compete": "non_compete",
            "exclusiv": "non_compete",
            "solicit": "non_compete",
            "confidential": "confidentiality",
            "govern": "dispute",
            "arbitrat": "dispute",
        }
        
        for keyword, category in keyword_mapping.items():
            if keyword in question_lower:
                return category
        
        return None
    
    def _assess_risk_level(self, text: str, category: str) -> str:
        """Assess risk level based on clause content"""
        text_lower = text.lower()
        
        # High risk indicators
        high_risk_patterns = [
            r"unlimited",
            r"without limit",
            r"uncapped",
            r"sole discretion",
            r"unilateral",
            r"perpetual.*exclusive",
            r"worldwide.*exclusive",
            r"indemnify.*all claims",
            r"waive.*all rights",
        ]
        
        for pattern in high_risk_patterns:
            if re.search(pattern, text_lower):
                return "high"
        
        # Medium risk indicators (category-specific)
        if category == "liability":
            if any(x in text_lower for x in ["cap", "limit", "not exceed"]):
                return "medium"
        
        if category == "termination":
            if "without cause" in text_lower or "at any time" in text_lower:
                return "medium"
        
        if category == "non_compete":
            if any(x in text_lower for x in ["year", "month", "geographic"]):
                return "medium"
        
        return "low"
    
    def _generate_chunk_id(self, text: str, contract_id: str, category: str) -> str:
        """Generate unique chunk ID"""
        content = f"{contract_id}_{category}_{text[:100]}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"cuad_{hash_val}"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize clause text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and artifacts
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'Page \d+ of \d+', '', text)
        # Trim
        text = text.strip()
        return text
    
    def process_dataset(self, dataset: Any) -> List[ProcessedClause]:
        """Process the CUAD dataset and extract clauses"""
        print("\n🔍 Processing CUAD dataset...")
        
        # Combine train and test
        all_samples = list(dataset['train']) + list(dataset['test'])
        self.stats["total_samples"] = len(all_samples)
        
        print(f"   Processing {len(all_samples):,} QA pairs...")
        
        seen_texts = set()  # Deduplication
        
        for sample in tqdm(all_samples, desc="   Extracting clauses"):
            question = sample.get('question', '')
            answers = sample.get('answers', {})
            context = sample.get('context', '')
            
            # Get category
            category = self._extract_category_from_question(question)
            if not category:
                continue
            
            # Extract answer text(s)
            answer_texts = answers.get('text', [])
            if not answer_texts:
                continue
            
            # Generate contract ID from context hash
            contract_id = hashlib.md5(context[:500].encode()).hexdigest()[:8]
            self.stats["contracts_processed"].add(contract_id)
            
            for answer_text in answer_texts:
                # Clean text
                clean_text = self._clean_text(answer_text)
                
                # Skip empty, too short, or duplicate
                if len(clean_text) < 50:
                    continue
                if clean_text in seen_texts:
                    continue
                seen_texts.add(clean_text)
                
                # Create processed clause
                chunk_id = self._generate_chunk_id(clean_text, contract_id, category)
                risk_level = self._assess_risk_level(clean_text, category)
                
                clause = ProcessedClause(
                    chunk_id=chunk_id,
                    text=clean_text,
                    clause_category=category,
                    cuad_category=question,
                    contract_id=contract_id,
                    risk_level=risk_level,
                )
                
                self.processed_clauses.append(clause)
                
                # Update stats
                if category not in self.stats["by_category"]:
                    self.stats["by_category"][category] = 0
                self.stats["by_category"][category] += 1
        
        self.stats["clauses_extracted"] = len(self.processed_clauses)
        
        return self.processed_clauses
    
    def print_stats(self):
        """Print processing statistics"""
        print(f"\n📊 Processing Statistics:")
        print(f"   Total QA pairs processed: {self.stats['total_samples']:,}")
        print(f"   Unique contracts: {len(self.stats['contracts_processed']):,}")
        print(f"   Clauses extracted: {self.stats['clauses_extracted']:,}")
        print(f"\n   By category:")
        for category, count in sorted(self.stats["by_category"].items(), key=lambda x: -x[1]):
            print(f"     - {category}: {count:,}")
    
    def save_to_json(self, output_path: str):
        """Save processed clauses to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            "version": "1.0",
            "source": "CUAD (Contract Understanding Atticus Dataset)",
            "processed_date": datetime.now().isoformat(),
            "stats": {
                "total_clauses": len(self.processed_clauses),
                "unique_contracts": len(self.stats["contracts_processed"]),
                "by_category": self.stats["by_category"]
            },
            "clauses": [asdict(c) for c in self.processed_clauses]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Saved {len(self.processed_clauses):,} clauses to {output_path}")
    
    @staticmethod
    def load_from_json(path: str) -> List[Dict[str, Any]]:
        """Load processed clauses from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get("clauses", [])


class CUADSampleGenerator:
    """
    Generates sample/synthetic CUAD-like data for testing
    when the full dataset download is not practical.
    """
    
    @staticmethod
    def generate_sample_clauses() -> List[Dict[str, Any]]:
        """Generate sample clauses for each category"""
        samples = [
            # Liability clauses
            {
                "chunk_id": "sample_liab_001",
                "text": "The Contractor's total liability under this Agreement shall not exceed the total fees paid to Contractor during the twelve (12) month period immediately preceding the claim.",
                "clause_category": "liability",
                "cuad_category": "Cap On Liability",
                "contract_id": "sample_001",
                "risk_level": "low",
            },
            {
                "chunk_id": "sample_liab_002",
                "text": "Contractor shall indemnify, defend, and hold harmless Client from and against any and all claims, damages, losses, costs, and expenses, including attorney's fees, arising from or relating to Contractor's performance of the Services, without any limitation or cap.",
                "clause_category": "liability",
                "cuad_category": "Indemnification",
                "contract_id": "sample_002",
                "risk_level": "high",
            },
            {
                "chunk_id": "sample_liab_003",
                "text": "Neither party shall be liable to the other for any indirect, incidental, consequential, special, or exemplary damages arising out of this Agreement, even if such party has been advised of the possibility of such damages.",
                "clause_category": "liability",
                "cuad_category": "Limitation Of Liability",
                "contract_id": "sample_003",
                "risk_level": "low",
            },
            
            # Payment clauses
            {
                "chunk_id": "sample_pay_001",
                "text": "Client shall pay all undisputed invoices within thirty (30) days of receipt. Late payments shall accrue interest at the rate of 1.5% per month.",
                "clause_category": "payment_terms",
                "cuad_category": "Minimum Commitment",
                "contract_id": "sample_004",
                "risk_level": "low",
            },
            {
                "chunk_id": "sample_pay_002",
                "text": "Payment shall be due upon Client's acceptance of the deliverables, such acceptance not to be unreasonably withheld. Client shall have sixty (60) days to review and accept deliverables.",
                "clause_category": "payment_terms",
                "cuad_category": "Minimum Commitment",
                "contract_id": "sample_005",
                "risk_level": "medium",
            },
            {
                "chunk_id": "sample_pay_003",
                "text": "All fees are contingent upon Client's satisfaction with the Services. Client shall have sole discretion to determine whether the Services meet Client's requirements.",
                "clause_category": "payment_terms",
                "cuad_category": "Minimum Commitment",
                "contract_id": "sample_006",
                "risk_level": "high",
            },
            
            # IP ownership clauses
            {
                "chunk_id": "sample_ip_001",
                "text": "Upon full payment, Contractor hereby assigns to Client all right, title, and interest in and to the Deliverables, including all intellectual property rights therein. Contractor retains ownership of pre-existing materials and general skills.",
                "clause_category": "ip_ownership",
                "cuad_category": "IP Ownership Assignment",
                "contract_id": "sample_007",
                "risk_level": "low",
            },
            {
                "chunk_id": "sample_ip_002",
                "text": "All work product, inventions, discoveries, and intellectual property created by Contractor during the term of this Agreement, whether or not related to the Services, shall be the sole and exclusive property of Client.",
                "clause_category": "ip_ownership",
                "cuad_category": "IP Ownership Assignment",
                "contract_id": "sample_008",
                "risk_level": "high",
            },
            {
                "chunk_id": "sample_ip_003",
                "text": "Client is granted a non-exclusive, perpetual, royalty-free license to use the Deliverables for Client's internal business purposes. Contractor retains all ownership rights in the Deliverables.",
                "clause_category": "ip_ownership",
                "cuad_category": "License Grant",
                "contract_id": "sample_009",
                "risk_level": "low",
            },
            
            # Termination clauses
            {
                "chunk_id": "sample_term_001",
                "text": "Either party may terminate this Agreement for convenience upon thirty (30) days prior written notice. Upon termination, Client shall pay for all Services rendered through the effective date of termination.",
                "clause_category": "termination",
                "cuad_category": "Termination For Convenience",
                "contract_id": "sample_010",
                "risk_level": "low",
            },
            {
                "chunk_id": "sample_term_002",
                "text": "Client may terminate this Agreement at any time, for any reason or no reason, effective immediately upon notice to Contractor. Upon termination, Contractor shall not be entitled to any further compensation.",
                "clause_category": "termination",
                "cuad_category": "Termination For Convenience",
                "contract_id": "sample_011",
                "risk_level": "high",
            },
            {
                "chunk_id": "sample_term_003",
                "text": "This Agreement shall automatically renew for successive one (1) year periods unless either party provides written notice of non-renewal at least sixty (60) days prior to the end of the then-current term.",
                "clause_category": "termination",
                "cuad_category": "Renewal Term",
                "contract_id": "sample_012",
                "risk_level": "medium",
            },
            
            # Non-compete clauses
            {
                "chunk_id": "sample_nc_001",
                "text": "During the term of this Agreement and for a period of six (6) months thereafter, Contractor shall not provide similar services to direct competitors of Client in the same metropolitan area.",
                "clause_category": "non_compete",
                "cuad_category": "Non-Compete",
                "contract_id": "sample_013",
                "risk_level": "medium",
            },
            {
                "chunk_id": "sample_nc_002",
                "text": "Contractor agrees not to engage in any business that competes with Client anywhere in the world for a period of two (2) years following termination of this Agreement.",
                "clause_category": "non_compete",
                "cuad_category": "Non-Compete",
                "contract_id": "sample_014",
                "risk_level": "high",
            },
            {
                "chunk_id": "sample_nc_003",
                "text": "During the term of this Agreement, Contractor agrees to work exclusively for Client and shall not provide services to any third party without Client's prior written consent.",
                "clause_category": "non_compete",
                "cuad_category": "Exclusivity",
                "contract_id": "sample_015",
                "risk_level": "medium",
            },
        ]
        
        # Add metadata
        for sample in samples:
            sample["source_dataset"] = "CUAD_sample"
            sample["char_count"] = len(sample["text"])
            sample["word_count"] = len(sample["text"].split())
            sample["extraction_date"] = datetime.now().isoformat()
        
        return samples


def main():
    """Main function to process CUAD dataset"""
    processor = CUADProcessor()
    
    try:
        # Try to download real CUAD data
        dataset = processor.download_cuad()
        processor.process_dataset(dataset)
        
    except Exception as e:
        print(f"\n⚠️ Could not download CUAD: {e}")
        print("   Generating sample data instead...")
        
        # Use sample data as fallback
        samples = CUADSampleGenerator.generate_sample_clauses()
        processor.processed_clauses = [
            ProcessedClause(**s) for s in samples
        ]
        processor.stats["clauses_extracted"] = len(samples)
        for s in samples:
            cat = s["clause_category"]
            if cat not in processor.stats["by_category"]:
                processor.stats["by_category"][cat] = 0
            processor.stats["by_category"][cat] += 1
    
    # Print stats and save
    processor.print_stats()
    processor.save_to_json("benchmarks/data/cuad_processed.json")
    
    print("\n✅ CUAD processing complete!")


if __name__ == "__main__":
    main()