# /negotify/benchmarks/vector_search.py
"""
Vector Search Module for Negotify - Vertex AI Version
======================================================

Production-grade semantic search using Google Vertex AI Vector Search.

Features:
- Sub-100ms query latency at scale
- Automatic scaling with demand
- Category and risk level filtering
- Full text retrieval via GCS metadata

Cost: ~$70-150/month for small-scale deployment
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result"""
    chunk_id: str
    text: str
    score: float
    category: str
    risk_level: str
    metadata: Dict[str, Any]


class NegotifyVectorSearch:
    """
    Vertex AI Vector Search for contract clause similarity.
    
    Usage:
        vs = NegotifyVectorSearch()
        results = vs.search("indemnify and hold harmless", category="liability")
    """
    
    def __init__(
        self,
        config_path: str = "benchmarks/config/vertex_ai_config.json",
        use_vertex_ai: bool = True
    ):
        """
        Initialize vector search.
        
        Args:
            config_path: Path to Vertex AI configuration file
            use_vertex_ai: If True, use Vertex AI. If False, fall back to ChromaDB.
        """
        self.use_vertex_ai = use_vertex_ai
        self.config_path = config_path
        self.config = None
        self.endpoint = None
        self.embedding_model = None
        self.metadata_cache = {}
        
        if use_vertex_ai:
            self._init_vertex_ai()
        else:
            self._init_chromadb()
    
    def _init_vertex_ai(self):
        """Initialize Vertex AI components"""
        try:
            # Load configuration
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"✅ Loaded config from {self.config_path}")
            else:
                # Try environment variables
                self.config = {
                    "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT", "negotify-project"),
                    "region": os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                    "endpoint_resource_name": os.environ.get("VERTEX_SEARCH_ENDPOINT"),
                    "deployed_index_id": os.environ.get("VERTEX_DEPLOYED_INDEX_ID", "negotify_clauses_v1"),
                    "bucket_name": os.environ.get("GCS_BUCKET_NAME"),
                    "embedding_model": "text-embedding-004"
                }
            
            # Initialize Vertex AI
            import vertexai
            from google.cloud import aiplatform
            from vertexai.language_models import TextEmbeddingModel
            
            vertexai.init(
                project=self.config["project_id"],
                location=self.config["region"]
            )
            aiplatform.init(
                project=self.config["project_id"],
                location=self.config["region"]
            )
            
            # Load embedding model
            self.embedding_model = TextEmbeddingModel.from_pretrained(
                self.config.get("embedding_model", "text-embedding-004")
            )
            logger.info("✅ Embedding model loaded")
            
            # Get endpoint (if configured)
            if self.config.get("endpoint_resource_name") and self.config["endpoint_resource_name"] != "PENDING_DEPLOYMENT":
                self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    self.config["endpoint_resource_name"]
                )
                logger.info(f"✅ Connected to endpoint: {self.config['endpoint_resource_name']}")
            else:
                logger.warning("⚠️ No endpoint configured - search will not work until deployed")
            
            # Load metadata cache from GCS
            self._load_metadata_cache()
            
            self.initialized = True
            logger.info("✅ Vertex AI Vector Search initialized")
            
        except ImportError as e:
            logger.error(f"❌ Missing Vertex AI dependencies: {e}")
            logger.info("   Run: pip install google-cloud-aiplatform vertexai")
            self.initialized = False
            self._init_chromadb_fallback()
            
        except Exception as e:
            logger.error(f"❌ Vertex AI initialization failed: {e}")
            self.initialized = False
            self._init_chromadb_fallback()
    
    def _init_chromadb(self):
        """Initialize ChromaDB as primary backend"""
        self._init_chromadb_backend()
    
    def _init_chromadb_fallback(self):
        """Fall back to ChromaDB if Vertex AI fails"""
        logger.info("📦 Falling back to ChromaDB...")
        self.use_vertex_ai = False
        self._init_chromadb_backend()
    
    def _init_chromadb_backend(self):
        """Initialize ChromaDB backend"""
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer
            
            persist_dir = "benchmarks/data/embeddings/chroma_db"
            os.makedirs(persist_dir, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="negotify_contracts",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.initialized = True
            logger.info("✅ ChromaDB initialized")
            
        except ImportError as e:
            logger.error(f"❌ ChromaDB not available: {e}")
            self.initialized = False
    
    def _load_metadata_cache(self):
        """Load clause metadata from GCS"""
        if not self.config or not self.config.get("bucket_name"):
            return
        
        try:
            from google.cloud import storage
            
            client = storage.Client(project=self.config["project_id"])
            bucket = client.bucket(self.config["bucket_name"])
            blob = bucket.blob("metadata/clause_metadata.json")
            
            if blob.exists():
                content = blob.download_as_text()
                self.metadata_cache = json.loads(content)
                logger.info(f"✅ Loaded metadata for {len(self.metadata_cache)} clauses")
            else:
                logger.warning("⚠️ No metadata file found in GCS")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load metadata cache: {e}")
    
    def _generate_embedding(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
        """Generate embedding for text"""
        if self.use_vertex_ai:
            from vertexai.language_models import TextEmbeddingInput
            
            inputs = [TextEmbeddingInput(text, task_type)]
            embeddings = self.embedding_model.get_embeddings(inputs)
            return embeddings[0].values
        else:
            return self.local_embedding_model.encode(text).tolist()
    
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        risk_level: Optional[str] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search for similar contract clauses.
        
        Args:
            query: The clause text to find similar matches for
            category: Filter by category (liability, payment_terms, ip_ownership, termination, non_compete)
            risk_level: Filter by risk level (low, medium, high)
            top_k: Number of results to return (1-50)
            
        Returns:
            List of SearchResult objects with similar clauses
        """
        if not self.initialized:
            logger.error("Vector search not initialized")
            return []
        
        if self.use_vertex_ai:
            return self._search_vertex_ai(query, category, risk_level, top_k)
        else:
            return self._search_chromadb(query, category, risk_level, top_k)
    
    def _search_vertex_ai(
        self,
        query: str,
        category: Optional[str],
        risk_level: Optional[str],
        top_k: int
    ) -> List[SearchResult]:
        """Search using Vertex AI Vector Search"""
        if not self.endpoint:
            logger.error("No endpoint configured for Vertex AI search")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query, "RETRIEVAL_QUERY")
            
            # Build filters
            from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
            
            filters = []
            if category:
                filters.append(Namespace(name="clause_category", allow_tokens=[category]))
            if risk_level:
                filters.append(Namespace(name="risk_level", allow_tokens=[risk_level]))
            
            # Execute search
            response = self.endpoint.find_neighbors(
                deployed_index_id=self.config.get("deployed_index_id", "negotify_clauses_v1"),
                queries=[query_embedding],
                num_neighbors=top_k,
                filter=filters if filters else None
            )
            
            # Convert to SearchResult objects
            results = []
            for neighbor in response[0]:
                # Get metadata from cache
                metadata = self.metadata_cache.get(neighbor.id, {})
                
                results.append(SearchResult(
                    chunk_id=neighbor.id,
                    text=metadata.get('text', ''),
                    score=float(neighbor.distance),
                    category=metadata.get('metadata', {}).get('clause_category', ''),
                    risk_level=metadata.get('metadata', {}).get('risk_level', ''),
                    metadata=metadata.get('metadata', {})
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Vertex AI search failed: {e}")
            return []
    
    def _search_chromadb(
        self,
        query: str,
        category: Optional[str],
        risk_level: Optional[str],
        top_k: int
    ) -> List[SearchResult]:
        """Search using ChromaDB"""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Build filter
            where_filter = None
            if category or risk_level:
                conditions = []
                if category:
                    conditions.append({"clause_category": category})
                if risk_level:
                    conditions.append({"risk_level": risk_level})
                
                if len(conditions) == 1:
                    where_filter = conditions[0]
                else:
                    where_filter = {"$and": conditions}
            
            # Search
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to SearchResult
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    score = 1 - distance
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    document = results['documents'][0][i] if results['documents'] else ""
                    
                    search_results.append(SearchResult(
                        chunk_id=chunk_id,
                        text=document,
                        score=score,
                        category=metadata.get("clause_category", ""),
                        risk_level=metadata.get("risk_level", ""),
                        metadata=metadata
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    def search_by_category(
        self,
        query: str,
        category: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Convenience method for ADK tool integration.
        Returns dict results instead of SearchResult objects.
        """
        results = self.search(query=query, category=category, top_k=top_k)
        
        return [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": round(r.score, 4),
                "category": r.category,
                "risk_level": r.risk_level,
                "metadata": r.metadata
            }
            for r in results
        ]
    
    def build_index_from_cuad(self, cuad_json_path: str) -> int:
        """
        Build vector index from CUAD data.
        For Vertex AI, use setup_vertex_ai_search.py instead.
        This method is for ChromaDB fallback.
        """
        if self.use_vertex_ai:
            logger.warning("For Vertex AI, use setup_vertex_ai_search.py instead")
            return 0
        
        # ChromaDB indexing
        with open(cuad_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        clauses = data.get('clauses', [])
        if not clauses:
            return 0
        
        # Prepare data
        ids = []
        texts = []
        metadatas = []
        
        for clause in clauses:
            ids.append(clause['chunk_id'])
            texts.append(clause['text'])
            metadatas.append({
                'clause_category': clause.get('clause_category', ''),
                'risk_level': clause.get('risk_level', ''),
                'cuad_category': clause.get('cuad_category', '')
            })
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} clauses...")
        embeddings = self.local_embedding_model.encode(texts, show_progress_bar=True).tolist()
        
        # Add to collection
        self.chroma_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        
        logger.info(f"✅ Indexed {len(ids)} clauses")
        return len(ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        if self.use_vertex_ai:
            return {
                "backend": "Vertex AI Vector Search",
                "project": self.config.get("project_id") if self.config else None,
                "region": self.config.get("region") if self.config else None,
                "endpoint": self.config.get("endpoint_resource_name") if self.config else None,
                "embedding_model": "text-embedding-004",
                "metadata_count": len(self.metadata_cache),
                "initialized": self.initialized
            }
        else:
            count = self.chroma_collection.count() if hasattr(self, 'chroma_collection') else 0
            return {
                "backend": "ChromaDB",
                "document_count": count,
                "embedding_model": "all-MiniLM-L6-v2",
                "initialized": self.initialized
            }


# Singleton instance
_instance: Optional[NegotifyVectorSearch] = None


def get_vector_search(use_vertex_ai: bool = True, force_reinit: bool = False) -> NegotifyVectorSearch:
    """
    Get or create vector search instance.
    
    Args:
        use_vertex_ai: Use Vertex AI (True) or ChromaDB (False)
        force_reinit: Force reinitialization
    """
    global _instance
    
    if _instance is None or force_reinit:
        _instance = NegotifyVectorSearch(use_vertex_ai=use_vertex_ai)
    
    return _instance


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Vector Search\n" + "="*50)
    
    # Test with Vertex AI
    print("\n1️⃣ Testing Vertex AI backend...")
    try:
        vs = NegotifyVectorSearch(use_vertex_ai=True)
        stats = vs.get_stats()
        print(f"   Backend: {stats['backend']}")
        print(f"   Initialized: {stats['initialized']}")
        
        if stats['initialized'] and vs.endpoint:
            results = vs.search("indemnify the client", category="liability", top_k=3)
            print(f"   Test search returned {len(results)} results")
    except Exception as e:
        print(f"   ❌ Vertex AI test failed: {e}")
    
    # Test with ChromaDB
    print("\n2️⃣ Testing ChromaDB backend...")
    try:
        vs = NegotifyVectorSearch(use_vertex_ai=False)
        stats = vs.get_stats()
        print(f"   Backend: {stats['backend']}")
        print(f"   Documents: {stats.get('document_count', 0)}")
        
        if stats.get('document_count', 0) > 0:
            results = vs.search("payment within 30 days", category="payment_terms", top_k=3)
            print(f"   Test search returned {len(results)} results")
    except Exception as e:
        print(f"   ❌ ChromaDB test failed: {e}")
    
    print("\n✅ Testing complete!")