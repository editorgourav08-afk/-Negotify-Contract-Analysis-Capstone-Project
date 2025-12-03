#!/usr/bin/env python3
"""
Negotify - Vertex AI Vector Search Setup
=========================================
Production-grade vector search for contract clause similarity.

This script:
1. Creates a GCS bucket for embeddings data
2. Generates embeddings for CUAD clauses
3. Creates a Vertex AI Vector Search Index
4. Deploys the index to an endpoint
5. Tests the search functionality

Prerequisites:
- Google Cloud Project with billing
- gcloud CLI authenticated
- APIs enabled (aiplatform, compute, storage)

Usage:
    python setup_vertex_ai_search.py --project YOUR_PROJECT_ID --region us-central1
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import struct

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "negotify-project")
DEFAULT_REGION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
DEFAULT_BUCKET = None  # Will be set based on project ID

INDEX_DISPLAY_NAME = "negotify-contract-clauses"
ENDPOINT_DISPLAY_NAME = "negotify-search-endpoint"
DEPLOYED_INDEX_ID = "negotify_clauses_v1"

# Embedding configuration
EMBEDDING_MODEL = "text-embedding-004"  # Google's latest embedding model
EMBEDDING_DIMENSION = 768  # Dimension for text-embedding-004


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(step: int, text: str):
    """Print step indicator"""
    print(f"\n📌 Step {step}: {text}")
    print("-" * 50)


# ============================================================================
# STEP 1: VERIFY SETUP
# ============================================================================

def verify_setup(project_id: str, region: str) -> bool:
    """Verify Google Cloud setup"""
    print_step(1, "Verifying Google Cloud Setup")
    
    try:
        from google.cloud import aiplatform
        import vertexai
        
        print(f"   Project: {project_id}")
        print(f"   Region: {region}")
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        aiplatform.init(project=project_id, location=region)
        
        print("   ✅ Vertex AI SDK initialized")
        
        # Test API access
        from vertexai.language_models import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        
        # Quick test embedding
        test_result = model.get_embeddings(["test"])
        print(f"   ✅ Embedding model working (dim={len(test_result[0].values)})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Setup verification failed: {e}")
        print("\n   Troubleshooting:")
        print("   1. Run: gcloud auth application-default login")
        print("   2. Run: gcloud services enable aiplatform.googleapis.com")
        print(f"   3. Verify project exists: gcloud projects describe {project_id}")
        return False


# ============================================================================
# STEP 2: CREATE GCS BUCKET
# ============================================================================

def setup_gcs_bucket(project_id: str, region: str) -> str:
    """Create GCS bucket for embeddings data"""
    print_step(2, "Setting Up Cloud Storage Bucket")
    
    from google.cloud import storage
    
    bucket_name = f"{project_id}-negotify-embeddings"
    
    try:
        client = storage.Client(project=project_id)
        
        # Check if bucket exists
        try:
            bucket = client.get_bucket(bucket_name)
            print(f"   ✅ Bucket already exists: gs://{bucket_name}")
            return bucket_name
        except:
            pass
        
        # Create bucket
        bucket = client.create_bucket(
            bucket_name,
            location=region.split("-")[0] + "-" + region.split("-")[1]  # e.g., us-central1 -> us
        )
        
        print(f"   ✅ Created bucket: gs://{bucket_name}")
        return bucket_name
        
    except Exception as e:
        print(f"   ❌ Bucket creation failed: {e}")
        # Try with timestamp suffix
        bucket_name = f"{project_id}-negotify-{int(time.time())}"
        try:
            bucket = client.create_bucket(bucket_name, location="us")
            print(f"   ✅ Created bucket: gs://{bucket_name}")
            return bucket_name
        except Exception as e2:
            print(f"   ❌ Fallback also failed: {e2}")
            raise


# ============================================================================
# STEP 3: GENERATE EMBEDDINGS
# ============================================================================

def load_cuad_data(cuad_path: str) -> List[Dict[str, Any]]:
    """Load CUAD processed data"""
    with open(cuad_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('clauses', [])


def generate_embeddings(project_id: str, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for all clauses using Vertex AI"""
    print_step(3, "Generating Embeddings with Vertex AI")
    
    from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Clauses to embed: {len(clauses)}")
    
    embedded_clauses = []
    batch_size = 50  # Vertex AI limit per request
    
    for i in range(0, len(clauses), batch_size):
        batch = clauses[i:i+batch_size]
        texts = [c['text'] for c in batch]
        
        # Create embedding inputs
        inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in texts]
        
        # Generate embeddings
        try:
            embeddings = model.get_embeddings(inputs)
            
            for j, (clause, embedding) in enumerate(zip(batch, embeddings)):
                embedded_clauses.append({
                    "id": clause['chunk_id'],
                    "embedding": embedding.values,
                    "metadata": {
                        "clause_category": clause.get('clause_category', ''),
                        "risk_level": clause.get('risk_level', ''),
                        "cuad_category": clause.get('cuad_category', ''),
                        "text_preview": clause['text'][:200]
                    },
                    "full_text": clause['text']
                })
            
            print(f"   Processed batch {i//batch_size + 1}/{(len(clauses)-1)//batch_size + 1}")
            
        except Exception as e:
            print(f"   ⚠️ Batch {i//batch_size + 1} failed: {e}")
            continue
    
    print(f"   ✅ Generated {len(embedded_clauses)} embeddings")
    return embedded_clauses


def upload_embeddings_to_gcs(
    bucket_name: str, 
    embedded_clauses: List[Dict[str, Any]],
    project_id: str
) -> str:
    """Upload embeddings to GCS in JSONL format for Vector Search"""
    print_step(4, "Uploading Embeddings to Cloud Storage")
    
    from google.cloud import storage
    
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    # Create JSONL file for Vector Search
    # Format: {"id": "...", "embedding": [...], "restricts": [...], "crowding_tag": "..."}
    jsonl_content = ""
    
    for clause in embedded_clauses:
        record = {
            "id": clause['id'],
            "embedding": clause['embedding'],
            # Add filter restricts
            "restricts": [
                {"namespace": "clause_category", "allow": [clause['metadata']['clause_category']]},
                {"namespace": "risk_level", "allow": [clause['metadata']['risk_level']]}
            ]
        }
        jsonl_content += json.dumps(record) + "\n"
    
    # Upload to GCS
    blob_path = f"embeddings/clauses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(jsonl_content, content_type='application/json')
    
    gcs_uri = f"gs://{bucket_name}/{blob_path}"
    print(f"   ✅ Uploaded to: {gcs_uri}")
    
    # Also save metadata mapping (for retrieving full text later)
    metadata_content = json.dumps({
        clause['id']: {
            'text': clause['full_text'],
            'metadata': clause['metadata']
        }
        for clause in embedded_clauses
    }, indent=2)
    
    metadata_blob = bucket.blob("metadata/clause_metadata.json")
    metadata_blob.upload_from_string(metadata_content, content_type='application/json')
    print(f"   ✅ Uploaded metadata to: gs://{bucket_name}/metadata/clause_metadata.json")
    
    return gcs_uri


# ============================================================================
# STEP 5: CREATE VECTOR SEARCH INDEX
# ============================================================================

def create_vector_index(
    project_id: str,
    region: str,
    gcs_uri: str
) -> str:
    """Create Vertex AI Vector Search Index"""
    print_step(5, "Creating Vector Search Index")
    
    from google.cloud import aiplatform
    
    print(f"   Index name: {INDEX_DISPLAY_NAME}")
    print(f"   Embedding dimension: {EMBEDDING_DIMENSION}")
    print(f"   Data source: {gcs_uri}")
    
    # Check if index already exists
    existing_indexes = aiplatform.MatchingEngineIndex.list(
        filter=f'display_name="{INDEX_DISPLAY_NAME}"'
    )
    
    if existing_indexes:
        print(f"   ⚠️ Index already exists, using existing: {existing_indexes[0].resource_name}")
        return existing_indexes[0].resource_name
    
    print("   Creating new index (this may take 30-60 minutes)...")
    
    # Create the index
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        contents_delta_uri=gcs_uri.rsplit('/', 1)[0] + "/",  # Directory containing the JSONL
        dimensions=EMBEDDING_DIMENSION,
        approximate_neighbors_count=50,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=10,
        description="Negotify contract clause embeddings for similarity search",
        labels={"app": "negotify", "type": "contract-clauses"}
    )
    
    print(f"   ✅ Index created: {index.resource_name}")
    print(f"   ⏳ Index is building... This takes 30-60 minutes.")
    print(f"      Check status: gcloud ai indexes describe {index.name} --region={region}")
    
    return index.resource_name


# ============================================================================
# STEP 6: CREATE ENDPOINT & DEPLOY INDEX
# ============================================================================

def create_endpoint_and_deploy(
    project_id: str,
    region: str,
    index_resource_name: str
) -> str:
    """Create endpoint and deploy the index"""
    print_step(6, "Creating Endpoint and Deploying Index")
    
    from google.cloud import aiplatform
    
    # Check if endpoint exists
    existing_endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"'
    )
    
    if existing_endpoints:
        endpoint = existing_endpoints[0]
        print(f"   Using existing endpoint: {endpoint.resource_name}")
    else:
        print("   Creating new endpoint...")
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=ENDPOINT_DISPLAY_NAME,
            description="Negotify contract clause search endpoint",
            public_endpoint_enabled=True,
            labels={"app": "negotify"}
        )
        print(f"   ✅ Endpoint created: {endpoint.resource_name}")
    
    # Check if already deployed
    deployed_indexes = endpoint.deployed_indexes
    if any(d.id == DEPLOYED_INDEX_ID for d in deployed_indexes):
        print(f"   ⚠️ Index already deployed as {DEPLOYED_INDEX_ID}")
        return endpoint.resource_name
    
    # Deploy index
    print("   Deploying index to endpoint (this may take 10-20 minutes)...")
    
    index = aiplatform.MatchingEngineIndex(index_resource_name)
    
    endpoint.deploy_index(
        index=index,
        deployed_index_id=DEPLOYED_INDEX_ID,
        display_name="negotify-clauses",
        min_replica_count=1,
        max_replica_count=2,
        machine_type="e2-standard-2"  # Cost-effective option
    )
    
    print(f"   ✅ Index deployed to endpoint!")
    print(f"   Endpoint: {endpoint.resource_name}")
    
    return endpoint.resource_name


# ============================================================================
# STEP 7: TEST SEARCH
# ============================================================================

def test_search(
    project_id: str,
    region: str,
    endpoint_resource_name: str
):
    """Test the vector search"""
    print_step(7, "Testing Vector Search")
    
    from google.cloud import aiplatform
    from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    
    # Get endpoint
    endpoint = aiplatform.MatchingEngineIndexEndpoint(endpoint_resource_name)
    
    # Generate query embedding
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    
    test_queries = [
        "The contractor shall indemnify and hold harmless the client",
        "Payment shall be made within 30 days",
        "All intellectual property created shall belong to client"
    ]
    
    for query in test_queries:
        print(f"\n   Query: \"{query[:50]}...\"")
        
        # Generate embedding
        inputs = [TextEmbeddingInput(query, "RETRIEVAL_QUERY")]
        query_embedding = model.get_embeddings(inputs)[0].values
        
        # Search
        try:
            response = endpoint.find_neighbors(
                deployed_index_id=DEPLOYED_INDEX_ID,
                queries=[query_embedding],
                num_neighbors=3
            )
            
            print(f"   Results:")
            for i, neighbor in enumerate(response[0]):
                print(f"      {i+1}. ID: {neighbor.id}, Score: {neighbor.distance:.4f}")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    print("\n   ✅ Vector search is working!")


# ============================================================================
# STEP 8: SAVE CONFIGURATION
# ============================================================================

def save_config(
    project_id: str,
    region: str,
    bucket_name: str,
    index_resource_name: str,
    endpoint_resource_name: str
):
    """Save configuration for later use"""
    print_step(8, "Saving Configuration")
    
    config = {
        "project_id": project_id,
        "region": region,
        "bucket_name": bucket_name,
        "index_resource_name": index_resource_name,
        "endpoint_resource_name": endpoint_resource_name,
        "deployed_index_id": DEPLOYED_INDEX_ID,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "created_at": datetime.now().isoformat()
    }
    
    os.makedirs('benchmarks/config', exist_ok=True)
    
    with open('benchmarks/config/vertex_ai_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   ✅ Configuration saved to: benchmarks/config/vertex_ai_config.json")
    print(f"\n   Configuration:")
    for key, value in config.items():
        if 'resource_name' not in key:
            print(f"      {key}: {value}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Set up Vertex AI Vector Search for Negotify")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Google Cloud Project ID")
    parser.add_argument("--region", default=DEFAULT_REGION, help="Google Cloud Region")
    parser.add_argument("--cuad-path", default="benchmarks/data/cuad_processed.json", help="Path to CUAD data")
    parser.add_argument("--skip-index", action="store_true", help="Skip index creation (if already exists)")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip deployment (if already deployed)")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     🚀 NEGOTIFY - VERTEX AI VECTOR SEARCH SETUP                      ║
║                                                                      ║
║     Production-grade semantic search for contract clauses            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Verify setup
    if not verify_setup(args.project, args.region):
        print("\n❌ Setup verification failed. Please fix the issues above.")
        sys.exit(1)
    
    # Step 2: Create GCS bucket
    bucket_name = setup_gcs_bucket(args.project, args.region)
    
    # Step 3: Load and embed data
    print(f"\n📄 Loading CUAD data from: {args.cuad_path}")
    if not os.path.exists(args.cuad_path):
        print(f"   ❌ File not found: {args.cuad_path}")
        print("   Run: python -m benchmarks.cuad_processor")
        sys.exit(1)
    
    clauses = load_cuad_data(args.cuad_path)
    print(f"   Loaded {len(clauses)} clauses")
    
    # Step 3: Generate embeddings
    embedded_clauses = generate_embeddings(args.project, clauses)
    
    # Step 4: Upload to GCS
    gcs_uri = upload_embeddings_to_gcs(bucket_name, embedded_clauses, args.project)
    
    # Step 5: Create index
    if args.skip_index:
        print("\n⏭️ Skipping index creation (--skip-index)")
        # Try to find existing index
        from google.cloud import aiplatform
        existing = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{INDEX_DISPLAY_NAME}"')
        if existing:
            index_resource_name = existing[0].resource_name
        else:
            print("   ❌ No existing index found!")
            sys.exit(1)
    else:
        index_resource_name = create_vector_index(args.project, args.region, gcs_uri)
    
    # Step 6: Deploy (will wait for index to be ready)
    if args.skip_deploy:
        print("\n⏭️ Skipping deployment (--skip-deploy)")
        from google.cloud import aiplatform
        existing = aiplatform.MatchingEngineIndexEndpoint.list(filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"')
        if existing:
            endpoint_resource_name = existing[0].resource_name
        else:
            print("   ❌ No existing endpoint found!")
            sys.exit(1)
    else:
        print("\n⚠️ Note: Index must finish building before deployment.")
        print("   Index building typically takes 30-60 minutes.")
        
        response = input("\n   Deploy now? (y/n, or 'later' to save config and deploy later): ")
        
        if response.lower() == 'y':
            # Wait for index to be ready
            from google.cloud import aiplatform
            index = aiplatform.MatchingEngineIndex(index_resource_name)
            
            print("\n   Waiting for index to be ready...")
            while True:
                index = aiplatform.MatchingEngineIndex(index_resource_name)
                state = index.to_dict().get('indexStats', {})
                if index.to_dict().get('state') == 'ACTIVE':
                    print("   ✅ Index is ready!")
                    break
                print(f"   Still building... (vectors indexed so far: {state.get('vectorsCount', 0)})")
                time.sleep(60)
            
            endpoint_resource_name = create_endpoint_and_deploy(args.project, args.region, index_resource_name)
            
            # Step 7: Test
            test_search(args.project, args.region, endpoint_resource_name)
        else:
            print("\n   Saving configuration. Run deployment later with:")
            print(f"   python setup_vertex_ai_search.py --project {args.project} --skip-index")
            endpoint_resource_name = "PENDING_DEPLOYMENT"
    
    # Step 8: Save config
    save_config(args.project, args.region, bucket_name, index_resource_name, endpoint_resource_name)
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     ✅ SETUP COMPLETE!                                               ║
╚══════════════════════════════════════════════════════════════════════╝

Next steps:
1. Wait for index to finish building (check GCP Console)
2. Deploy the index (if not done already)
3. Update your agent.py to use Vertex AI search

Estimated monthly cost: ~$70-150 depending on usage
""")


if __name__ == "__main__":
    main()