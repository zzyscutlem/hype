"""
Principle Memory module for HyPE agent system.

This module implements the PrincipleMemory class which manages storage and retrieval
of principles using Milvus vector database with BGE-large-en embeddings.

Key features:
- Semantic embedding computation using BGE-large-en (1024-dimensional)
- Efficient similarity search using Milvus with HNSW indexing
- Credit-weighted retrieval combining semantic similarity and credit scores
- Semantic deduplication to prevent redundant principles
"""

from typing import List, Optional, Tuple
import logging
from datetime import datetime
import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from sentence_transformers import SentenceTransformer

from ..core.data_models import Principle
from ..core.config import PrincipleMemoryConfig
from ..utils.error_handlers import ErrorHandler, DatabaseOperationError, retry_with_backoff


logger = logging.getLogger(__name__)


class PrincipleMemory:
    """
    Principle Memory for storing and retrieving principles with semantic embeddings.
    
    This class manages a vector database of principles, providing:
    - Semantic embedding computation using BGE-large-en
    - Credit-weighted retrieval
    - Semantic deduplication
    - Efficient similarity search
    
    Attributes:
        config: Configuration for principle memory
        embedding_model: BGE-large-en model for computing embeddings
        collection: Milvus collection for storing principles
    """
    
    def __init__(self, config: Optional[PrincipleMemoryConfig] = None):
        """
        Initialize Principle Memory.
        
        Args:
            config: Configuration for principle memory. If None, uses defaults.
        """
        self.config = config or PrincipleMemoryConfig()
        self.embedding_model = None
        self.collection = None
        self._connected = False
        self.error_handler = ErrorHandler()
        
        logger.info(f"Initializing PrincipleMemory with config: {self.config}")
    
    def connect(self):
        """
        Connect to Milvus and initialize the collection.
        
        This method:
        1. Establishes connection to Milvus server or Milvus Lite
        2. Loads the BGE-large-en embedding model
        3. Creates or loads the principles collection
        """
        if self._connected:
            logger.warning("Already connected to Milvus")
            return
        
        try:
            # Check if using Milvus Lite (embedded mode)
            use_milvus_lite = getattr(self.config, 'use_milvus_lite', False)
            
            if use_milvus_lite:
                # Use Milvus Lite (embedded mode)
                milvus_lite_path = getattr(self.config, 'milvus_lite_path', './data/milvus_lite.db')
                logger.info(f"Connecting to Milvus Lite at {milvus_lite_path}")
                
                # Create directory if it doesn't exist
                import os
                os.makedirs(os.path.dirname(milvus_lite_path), exist_ok=True)
                
                # Connect to Milvus Lite
                connections.connect(
                    alias="default",
                    uri=milvus_lite_path  # Use local file path for Milvus Lite
                )
                logger.info("✅ Connected to Milvus Lite (embedded mode)")
            else:
                # Connect to Milvus server
                logger.info(f"Connecting to Milvus server at {self.config.milvus_host}:{self.config.milvus_port}")
                connections.connect(
                    alias="default",
                    host=self.config.milvus_host,
                    port=self.config.milvus_port
                )
                logger.info("✅ Connected to Milvus server")
            
            # Load embedding model
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            
            # Force use of local cache
            import os
            from pathlib import Path
            
            model_name = self.config.embedding_model
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            
            # Convert model name to cache directory format
            cache_model_name = "models--" + model_name.replace("/", "--")
            model_cache_path = cache_dir / cache_model_name
            
            if model_cache_path.exists():
                logger.info(f"Found local cache: {model_cache_path}")
                
                # Find the snapshot directory
                snapshots_dir = model_cache_path / "snapshots"
                if snapshots_dir.exists():
                    snapshot_dirs = list(snapshots_dir.iterdir())
                    if snapshot_dirs:
                        snapshot_path = snapshot_dirs[0]
                        logger.info(f"Using snapshot: {snapshot_path.name}")
                        
                        # Force offline mode
                        os.environ['HF_HUB_OFFLINE'] = '1'
                        os.environ['TRANSFORMERS_OFFLINE'] = '1'
                        
                        try:
                            # Load directly from snapshot path
                            self.embedding_model = SentenceTransformer(str(snapshot_path))
                            logger.info("✅ Successfully loaded embedding model from local cache")
                        except Exception as e:
                            logger.error(f"Failed to load from snapshot: {e}")
                            raise RuntimeError(
                                f"Cannot load embedding model from local cache. "
                                f"Please ensure {model_cache_path} contains valid model files."
                            )
                    else:
                        raise RuntimeError(f"No snapshots found in {snapshots_dir}")
                else:
                    raise RuntimeError(f"Snapshots directory not found: {snapshots_dir}")
            else:
                raise RuntimeError(
                    f"Embedding model not found in local cache: {model_cache_path}\n"
                    f"Please download the model first or check the cache path."
                )
            
            # Create or load collection
            self._initialize_collection()
            
            self._connected = True
            logger.info("Successfully connected to Milvus and initialized collection")
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from Milvus."""
        if self._connected:
            connections.disconnect(alias="default")
            self._connected = False
            logger.info("Disconnected from Milvus")
    
    def _initialize_collection(self):
        """
        Create or load the Milvus collection for principles.
        
        Schema:
        - id: Primary key (VARCHAR)
        - text: Principle text (VARCHAR)
        - embedding: Semantic embedding (FLOAT_VECTOR, 1024-dim)
        - credit_score: Credit score (FLOAT)
        - application_count: Number of applications (INT64)
        - created_at: Creation timestamp (INT64, Unix timestamp)
        - last_used: Last used timestamp (INT64, Unix timestamp)
        - source_trajectory_id: Optional source trajectory (VARCHAR)
        """
        collection_name = self.config.collection_name
        
        # Check if collection exists
        if utility.has_collection(collection_name):
            logger.info(f"Loading existing collection: {collection_name}")
            self.collection = Collection(collection_name)
            self.collection.load()
            return
        
        # Create new collection
        logger.info(f"Creating new collection: {collection_name}")
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.embedding_dim),
            FieldSchema(name="credit_score", dtype=DataType.FLOAT),
            FieldSchema(name="application_count", dtype=DataType.INT64),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="last_used", dtype=DataType.INT64),
            FieldSchema(name="source_trajectory_id", dtype=DataType.VARCHAR, max_length=64),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Principles with semantic embeddings and credit scores"
        )
        
        # Create collection
        self.collection = Collection(
            name=collection_name,
            schema=schema
        )
        
        # Choose index type based on whether using Milvus Lite
        use_milvus_lite = getattr(self.config, 'use_milvus_lite', False)
        
        if use_milvus_lite:
            # Milvus Lite only supports FLAT, IVF_FLAT, and AUTOINDEX
            # Use AUTOINDEX for simplicity (automatically chooses best index)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "AUTOINDEX",
                "params": {}
            }
            logger.info("Creating AUTOINDEX on embedding field (Milvus Lite mode)")
        else:
            # Use HNSW for better performance on full Milvus server
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            logger.info("Creating HNSW index on embedding field (Milvus server mode)")
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        # Load collection into memory
        self.collection.load()
        logger.info(f"Collection {collection_name} created and loaded")
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute semantic embedding for text using BGE-large-en.
        
        Args:
            text: Text to embed
            
        Returns:
            1024-dimensional embedding vector
            
        Raises:
            RuntimeError: If not connected to Milvus
            ValueError: If text is empty
        """
        if not self._connected:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Compute embedding
        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False
        )
        
        # Ensure correct shape
        embedding = np.array(embedding, dtype=np.float32)
        if embedding.shape != (self.config.embedding_dim,):
            raise ValueError(
                f"Expected embedding dimension {self.config.embedding_dim}, "
                f"got {embedding.shape}"
            )
        
        return embedding
    
    def insert(self, principle: Principle) -> bool:
        """
        Insert a principle into memory with semantic deduplication.
        
        This method checks for semantic duplicates before insertion.
        If a duplicate is found (similarity > threshold), the principles
        are merged and the existing principle is updated.
        
        Args:
            principle: Principle to insert
            
        Returns:
            True if inserted as new principle, False if merged with existing
            
        Raises:
            RuntimeError: If not connected to Milvus
        """
        if not self._connected:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")
        
        logger.debug(f"Inserting principle: {principle.id}")
        
        # Check for duplicates
        duplicates = self._find_duplicates(principle)
        
        if duplicates:
            # Merge with the most similar duplicate
            logger.info(f"Found {len(duplicates)} duplicate(s), merging with most similar")
            self._merge_with_duplicate(principle, duplicates[0])
            return False
        
        # Insert new principle
        self._insert_principle(principle)
        logger.info(f"Inserted new principle: {principle.id}")
        return True
    
    def _find_duplicates(self, principle: Principle) -> List[Tuple[str, float]]:
        """
        Find duplicate principles based on semantic similarity.
        
        Args:
            principle: Principle to check for duplicates
            
        Returns:
            List of (principle_id, similarity_score) tuples for duplicates
        """
        # Skip search if collection is empty
        if self.collection.num_entities == 0:
            return []
        
        try:
            # Search for similar principles
            search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
            
            results = self.collection.search(
                data=[principle.embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=5,  # Check top 5 most similar
                output_fields=["id", "text"]
            )
            
            duplicates = []
            for hits in results:
                for hit in hits:
                    similarity = hit.score
                    if similarity > self.config.duplicate_threshold:
                        duplicates.append((hit.entity.get("id"), similarity))
            
            return duplicates
        except Exception as e:
            logger.warning(f"Failed to search for duplicates: {e}")
            return []
    
    def _merge_with_duplicate(self, new_principle: Principle, duplicate_info: Tuple[str, float]):
        """
        Merge a new principle with an existing duplicate.
        
        The merged credit score is computed as:
        merged_credit = (credit1 * count1 + credit2 * count2) / (count1 + count2)
        
        Args:
            new_principle: New principle to merge
            duplicate_info: (principle_id, similarity_score) of duplicate
        """
        duplicate_id, similarity = duplicate_info
        
        # Fetch existing principle
        query_result = self.collection.query(
            expr=f'id == "{duplicate_id}"',
            output_fields=["credit_score", "application_count"]
        )
        
        if not query_result:
            logger.warning(f"Duplicate principle {duplicate_id} not found, inserting as new")
            self._insert_principle(new_principle)
            return
        
        existing = query_result[0]
        existing_credit = existing["credit_score"]
        existing_count = existing["application_count"]
        
        # Compute merged credit score
        new_credit = new_principle.credit_score
        new_count = new_principle.application_count
        
        total_count = existing_count + new_count
        if total_count > 0:
            merged_credit = (existing_credit * existing_count + new_credit * new_count) / total_count
        else:
            merged_credit = (existing_credit + new_credit) / 2
        
        merged_count = total_count
        
        # Update existing principle
        self.collection.delete(expr=f'id == "{duplicate_id}"')
        
        # Re-insert with updated values
        updated_principle = Principle(
            id=duplicate_id,
            text=existing.get("text", new_principle.text),
            embedding=new_principle.embedding,
            credit_score=merged_credit,
            application_count=merged_count,
            created_at=datetime.fromtimestamp(existing.get("created_at", 0)),
            last_used=datetime.now(),
            source_trajectory_id=existing.get("source_trajectory_id")
        )
        
        self._insert_principle(updated_principle)
        logger.info(
            f"Merged principles: {duplicate_id} "
            f"(credit: {existing_credit:.3f} -> {merged_credit:.3f}, "
            f"count: {existing_count} -> {merged_count})"
        )
    
    def _insert_principle(self, principle: Principle):
        """
        Insert a principle into the collection.
        
        Args:
            principle: Principle to insert
        """
        data = [
            [principle.id],
            [principle.text],
            [principle.embedding.tolist()],
            [principle.credit_score],
            [principle.application_count],
            [int(principle.created_at.timestamp())],
            [int(principle.last_used.timestamp())],
            [principle.source_trajectory_id or ""],
        ]
        
        self.collection.insert(data)
        self.collection.flush()
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Principle]:
        """
        Retrieve top-K principles using credit-weighted semantic search.
        
        Retrieval score = α * semantic_similarity + (1-α) * normalized_credit_score
        where α is the semantic_weight from config.
        
        Args:
            query: Query text
            top_k: Number of principles to retrieve. If None, uses config default.
            
        Returns:
            List of top-K principles ranked by retrieval score
            
        Raises:
            RuntimeError: If not connected to Milvus
        """
        if not self._connected:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")
        
        k = top_k or self.config.top_k
        
        # Define query function for error handler
        def _query():
            # Compute query embedding
            query_embedding = self.compute_embedding(query)
            
            # Search for similar principles (retrieve more than k for credit weighting)
            search_params = {"metric_type": "COSINE", "params": {"ef": 200}}
            
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=min(k * 3, 100),  # Retrieve 3x for credit weighting
                output_fields=["id", "text", "credit_score", "application_count", 
                              "created_at", "last_used", "source_trajectory_id"]
            )
            
            if not results or not results[0]:
                logger.warning(f"No principles found for query: {query[:50]}...")
                return []
            
            # Compute credit-weighted scores
            principles_with_scores = []
            credit_scores = [hit.entity.get("credit_score") for hit in results[0]]
            
            # Normalize credit scores to [0, 1]
            if credit_scores:
                min_credit = min(credit_scores)
                max_credit = max(credit_scores)
                credit_range = max_credit - min_credit
            else:
                min_credit = 0
                credit_range = 1
            
            for hit in results[0]:
                semantic_sim = hit.score
                credit = hit.entity.get("credit_score")
                
                # Normalize credit score
                if credit_range > 0:
                    normalized_credit = (credit - min_credit) / credit_range
                else:
                    normalized_credit = 0.5
                
                # Compute retrieval score
                alpha = self.config.semantic_weight
                retrieval_score = alpha * semantic_sim + (1 - alpha) * normalized_credit
                
                # Create Principle object
                principle = Principle(
                    id=hit.entity.get("id"),
                    text=hit.entity.get("text"),
                    embedding=query_embedding,  # Use query embedding as placeholder
                    credit_score=credit,
                    application_count=hit.entity.get("application_count"),
                    created_at=datetime.fromtimestamp(hit.entity.get("created_at")),
                    last_used=datetime.fromtimestamp(hit.entity.get("last_used")),
                    source_trajectory_id=hit.entity.get("source_trajectory_id") or None
                )
                
                principles_with_scores.append((principle, retrieval_score))
            
            # Sort by retrieval score and return top-K
            principles_with_scores.sort(key=lambda x: x[1], reverse=True)
            top_principles = [p for p, _ in principles_with_scores[:k]]
            
            logger.debug(f"Retrieved {len(top_principles)} principles for query")
            return top_principles
        
        # Define fallback function for random sampling
        def _fallback(context):
            logger.warning("Database query failed, using random principle sampling")
            return self._random_sample_principles(k)
        
        # Try query with error handling
        try:
            return _query()
        except Exception as e:
            logger.warning(f"Principle retrieval failed: {e}")
            return self.error_handler.handle_database_error(
                error=e,
                query_fn=lambda **kwargs: _query(),
                query_args={'query': query, 'top_k': k},
                fallback_fn=_fallback
            )
    
    def update_credit(self, principle_id: str, credit_delta: float):
        """
        Update the credit score of a principle.
        
        Args:
            principle_id: ID of the principle to update
            credit_delta: Amount to add to credit score
            
        Raises:
            RuntimeError: If not connected to Milvus
            ValueError: If principle not found
        """
        if not self._connected:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")
        
        # Fetch existing principle
        query_result = self.collection.query(
            expr=f'id == "{principle_id}"',
            output_fields=["id", "text", "embedding", "credit_score", 
                          "application_count", "created_at", "last_used", 
                          "source_trajectory_id"]
        )
        
        if not query_result:
            raise ValueError(f"Principle {principle_id} not found")
        
        existing = query_result[0]
        new_credit = existing["credit_score"] + credit_delta
        new_count = existing["application_count"] + 1
        
        # Delete old entry
        self.collection.delete(expr=f'id == "{principle_id}"')
        
        # Re-insert with updated values
        data = [
            [principle_id],
            [existing["text"]],
            [existing["embedding"]],
            [new_credit],
            [new_count],
            [existing["created_at"]],
            [int(datetime.now().timestamp())],
            [existing.get("source_trajectory_id", "")],
        ]
        
        self.collection.insert(data)
        self.collection.flush()
        
        logger.debug(
            f"Updated principle {principle_id}: "
            f"credit {existing['credit_score']:.3f} -> {new_credit:.3f}"
        )
    
    def prune_low_credit(self, threshold: Optional[float] = None):
        """
        Remove principles with credit score below threshold.
        
        Args:
            threshold: Minimum credit score. If None, uses config default.
            
        Raises:
            RuntimeError: If not connected to Milvus
        """
        if not self._connected:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")
        
        threshold = threshold or self.config.min_credit_score
        
        # Delete principles below threshold
        expr = f"credit_score < {threshold}"
        
        logger.info(f"Pruning principles with credit_score < {threshold}")
        self.collection.delete(expr=expr)
        self.collection.flush()
        
        logger.info("Pruning completed")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the principle memory.
        
        Returns:
            Dictionary with statistics
        """
        if not self._connected:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")
        
        num_entities = self.collection.num_entities
        
        return {
            "num_principles": num_entities,
            "collection_name": self.config.collection_name,
            "embedding_dim": self.config.embedding_dim,
        }
    
    def clear(self):
        """
        Clear all principles from memory.
        
        WARNING: This deletes all data in the collection.
        """
        if not self._connected:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")
        
        logger.warning("Clearing all principles from memory")
        self.collection.delete(expr="id != ''")
        self.collection.flush()
        logger.info("All principles cleared")
    
    def _random_sample_principles(self, k: int) -> List[Principle]:
        """
        Randomly sample k principles from memory (fallback for failed queries).
        
        Args:
            k: Number of principles to sample
            
        Returns:
            List of randomly sampled principles
        """
        try:
            # Query all principles (limited to reasonable number)
            results = self.collection.query(
                expr="id != ''",
                output_fields=["id", "text", "credit_score", "application_count",
                              "created_at", "last_used", "source_trajectory_id"],
                limit=min(k * 10, 1000)
            )
            
            if not results:
                logger.warning("No principles available for random sampling")
                return []
            
            # Randomly sample k principles
            import random
            sampled = random.sample(results, min(k, len(results)))
            
            # Convert to Principle objects
            principles = []
            for entity in sampled:
                # Create dummy embedding
                dummy_embedding = np.zeros(self.config.embedding_dim)
                
                principle = Principle(
                    id=entity.get("id"),
                    text=entity.get("text"),
                    embedding=dummy_embedding,
                    credit_score=entity.get("credit_score"),
                    application_count=entity.get("application_count"),
                    created_at=datetime.fromtimestamp(entity.get("created_at")),
                    last_used=datetime.fromtimestamp(entity.get("last_used")),
                    source_trajectory_id=entity.get("source_trajectory_id") or None
                )
                principles.append(principle)
            
            logger.info(f"Randomly sampled {len(principles)} principles as fallback")
            return principles
            
        except Exception as e:
            logger.error(f"Random sampling failed: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
