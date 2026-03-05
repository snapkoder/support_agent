"""
Retriever Adapter - Combines embeddings and vector store for document retrieval
"""

import logging
import threading
from typing import List, Dict, Any, Optional
import json
import time
import asyncio

from ..ports import (
    RetrieverPort, EmbeddingsPort, VectorStorePort, 
    RetrievedChunk, RetrievalError, EmptyContextReason, EmptyContextDetails
)
from ..rag_facade import RAGFacade

logger = logging.getLogger(__name__)

class RetrieverAdapter(RetrieverPort):
    """
    Retriever adapter that combines embeddings and vector store
    """
    
    def __init__(
        self,
        embeddings_port: EmbeddingsPort,
        vector_store_port: VectorStorePort,
        default_top_k: int = 8,
        skip_compatibility_check: bool = False
    ):
        """Initialize retriever adapter"""
        self.embeddings_port = embeddings_port
        self.vector_store_port = vector_store_port
        self.default_top_k = default_top_k
        self.skip_compatibility_check = skip_compatibility_check
        self.logger = logging.getLogger(f"{__name__}.RetrieverAdapter")
        
        # Thread-safe compatibility checking
        self._compatibility_lock = threading.Lock()
        self._compatibility_checked = False
        self._is_compatible = None
        self._checked_signature_hash = None  # Track signature hash for runtime changes
        self._empty_context_details = None  # Cache empty context details
        
        self.logger.info(f"Retriever Adapter initialized (skip_compat={skip_compatibility_check})")
    
    def _check_compatibility(self) -> bool:
        """Check embedding compatibility with index (thread-safe with runtime change detection)"""
        with self._compatibility_lock:
            # Get current signature hash
            current_signature = self.embeddings_port.get_signature()
            current_hash = current_signature.stable_hash()
            
            # Check if signature changed at runtime
            if self._checked_signature_hash is not None and current_hash != self._checked_signature_hash:
                self.logger.warning(f"Embedding signature changed at runtime: {self._checked_signature_hash} -> {current_hash}")
                self.logger.info("Revalidating compatibility due to signature change")
                
                # Log structured event
                log_data = {
                    'event': 'rag_signature_changed_runtime',
                    'previous_hash': self._checked_signature_hash,
                    'current_hash': current_hash,
                    'timestamp': time.time()
                }
                self.logger.error(json.dumps(log_data, ensure_ascii=False))
                
                # Reset compatibility check to force revalidation
                self._compatibility_checked = False
                self._is_compatible = None
                self._empty_context_details = None  # Reset cached details
            
            if self._compatibility_checked:
                return self._is_compatible
            
            try:
                # Use RAG Facade to check compatibility
                is_compatible, empty_context_details = RAGFacade.check_index_compatibility(
                    self.embeddings_port, 
                    index_dir="./var/rag_index"
                )
                self._compatibility_checked = True
                self._checked_signature_hash = current_hash
                self._is_compatible = is_compatible
                self._empty_context_details = empty_context_details
                
                if not self._is_compatible:
                    self.logger.error("Embedding adapter incompatible with existing index")
                    
                    # Empty context details already set by RAGFacade
                    
                    # Trigger auto-rebuild if enabled
                    rebuild_triggered = RAGFacade.trigger_auto_rebuild_if_enabled(
                        self.embeddings_port,
                        index_dir="./var/rag_index"
                    )
                    
                    if rebuild_triggered:
                        self.logger.info("Auto-rebuild triggered - please run build_rag_index.py")
                
                return self._is_compatible
                
            except Exception as e:
                self.logger.error(f"Failed to check compatibility: {e}")
                self._compatibility_checked = True
                self._is_compatible = True  # Assume compatible on error
                return True
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 8, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            filters: Metadata filters
            
        Returns:
            List of retrieved chunks
        """
        if not query or not query.strip():
            return []
        
        start_time = time.time()
        
        # Check compatibility before any expensive operations (skip for in-memory mode)
        if not self.skip_compatibility_check and not self._check_compatibility():
            self.logger.warning("Skipping retrieval due to embedding signature incompatibility")
            return []  # Return empty list - RAG will continue with empty context
        
        try:
            # Generate query embedding
            embed_result = self.embeddings_port.embed([query])
            query_embedding = (await embed_result) if asyncio.iscoroutine(embed_result) else embed_result
            
            # Search vector store
            retrieved_chunks = await self.vector_store_port.similarity_search(
                query_vector=query_embedding[0],
                top_k=top_k,
                filters=filters
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Cache empty context details for zero hits
            if not retrieved_chunks:
                if self._empty_context_details is None:
                    self._empty_context_details = EmptyContextDetails(
                        reason=EmptyContextReason.RETRIEVAL_ZERO_HITS,
                        top_k=top_k,
                        filters=filters,
                        timestamp=time.time()
                    )
            
            return retrieved_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve chunks: {e}")
            raise RetrievalError(f"Retrieval failed: {e}")
    
    def get_empty_context_details(self) -> Optional[EmptyContextDetails]:
        """Get cached empty context details for the last retrieval"""
        return self._empty_context_details
    
    async def health_check(self) -> bool:
        """Check if retriever is healthy"""
        try:
            # Check both components
            eh = self.embeddings_port.health_check()
            embeddings_healthy = (await eh) if asyncio.iscoroutine(eh) else eh
            vh = self.vector_store_port.health_check()
            vector_store_healthy = (await vh) if asyncio.iscoroutine(vh) else vh
            
            overall_healthy = embeddings_healthy and vector_store_healthy
            
            self.logger.debug(f"Retriever health check: embeddings={embeddings_healthy}, vector_store={vector_store_healthy}")
            
            return overall_healthy
            
        except Exception as e:
            self.logger.error(f"Retriever health check failed: {e}")
            return False
