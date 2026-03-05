"""
RAG Ports - Contracts using typing.Protocol
Define interfaces for RAG components following dependency inversion principle
"""

from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime
from .models import RAGConfig, QualityMetrics, validate_query, EmptyContextReason, EmptyContextDetails, DocumentChunk, RetrievedChunk, EmbeddingSignature

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RAGQuery:
    """RAG query with parameters"""
    query: str
    agent_type: str
    top_k: int = 8
    filters: Optional[Dict[str, Any]] = None
    requires_rag: bool = False

@dataclass
class RAGResult:
    """RAG query result with enhanced empty context reasoning"""
    chunks: List[RetrievedChunk] = field(default_factory=list)
    rag_used: bool = False
    rag_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Computed properties with defaults
    rag_hits: int = 0
    rag_sources: List[str] = field(default_factory=list)
    rag_domains: List[str] = field(default_factory=list)
    avg_score: float = 0.0
    model_used: str = "unknown"
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    rag_always_on: bool = True
    local_first: bool = True
    local_hits: int = 0
    external_hits: int = 0
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    retrieval_latency_ms: float = 0.0
    # Enhanced empty context reasoning
    empty_context_reason: Optional[EmptyContextReason] = None
    empty_context_details: Optional[EmptyContextDetails] = None

# ============================================================================
# PORTS (INTERFACES)
# ============================================================================

@runtime_checkable
class EmbeddingsPort(Protocol):
    """Port for embedding generation"""
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        ...
    
    def get_signature(self) -> EmbeddingSignature:
        """
        Get the embedding signature for this adapter.
        
        Returns:
            EmbeddingSignature object
        """
        ...
    
    async def health_check(self) -> bool:
        """Check if embedding service is healthy"""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get embedding model information"""
        ...

@runtime_checkable
class VectorStorePort(Protocol):
    """Port for vector storage and retrieval"""
    
    async def upsert(self, chunks: List[DocumentChunk]) -> None:
        """
        Insert or update document chunks.
        
        Args:
            chunks: List of chunks to store
        """
        ...
    
    async def similarity_search(
        self, 
        query_vector: List[float], 
        top_k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Search for similar chunks.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of retrieved chunks with scores
        """
        ...
    
    async def persist(self) -> None:
        """Persist vector store to disk"""
        ...
    
    async def load(self) -> None:
        """Load vector store from disk"""
        ...
    
    async def health_check(self) -> bool:
        """Check if vector store is healthy"""
        ...
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        ...

@runtime_checkable
class RetrieverPort(Protocol):
    """Port for document retrieval"""
    
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
            List of retrieved chunks with scores
        """
        ...
    
    async def health_check(self) -> bool:
        """Check if retriever is healthy"""
        ...

@runtime_checkable
class KnowledgeBasePort(Protocol):
    """Port for knowledge base document management"""
    
    async def list_documents(self) -> List[str]:
        """
        List all documents in knowledge base.
        
        Returns:
            List of document file paths
        """
        ...
    
    async def load_documents(self, file_paths: List[str]) -> List[str]:
        """
        Load documents from file paths.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of document contents
        """
        ...
    
    async def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Document metadata
        """
        ...
    
    async def health_check(self) -> bool:
        """Check if knowledge base is healthy"""
        ...

# ============================================================================
# EXTERNAL KNOWLEDGE PORT
# ============================================================================

class ExternalKnowledgePort(Protocol):
    """Port for external knowledge base integration"""
    
    async def query_external_knowledge(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """
        Query external knowledge base for relevant chunks.
        
        Args:
            query: Search query
            filters: Metadata filters
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks from external source
        """
        ...
    
    async def health_check(self) -> bool:
        """Check if external knowledge source is healthy"""
        ...

# ============================================================================
# EXCEPTIONS
# ============================================================================

class RAGError(Exception):
    """Base exception for RAG operations"""
    pass

class EmbeddingError(RAGError):
    """Exception for embedding operations"""
    pass

class VectorStoreError(RAGError):
    """Exception for vector store operations"""
    pass

class RetrievalError(RAGError):
    """Exception for retrieval operations"""
    pass

class KnowledgeBaseError(RAGError):
    """Exception for knowledge base operations"""
    pass
