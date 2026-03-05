"""
RAG Module - Retrieval Augmented Generation
Modular RAG system with ports and adapters.
Active implementation: RAGService (this module), consumed via JotaRAGSystem in core/agent_orchestrator.py.
"""

from .ports import (
    EmbeddingsPort,
    VectorStorePort, 
    RetrieverPort,
    KnowledgeBasePort,
    DocumentChunk,
    RetrievedChunk,
    RAGQuery,
    RAGResult
)

from .rag_service import RAGService
from .models import (
    RAGMetrics,
    QualityMetrics,
    RAGConfig,
    IndexMetadata
)

__all__ = [
    'EmbeddingsPort',
    'VectorStorePort',
    'RetrieverPort', 
    'KnowledgeBasePort',
    'RAGService',
    'DocumentChunk',
    'RetrievedChunk',
    'RAGQuery',
    'RAGResult',
    'RAGMetrics',
    'QualityMetrics',
    'RAGConfig',
    'IndexMetadata'
]
