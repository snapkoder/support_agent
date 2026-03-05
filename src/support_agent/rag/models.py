"""
RAG Models - Data structures and configurations for RAG system
"""

import os
import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

# ============================================================================
# DATA MODELS (moved from ports.py to break circular import)
# ============================================================================

from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Document chunk with metadata"""
    chunk_id: str
    content: str
    source_file: str
    section_title: str
    domain: str  # atendimento_geral, open_finance, golpe_med, criacao_conta
    agent_type: str
    language: str = "pt-BR"
    version: str = "1.0"
    chunk_index: int = 0
    total_chunks: int = 0
    breadcrumb: str = ""
    content_type: str = "text"
    priority: int = 1
    last_updated: Optional[str] = None

@dataclass
class RetrievedChunk:
    """Retrieved chunk with similarity score (nested DocumentChunk + score)"""
    chunk: DocumentChunk
    score: float = 0.0
    retrieval_method: str = "similarity"

# Import ports after moving data classes
# from .ports import EmbeddingsPort  # Moved to functions to avoid circular import

# EmbeddingProvider as simple string constants to avoid circular import
class EmbeddingProvider:
    OPENAI = "openai"
    LOCAL = "local"

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingSignature:
    """Unique signature for embedding configuration"""
    provider: str  # "openai" | "local"
    model_name: str
    dimensions: int
    normalize: bool = False
    extra: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signature to dictionary"""
        result = {
            'provider': self.provider,
            'model_name': self.model_name,
            'dimensions': self.dimensions,
            'normalize': self.normalize
        }
        if self.extra:
            result['extra'] = self.extra
        return result
    
    def stable_hash(self) -> str:
        """Generate stable hash for this signature"""
        # Create deterministic JSON
        signature_dict = self.to_dict()
        
        # Sort keys for deterministic hash
        sorted_dict = json.dumps(signature_dict, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        return hashlib.sha256(sorted_dict.encode('utf-8')).hexdigest()[:16]
    
    def is_compatible_with(self, other: 'EmbeddingSignature') -> bool:
        """Check if this signature is compatible with another"""
        return (
            self.provider == other.provider and
            self.model_name == other.model_name and
            self.dimensions == other.dimensions and
            self.normalize == other.normalize
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingSignature':
        """Create signature from dictionary"""
        return cls(
            provider=data.get('provider', 'local'),
            model_name=data.get('model_name', 'unknown'),
            dimensions=data.get('dimensions', 0),
            normalize=data.get('normalize', False),
            extra=data.get('extra')
        )
    
    def __str__(self) -> str:
        """String representation of signature"""
        return f"{self.provider}:{self.model_name}:{self.dimensions}d"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"EmbeddingSignature({self.__str__()})"

# Common signature presets
OPENAI_TEXT_EMBEDDING_3_SMALL = EmbeddingSignature(
    provider=EmbeddingProvider.OPENAI,
    model_name="text-embedding-3-small",
    dimensions=1536,
    normalize=False
)

OPENAI_TEXT_EMBEDDING_3_LARGE = EmbeddingSignature(
    provider=EmbeddingProvider.OPENAI,
    model_name="text-embedding-3-large",
    dimensions=3072,
    normalize=False
)

LOCAL_TFIDF_384 = EmbeddingSignature(
    provider=EmbeddingProvider.LOCAL,
    model_name="tfidf",
    dimensions=384,
    normalize=True
)

LOCAL_MINILM_384 = EmbeddingSignature(
    provider=EmbeddingProvider.LOCAL,
    model_name="minilm",
    dimensions=384,
    normalize=False
)

# ============================================================================
# EMPTY CONTEXT REASONING
# ============================================================================

from enum import Enum

class EmptyContextReason(Enum):
    """Canonical reasons for empty RAG context"""
    INCOMPATIBLE_SIGNATURE = "incompatible_signature"
    METADATA_MISSING = "metadata_missing"
    METADATA_CORRUPT = "metadata_corrupt"
    SIGNATURE_INCOMPLETE = "signature_incomplete"
    RETRIEVAL_ZERO_HITS = "retrieval_zero_hits"
    INDEX_MISSING = "index_missing"
    INDEX_UNBUILT = "index_unbuilt"
    REBUILD_IN_PROGRESS = "rebuild_in_progress"
    UNEXPECTED_ERROR = "unexpected_error"

@dataclass
class EmptyContextDetails:
    """Structured details for empty context reasoning"""
    reason: EmptyContextReason
    index_path: Optional[str] = None
    expected_signature: Optional[Dict[str, Any]] = None
    current_signature: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    top_k: Optional[int] = None
    score_threshold: Optional[float] = None
    retrieval_count: Optional[int] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    timestamp: Optional[float] = None

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

class IndexIncompatibleError(RAGError):
    """Exception raised when index embedding signature is incompatible with current adapter"""
    
    def __init__(self, message: str, expected_signature: dict, current_signature: dict):
        super().__init__(message)
        self.expected_signature = expected_signature
        self.current_signature = current_signature
    
    def __str__(self) -> str:
        return f"{super().__str__()}\nExpected: {self.expected_signature}\nCurrent: {self.current_signature}"

@dataclass
class RAGMetrics:
    """Metrics for RAG operations"""
    total_queries: int = 0
    rag_used_count: int = 0
    avg_rag_latency_ms: float = 0.0
    avg_chunks_retrieved: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def rag_usage_rate(self) -> float:
        """Calculate RAG usage rate"""
        if self.total_queries == 0:
            return 0.0
        return (self.rag_used_count / self.total_queries) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return (self.cache_hits / total_requests) * 100
    
    def update_query(self, rag_used: bool, rag_latency_ms: float, chunks_count: int):
        """Update metrics after a query"""
        self.total_queries += 1
        if rag_used:
            self.rag_used_count += 1
            # Update running average for latency
            if self.rag_used_count == 1:
                self.avg_rag_latency_ms = rag_latency_ms
            else:
                self.avg_rag_latency_ms = (
                    (self.avg_rag_latency_ms * (self.rag_used_count - 1) + rag_latency_ms) / 
                    self.rag_used_count
                )
            # Update running average for chunks
            if self.rag_used_count == 1:
                self.avg_chunks_retrieved = chunks_count
            else:
                self.avg_chunks_retrieved = (
                    (self.avg_chunks_retrieved * (self.rag_used_count - 1) + chunks_count) / 
                    self.rag_used_count
                )

@dataclass
class QualityMetrics:
    """Metrics for response quality assessment"""
    short_responses_count: int = 0
    negatives_count: int = 0
    template_responses_count: int = 0
    total_responses: int = 0
    avg_response_length: float = 0.0
    
    @property
    def short_response_rate(self) -> float:
        """Calculate short response rate"""
        if self.total_responses == 0:
            return 0.0
        return (self.short_responses_count / self.total_responses) * 100
    
    @property
    def negative_response_rate(self) -> float:
        """Calculate negative response rate"""
        if self.total_responses == 0:
            return 0.0
        return (self.negatives_count / self.total_responses) * 100
    
    @property
    def template_response_rate(self) -> float:
        """Calculate template response rate"""
        if self.total_responses == 0:
            return 0.0
        return (self.template_responses_count / self.total_responses) * 100
    
    def update_response(self, response_text: str, was_rag_used: bool):
        """Update metrics after a response"""
        self.total_responses += 1
        
        # Update average response length
        response_length = len(response_text)
        if self.total_responses == 1:
            self.avg_response_length = response_length
        else:
            self.avg_response_length = (
                (self.avg_response_length * (self.total_responses - 1) + response_length) / 
                self.total_responses
            )
        
        # Check for quality issues
        if response_length < 60:  # Short response threshold
            self.short_responses_count += 1
        
        # Check for negative responses
        negative_phrases = ["não sei", "não tenho", "não posso", "desculpe", "infelizmente"]
        if any(phrase.lower() in response_text.lower() for phrase in negative_phrases):
            self.negatives_count += 1
        
        # Check for template responses (when RAG was used but response is generic)
        if was_rag_used and response_length < 100:
            template_phrases = ["olá", "bem-vindo", "posso ajudar", "em que posso ajudar"]
            if any(phrase.lower() in response_text.lower() for phrase in template_phrases):
                self.template_responses_count += 1

@dataclass
class RAGConfig:
    """Configuration for RAG operations"""
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Retrieval
    default_top_k: int = 8
    similarity_threshold: float = 0.7
    
    # Quality
    quality_retry_enabled: bool = True
    quality_min_chars: int = 60
    quality_max_retries: int = 2
    
    # Debug
    debug_citations: bool = False
    
    # Persistence
    index_dir: str = "./var/rag_index"
    persist_enabled: bool = True
    
    # Performance
    cache_enabled: bool = True
    cache_size: int = 1000
    
    # RAG Always On
    rag_always_on: bool = True
    rag_top_k: int = 6
    
    # Local-First and External KB
    local_first: bool = True
    min_local_hits: int = 1
    external_kb_enabled: bool = False
    rag_two_stage: bool = False
    min_score: float = 0.0
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create config from environment variables"""
        import os
        
        return cls(
            chunk_size=int(os.getenv('RAG_CHUNK_SIZE', '500')),
            chunk_overlap=int(os.getenv('RAG_CHUNK_OVERLAP', '50')),
            default_top_k=int(os.getenv('RAG_DEFAULT_TOP_K', '8')),
            similarity_threshold=float(os.getenv('RAG_SIMILARITY_THRESHOLD', '0.7')),
            quality_retry_enabled=os.getenv('RAG_QUALITY_RETRY', 'true').lower() == 'true',
            quality_min_chars=int(os.getenv('RAG_QUALITY_MIN_CHARS', '60')),
            quality_max_retries=int(os.getenv('RAG_QUALITY_MAX_RETRIES', '2')),
            debug_citations=os.getenv('RAG_DEBUG_CITATIONS', 'false').lower() == 'true',
            index_dir=os.getenv('RAG_INDEX_DIR', './var/rag_index'),
            persist_enabled=os.getenv('RAG_PERSIST_ENABLED', 'true').lower() == 'true',
            cache_enabled=os.getenv('RAG_CACHE_ENABLED', 'true').lower() == 'true',
            cache_size=int(os.getenv('RAG_CACHE_SIZE', '1000')),
            rag_always_on=os.getenv('RAG_ALWAYS_ON', 'true').lower() == 'true',
            rag_top_k=int(os.getenv('RAG_TOP_K', '6')),
            local_first=os.getenv('RAG_LOCAL_FIRST', 'true').lower() == 'true',
            min_local_hits=int(os.getenv('RAG_MIN_LOCAL_HITS', '1')),
            external_kb_enabled=os.getenv('EXTERNAL_KB_ENABLED', 'false').lower() == 'true',
            rag_two_stage=os.getenv('RAG_TWO_STAGE', 'false').lower() == 'true',
            min_score=float(os.getenv('RAG_MIN_SCORE', '0.0'))
        )

@dataclass
class IndexMetadata:
    """Metadata for the RAG index"""
    corpus_hash: str
    total_chunks: int
    total_documents: int
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"
    config_hash: str = ""
    embedding_signature: Optional[Dict[str, Any]] = None
    adapter_name: Optional[str] = None
    store_type: Optional[str] = None  # "chroma" or "sqlite"
    collection_name: Optional[str] = None
    index_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'corpus_hash': self.corpus_hash,
            'total_chunks': self.total_chunks,
            'total_documents': self.total_documents,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'config_hash': self.config_hash,
            'embedding_signature': self.embedding_signature,
            'adapter_name': self.adapter_name,
            'store_type': self.store_type,
            'collection_name': self.collection_name,
            'index_path': self.index_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexMetadata':
        """Create from dictionary"""
        return cls(
            corpus_hash=data['corpus_hash'],
            total_chunks=data['total_chunks'],
            total_documents=data['total_documents'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            version=data.get('version', '1.0'),
            config_hash=data.get('config_hash', ''),
            embedding_signature=data.get('embedding_signature'),
            adapter_name=data.get('adapter_name'),
            store_type=data.get('store_type'),
            collection_name=data.get('collection_name'),
            index_path=data.get('index_path')
        )
    
    def save_to_file(self, file_path: str) -> None:
        """Save metadata to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Index metadata saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save index metadata: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, file_path: str) -> Optional['IndexMetadata']:
        """Load metadata from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            metadata = cls.from_dict(data)
            logger.info(f"Index metadata loaded from {file_path}")
            return metadata
        except FileNotFoundError:
            logger.warning(f"Index metadata file not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load index metadata: {e}")
            return None

def validate_chunk(chunk: 'DocumentChunk') -> bool:
    """Validate chunk data"""
    if not chunk.content or not chunk.content.strip():
        return False
    
    if not chunk.chunk_id or not chunk.chunk_id.strip():
        return False
    
    if not chunk.source_file or not chunk.source_file.strip():
        return False
    
    if not chunk.domain or chunk.domain not in ['atendimento_geral', 'open_finance', 'golpe_med', 'criacao_conta']:
        return False
    
    return True

def validate_query(query: 'RAGQuery') -> bool:
    """Validate query data"""
    if not query.query or not query.query.strip():
        return False
    
    if not query.agent_type or query.agent_type not in ['atendimento_geral', 'open_finance', 'golpe_med', 'criacao_conta']:
        return False
    
    if query.top_k <= 0 or query.top_k > 50:
        return False
    
    return True

def calculate_corpus_hash(file_paths: List[str]) -> str:
    """Calculate hash for corpus files"""
    import hashlib
    
    hash_md5 = hashlib.md5()
    
    for file_path in sorted(file_paths):
        try:
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception as e:
            logger.error(f"Failed to read file {file_path} for hashing: {e}")
            continue
    
    return hash_md5.hexdigest()

def calculate_config_hash(config: RAGConfig) -> str:
    """Calculate hash for configuration"""
    import hashlib
    
    config_str = json.dumps({
        'chunk_size': config.chunk_size,
        'chunk_overlap': config.chunk_overlap,
        'default_top_k': config.default_top_k,
        'similarity_threshold': config.similarity_threshold
    }, sort_keys=True)
    
    return hashlib.md5(config_str.encode()).hexdigest()
