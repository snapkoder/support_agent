"""
RAG Facade - Single Source of Truth for RAG components
Provides centralized adapter selection and compatibility checking
"""

import os
import logging
import json
import time
import socket
import psutil
from typing import Dict, Any, Optional, List, Tuple
import asyncio
from pathlib import Path

from .models import RAGConfig, EmbeddingSignature, IndexMetadata, IndexIncompatibleError, EmptyContextReason, EmptyContextDetails
from .adapters import OpenAIEmbeddingsAdapter, LocalEmbeddingsAdapter

logger = logging.getLogger(__name__)

class RAGFacade:
    """
    Facade for RAG operations with centralized adapter selection
    """
    
    # Cache for metadata to avoid repeated IO
    _metadata_cache: Dict[str, Any] = {}
    _metadata_cache_timestamp: float = 0
    _metadata_cache_ttl: float = 300  # 5 minutes TTL
    
    def __init__(self, rag_service: Optional['RAGService'] = None):
        """Initialize RAG facade"""
        self.rag_service = rag_service
        self.logger = logging.getLogger(f"{__name__}.RAGFacade")
    
    @staticmethod
    def build_embeddings_adapter_from_config() -> 'EmbeddingsPort':
        """
        Build embeddings adapter from environment configuration.
        
        This is the single source of truth for embedding adapter selection.
        Used by both indexer and retriever to ensure consistency.
        
        Returns:
            Configured embeddings adapter
        """
        # Check for OpenAI embeddings preference
        openai_enabled = os.getenv('OPENAI_EMBEDDINGS_ENABLED', 'true').lower() == 'true'
        openai_key = os.getenv('OPENAI_API_KEY', '')
        
        if openai_enabled and openai_key and not openai_key.startswith('YOUR_'):
            try:
                # Use OpenAI embeddings
                model = os.getenv('OPENAI_EMBEDDINGS_MODEL', 'text-embedding-3-small')
                adapter = OpenAIEmbeddingsAdapter(api_key=openai_key, model_name=model)
                
                # Log selection
                signature = adapter.get_signature()
                logger.info(f"Embedding adapter selected: {signature}")
                
                # Log structured event
                log_data = {
                    'event': 'embedding_adapter_selected',
                    'provider': signature.provider,
                    'model_name': signature.model_name,
                    'dimensions': signature.dimensions,
                    'normalize': signature.normalize,
                    'stable_hash': signature.stable_hash(),
                    'timestamp': time.time()
                }
                logger.info(json.dumps(log_data, ensure_ascii=False))
                
                return adapter
                
            except Exception as e:
                logger.warning(f"Failed to create OpenAI embeddings adapter: {e}")
                logger.info("Falling back to local embeddings")
        
        # Default to local embeddings
        vector_size = int(os.getenv('LOCAL_EMBEDDINGS_SIZE', '384'))
        adapter = LocalEmbeddingsAdapter(vector_size=vector_size)
        
        # Log selection
        signature = adapter.get_signature()
        logger.info(f"Embedding adapter selected (fallback): {signature}")
        
        # Log structured event
        log_data = {
            'event': 'embedding_adapter_selected',
            'provider': signature.provider,
            'model_name': signature.model_name,
            'dimensions': signature.dimensions,
            'normalize': signature.normalize,
            'stable_hash': signature.stable_hash(),
            'timestamp': time.time()
        }
        logger.info(json.dumps(log_data, ensure_ascii=False))
        
        return adapter
    
    @staticmethod
    def _load_metadata_cached(index_dir: str) -> Optional[Dict[str, Any]]:
        """Load index metadata with caching to avoid repeated IO"""
        metadata_path = Path(index_dir) / "index_metadata.json"
        current_time = time.time()
        
        # Check cache validity
        if (RAGFacade._metadata_cache and 
            current_time - RAGFacade._metadata_cache_timestamp < RAGFacade._metadata_cache_ttl):
            return RAGFacade._metadata_cache
        
        try:
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update cache
            RAGFacade._metadata_cache = data
            RAGFacade._metadata_cache_timestamp = current_time
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            return None
    
    @staticmethod
    def _invalidate_metadata_cache():
        """Invalidate metadata cache (e.g., after rebuild)"""
        RAGFacade._metadata_cache = {}
        RAGFacade._metadata_cache_timestamp = 0
    
    @staticmethod
    def check_index_compatibility(embeddings_port: 'EmbeddingsPort', index_dir: str = "./var/rag_index") -> Tuple[bool, Optional[EmptyContextDetails]]:
        """
        Check if current embeddings adapter is compatible with existing index.
        
        Args:
            embeddings_port: Current embeddings adapter
            index_dir: Directory containing index metadata
            
        Returns:
            Tuple of (is_compatible, empty_context_details if incompatible)
        """
        try:
            # Load index metadata with caching
            data = RAGFacade._load_metadata_cached(index_dir)
            if not data:
                logger.error("Index metadata file not found - assuming incompatible")
                # Log structured event
                log_data = {
                    'event': 'rag_metadata_missing',
                    'action_taken': 'blocked_query',
                    'metadata_path': str(Path(index_dir) / "index_metadata.json"),
                    'timestamp': time.time()
                }
                logger.error(json.dumps(log_data, ensure_ascii=False))
                
                details = EmptyContextDetails(
                    reason=EmptyContextReason.METADATA_MISSING,
                    index_path=str(Path(index_dir) / "index_metadata.json"),
                    timestamp=time.time()
                )
                return False, details
            
            # Validate metadata structure
            if not RAGFacade._validate_metadata_structure(data):
                logger.error("Invalid index metadata structure - assuming incompatible")
                # Log structured event
                log_data = {
                    'event': 'rag_metadata_invalid',
                    'action_taken': 'blocked_query',
                    'metadata_path': str(Path(index_dir) / "index_metadata.json"),
                    'validation_errors': 'structure_validation_failed',
                    'timestamp': time.time()
                }
                logger.error(json.dumps(log_data, ensure_ascii=False))
                
                details = EmptyContextDetails(
                    reason=EmptyContextReason.METADATA_CORRUPT,
                    index_path=str(Path(index_dir) / "index_metadata.json"),
                    error_type='structure_validation_failed',
                    timestamp=time.time()
                )
                return False, details
            
            try:
                index_metadata = IndexMetadata.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to parse index metadata: {e} - assuming incompatible")
                # Log structured event
                log_data = {
                    'event': 'rag_metadata_parse_error',
                    'action_taken': 'blocked_query',
                    'metadata_path': str(metadata_path),
                    'parse_error': str(e),
                    'timestamp': time.time()
                }
                logger.error(json.dumps(log_data, ensure_ascii=False))
                
                details = EmptyContextDetails(
                    reason=EmptyContextReason.METADATA_CORRUPT,
                    index_path=str(metadata_path),
                    error_type='parse_error',
                    timestamp=time.time()
                )
                return False, details
            
            if not index_metadata.embedding_signature:
                logger.error("No embedding signature in index metadata - assuming incompatible")
                # Log structured event
                log_data = {
                    'event': 'rag_signature_missing',
                    'action_taken': 'blocked_query',
                    'metadata_path': str(metadata_path),
                    'timestamp': time.time()
                }
                logger.error(json.dumps(log_data, ensure_ascii=False))
                
                details = EmptyContextDetails(
                    reason=EmptyContextReason.SIGNATURE_INCOMPLETE,
                    index_path=str(metadata_path),
                    timestamp=time.time()
                )
                return False, details
            
            # Get current signature
            current_signature = embeddings_port.get_signature()
            index_signature = EmbeddingSignature.from_dict(index_metadata.embedding_signature)
            
            # Check compatibility
            is_compatible = current_signature.is_compatible_with(index_signature)
            
            if not is_compatible:
                logger.error(f"Embedding signature mismatch detected!")
                logger.error(f"Index signature: {index_signature}")
                logger.error(f"Current signature: {current_signature}")
                
                # Log structured incompatibility event
                log_data = {
                    'event': 'rag_index_incompatible',
                    'action_taken': 'blocked_query',
                    'expected_signature': index_signature.to_dict(),
                    'current_signature': current_signature.to_dict(),
                    'index_path': str(metadata_path),
                    'timestamp': time.time()
                }
                logger.error(json.dumps(log_data, ensure_ascii=False))
                
                details = EmptyContextDetails(
                    reason=EmptyContextReason.INCOMPATIBLE_SIGNATURE,
                    index_path=str(metadata_path),
                    expected_signature=index_signature.to_dict(),
                    current_signature=current_signature.to_dict(),
                    timestamp=time.time()
                )
                return False, details
            
            logger.info("Embedding signature compatibility check passed")
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to check index compatibility: {e}")
            # Log structured event for unexpected errors
            log_data = {
                'event': 'rag_compatibility_check_error',
                'action_taken': 'blocked_query',
                'error': str(e),
                'index_dir': index_dir,
                'timestamp': time.time()
            }
            logger.error(json.dumps(log_data, ensure_ascii=False))
            
            details = EmptyContextDetails(
                reason=EmptyContextReason.UNEXPECTED_ERROR,
                error_type='compatibility_check_error',
                timestamp=time.time()
            )
            return False, details
    
    @staticmethod
    def _validate_metadata_structure(data: Dict[str, Any]) -> bool:
        """Validate index metadata structure with version support"""
        try:
            # Check required top-level fields
            required_fields = ['corpus_hash', 'total_chunks', 'total_documents', 'created_at', 'updated_at']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required metadata field: {field}")
                    return False
            
            # Get metadata version (default to 1 for compatibility)
            metadata_version = data.get('metadata_version', 1)
            
            # Validate embedding signature based on version
            if 'embedding_signature' in data:
                signature = data['embedding_signature']
                
                if metadata_version == 1:
                    # v1: basic validation
                    required_signature_fields = ['provider', 'model_name', 'dimensions']
                elif metadata_version == 2:
                    # v2: full validation
                    required_signature_fields = ['provider', 'model_name', 'dimensions', 'normalize']
                else:
                    # Unknown version - assume incompatible
                    logger.error(f"Unknown metadata version: {metadata_version} - assuming incompatible")
                    return False
                
                for field in required_signature_fields:
                    if field not in signature:
                        logger.error(f"Missing required signature field for v{metadata_version}: {field}")
                        return False
            
            logger.info(f"Metadata validation passed for version {metadata_version}")
            return True
            
        except Exception as e:
            logger.error(f"Metadata validation failed: {e}")
            return False
    
    @staticmethod
    def trigger_auto_rebuild_if_enabled(embeddings_adapter: 'EmbeddingsPort', index_dir: str = "./var/rag_index") -> bool:
        """
        Trigger automatic rebuild if enabled and compatible.
        
        Args:
            embeddings_adapter: Current embeddings adapter
            index_dir: Directory containing index
            
        Returns:
            True if rebuild was triggered, False otherwise
        """
        auto_rebuild = os.getenv('RAG_AUTO_REBUILD_ON_MISMATCH', 'true').lower() == 'true'
        
        if not auto_rebuild:
            logger.info("Auto-rebuild disabled - manual rebuild required")
            return False
        
        lock_file = Path(index_dir) / ".rebuild.lock"
        
        # Clean up stale locks (older than 1 hour)
        RAGFacade._cleanup_stale_locks(lock_file)
        
        # Check for existing rebuild lock
        if lock_file.exists():
            try:
                with open(lock_file, 'r') as f:
                    lock_data = json.load(f)
                logger.info(f"Rebuild already in progress (started at {lock_data.get('timestamp')})")
                return False
            except Exception:
                logger.warning("Invalid lock file - proceeding with rebuild")
        
        # Create rebuild lock
        try:
            signature = embeddings_adapter.get_signature()
            lock_data = RAGFacade._create_structured_lock(lock_file, signature)
            
            # Trigger rebuild (this would be called by the indexer)
            logger.info("Auto-rebuild triggered - run build_rag_index.py to complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create rebuild lock: {e}")
            return False
    
    @staticmethod
    def _cleanup_stale_locks(lock_file: Path, max_age_hours: int = 1) -> None:
        """Clean up stale rebuild locks with PID verification"""
        if not lock_file.exists():
            return
        
        try:
            with open(lock_file, 'r') as f:
                lock_data = json.load(f)
            
            lock_age = time.time() - lock_data.get('created_at', 0)
            
            # Check if lock is old enough to consider stale
            if lock_age <= max_age_hours * 3600:
                return  # Lock is still fresh
            
            # Verify PID and host before cleanup
            pid = lock_data.get('pid')
            host = lock_data.get('host')
            
            should_cleanup = False
            
            if pid is not None:
                try:
                    # Check if PID exists
                    process = psutil.Process(pid)
                    if process.is_running():
                        # Check if it's the same host
                        current_host = socket.gethostname()
                        if host == current_host:
                            logger.info(f"Lock PID {pid} still running on host {host} - keeping lock")
                            return
                        else:
                            logger.info(f"Lock PID {pid} running on different host ({host} vs {current_host}) - cleaning up")
                            should_cleanup = True
                    else:
                        logger.info(f"Lock PID {pid} no longer running - cleaning up")
                        should_cleanup = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.info(f"Lock PID {pid} not found or inaccessible - cleaning up")
                    should_cleanup = True
            else:
                logger.warning("Lock has no PID - cleaning up")
                should_cleanup = True
            
            if should_cleanup:
                lock_file.unlink()
                logger.warning(f"Removed stale rebuild lock: {lock_file} (age: {lock_age/3600:.1f}h)")
                
        except Exception as e:
            logger.error(f"Failed to cleanup stale lock {lock_file}: {e}")
    
    @staticmethod
    def _create_structured_lock(lock_file: Path, signature: EmbeddingSignature) -> Dict[str, Any]:
        """Create structured lock with process information"""
        try:
            lock_data = {
                'pid': os.getpid(),
                'created_at': time.time(),
                'host': socket.gethostname(),
                'signature': signature.to_dict(),
                'status': 'in_progress'
            }
            
            with open(lock_file, 'w') as f:
                json.dump(lock_data, f, indent=2)
            
            logger.info(f"Created structured rebuild lock: {lock_file}")
            return lock_data
            
        except Exception as e:
            logger.error(f"Failed to create structured lock: {e}")
            raise
    
    async def search(self, query: str, agent_type: str = "atendimento_geral", 
                    top_k: int = 8, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for relevant documents (compatibility method)
        
        Args:
            query: Search query
            agent_type: Type of agent
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            Dictionary with search results in legacy format
        """
        try:
            # Use new RAG service
            result = await self.rag_service.process_query(
                query=query,
                agent_type=agent_type,
                requires_rag=True,  # Force RAG for explicit search
                top_k=top_k,
                filters=filters
            )
            
            # Convert to legacy format
            documents = []
            for chunk in result.chunks:
                doc = {
                    'content': chunk.chunk.content,
                    'metadata': {
                        'source_file': chunk.chunk.source_file,
                        'section_title': chunk.chunk.section_title,
                        'domain': chunk.chunk.domain,
                        'agent_type': chunk.chunk.agent_type,
                        'chunk_id': chunk.chunk.chunk_id,
                        'breadcrumb': chunk.chunk.breadcrumb
                    },
                    'score': chunk.score,
                    'doc_id': chunk.chunk.chunk_id,
                    'chunk_id': chunk.chunk.chunk_id,
                    'source': chunk.chunk.source_file
                }
                documents.append(doc)
            
            return {
                'documents': documents,
                'query': query,
                'confidence': result.metadata.get('avg_score', 0.0),
                'processing_time': result.rag_latency_ms / 1000.0,  # Convert to seconds
                'source': 'knowledge_base',
                'rag_used': result.rag_used,
                'rag_latency_ms': result.rag_latency_ms,
                'rag_hits': len(result.chunks)
            }
            
        except Exception as e:
            self.logger.error(f"RAG search failed: {e}")
            return {
                'documents': [],
                'query': query,
                'confidence': 0.0,
                'processing_time': 0.0,
                'source': 'knowledge_base',
                'error': str(e)
            }
    
    async def health_check(self) -> bool:
        """Check RAG system health"""
        try:
            health_status = await self.rag_service.health_check()
            return health_status.get('overall', False)
        except Exception as e:
            self.logger.error(f"RAG health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get RAG metrics"""
        try:
            metrics = self.rag_service.get_metrics()
            return {
                'total_queries': metrics.total_queries,
                'rag_used_count': metrics.rag_used_count,
                'rag_usage_rate': metrics.rag_usage_rate,
                'avg_rag_latency_ms': metrics.avg_rag_latency_ms,
                'avg_chunks_retrieved': metrics.avg_chunks_retrieved,
                'cache_hits': metrics.cache_hits,
                'cache_misses': metrics.cache_misses,
                'cache_hit_rate': metrics.cache_hit_rate
            }
        except Exception as e:
            self.logger.error(f"Failed to get RAG metrics: {e}")
            return {}
