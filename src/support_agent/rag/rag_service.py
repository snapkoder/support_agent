"""
RAG Service - Use Case layer for RAG operations
Orchestrates RAG workflow using only ports (no concrete dependencies)
"""

import time
import logging
from typing import Dict, Any, Optional, List
import json

from .ports import (
    EmbeddingsPort, VectorStorePort, RetrieverPort, KnowledgeBasePort,
    DocumentChunk, RetrievedChunk, RAGQuery, RAGResult
)
from .adapters.external_kb_stub_adapter import ExternalKBStubAdapter
from .models import RAGConfig, RAGMetrics, validate_query
from .rag_facade import RAGFacade

logger = logging.getLogger(__name__)

class RAGService:
    """
    RAG Service - Main use case for RAG operations
    
    This service orchestrates the RAG workflow using only port interfaces.
    It decides when to use RAG, executes retrieval, and returns structured results.
    """
    
    def __init__(
        self,
        embeddings_port: EmbeddingsPort,
        vector_store_port: VectorStorePort,
        retriever_port: RetrieverPort,
        knowledge_base_port: KnowledgeBasePort,
        config: Optional[RAGConfig] = None
    ):
        """Initialize RAG service with required ports"""
        self.embeddings_port = embeddings_port
        self.vector_store_port = vector_store_port
        self.retriever_port = retriever_port
        self.knowledge_base_port = knowledge_base_port
        self.config = config or RAGConfig.from_env()
        
        # External KB adapter (optional)
        self.external_kb_adapter = ExternalKBStubAdapter()
        
        # Metrics tracking
        self.metrics = RAGMetrics()
        
        # Cache for query results (if enabled)
        self._query_cache: Dict[str, RAGResult] = {}
        
        self.logger = logging.getLogger(f"{__name__}.RAGService")
        self.logger.info("RAG Service initialized with ports and external KB")
    
    async def should_use_rag(self, query: str, agent_type: str, requires_rag: bool = False) -> bool:
        """
        Decide if RAG should be used for this query.
        
        Args:
            query: User query
            agent_type: Type of agent handling the query
            requires_rag: Explicit flag from dataset
            
        Returns:
            True if RAG should be used
        """
        # RAG Always On - Always return True
        if self.config.rag_always_on:
            self.logger.info(f"RAG always enabled for agent {agent_type}")
            return True
        
        # Legacy logic (kept for compatibility but never used)
        # 1. Explicit flag takes precedence
        if requires_rag:
            self.logger.info(f"RAG triggered by explicit flag for agent {agent_type}")
            return True
        
        # 2. Check for 🔍 symbol in query
        if '🔍' in query:
            self.logger.info(f"RAG triggered by 🔍 symbol for agent {agent_type}")
            return True
        
        # 3. Heuristic detection by agent type
        rag_keywords = {
            "atendimento_geral": [
                "limite", "horário", "funciona", "disponível", "taxa", "tarifa",
                "custo", "preço", "valor", "quanto custa", "quanto tempo",
                "como funciona", "endereço", "contato", "telefone", "whatsapp"
            ],
            "open_finance": [
                "conectar", "banco", "open finance", "compatível", "instituição",
                "outro banco", "banco digital", "conta bancária", "extrato",
                "saldo", "pagar boleto", "transferir", "pix"
            ],
            "golpe_med": [
                "med", "devolução", "prazo", "banco central", "bo", "bolsão",
                "golpe", "fraude", "estorno", "reembolso", "dinheiro devolvido"
            ],
            "criacao_conta": [
                "documento", "aprovação", "validação", "cadastro", "conta",
                "abrir conta", "criar conta", "cpf", "cnh", "rg", "selfie"
            ]
        }
        
        keywords = rag_keywords.get(agent_type, [])
        query_lower = query.lower()
        
        for keyword in keywords:
            if keyword.lower() in query_lower:
                self.logger.info(f"RAG triggered by keyword '{keyword}' for agent {agent_type}")
                return True
        
        # 4. Default: don't use RAG for generic queries
        self.logger.debug(f"RAG not triggered for agent {agent_type}: {query[:50]}...")
        return False
    
    async def process_query(
        self, 
        query: str, 
        agent_type: str, 
        requires_rag: bool = False,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        """
        Process a RAG query end-to-end.
        
        Args:
            query: User query
            agent_type: Type of agent
            requires_rag: Explicit RAG flag
            top_k: Number of chunks to retrieve
            filters: Metadata filters
            
        Returns:
            RAGResult with retrieved chunks and metadata
        """
        start_time = time.time()
        
        # Create RAG query object
        # Use configured top_k for RAG Always On
        effective_top_k = top_k or self.config.rag_top_k
        
        # Apply intelligent filtering based on agent_type
        effective_filters = filters or {}
        if agent_type and agent_type != 'unknown':
            # Map agent_type to domain filter - less restrictive
            domain_mapping = {
                'atendimento_geral': 'atendimento_geral',
                'open_finance': 'open_finance', 
                'golpe_med': 'golpe_med',
                'criacao_conta': 'criacao_conta'
            }
            # Only apply filter if we have chunks with that domain
            # For now, don't apply filters to ensure retrieval works
            # effective_filters['domain'] = domain_mapping.get(agent_type, agent_type)
            # effective_filters['agent_type'] = agent_type
        
        rag_query = RAGQuery(
            query=query,
            agent_type=agent_type,
            top_k=effective_top_k,
            filters=effective_filters,
            requires_rag=requires_rag
        )
        
        # Validate query
        if not validate_query(rag_query):
            raise RetrievalError(f"Invalid RAG query: {rag_query}")
        
        # Decide if RAG should be used (always true with RAG Always On)
        rag_used = await self.should_use_rag(query, agent_type, requires_rag)
        
        # RAG Always On - No conditional branch, always proceed with retrieval
        self.logger.info(f"RAG always enabled for query: {query[:50]}...")
        
        # Check cache first (if enabled)
        cache_key = self._get_cache_key(rag_query)
        if self.config.cache_enabled and cache_key in self._query_cache:
            self.metrics.cache_hits += 1
            cached_result = self._query_cache[cache_key]
            self.logger.info(f"RAG cache hit for query: {query[:50]}...")
            return cached_result
        
        self.metrics.cache_misses += 1
        
        try:
            # Local-First Retrieval
            local_chunks = await self._retrieve_local_first(
                query=query,
                top_k=rag_query.top_k,
                filters=rag_query.filters,
                agent_type=agent_type
            )
            
            # Protection against empty context
            if not local_chunks:
                self.logger.warning(f"No chunks retrieved for query: {query[:50]}... - Continuing with minimal context")
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            self.metrics.update_query(True, processing_time, len(local_chunks))
            
            # Get empty context details from retriever if available
            empty_context_reason = None
            empty_context_details = None
            
            if hasattr(self.retriever_port, 'get_empty_context_details'):
                empty_context_details = self.retriever_port.get_empty_context_details()
                if empty_context_details:
                    empty_context_reason = empty_context_details.reason
            
            # Create result with all fields populated
            avg_score = sum(chunk.score for chunk in local_chunks) / len(local_chunks) if local_chunks else 0.0
            sources = [chunk.chunk.source_file for chunk in local_chunks]
            domains = list(set(chunk.chunk.domain for chunk in local_chunks))
            
            result = RAGResult(
                chunks=local_chunks,
                rag_used=True,
                rag_latency_ms=processing_time,
                rag_hits=len(local_chunks),
                rag_sources=sources,
                rag_domains=domains,
                avg_score=avg_score,
                latency_ms=processing_time,
                rag_always_on=self.config.rag_always_on,
                local_first=self.config.local_first,
                local_hits=len(local_chunks),
                external_hits=0,
                filters_applied=rag_query.filters or {},
                retrieval_latency_ms=processing_time,
                metadata={
                    'query': query,
                    'agent_type': agent_type,
                    'top_k': rag_query.top_k,
                    'filters': rag_query.filters or {},
                    'retrieval_count': len(local_chunks),
                    'avg_score': avg_score,
                    'sources': sources,
                    'domains': domains,
                    'retrieved_chunk_ids': [chunk.chunk.chunk_id for chunk in local_chunks],
                    'retrieved_breadcrumbs': [chunk.chunk.breadcrumb for chunk in local_chunks],
                    'rag_always_on': self.config.rag_always_on,
                    'local_first': self.config.local_first,
                    'local_hits': len(local_chunks),
                    'external_hits': 0
                },
                empty_context_reason=empty_context_reason,
                empty_context_details=empty_context_details
            )
            
            # Cache result (if enabled)
            if self.config.cache_enabled:
                self._cache_result(cache_key, result)
            
            # Log structured event
            self._log_rag_result(result)
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"RAG processing failed: {e}")
            
            # Return empty result on error
            return RAGResult(
                chunks=[],
                rag_used=False,
                rag_latency_ms=processing_time,
                metadata={
                    'query': query,
                    'agent_type': agent_type,
                    'error': str(e),
                    'reason': 'processing_error'
                }
            )
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all RAG components.
        
        Returns:
            Dictionary with health status of each component
        """
        health_status = {}
        
        try:
            health_status['embeddings'] = await self.embeddings_port.health_check()
        except Exception as e:
            self.logger.error(f"Embeddings health check failed: {e}")
            health_status['embeddings'] = False
        
        try:
            health_status['vector_store'] = await self.vector_store_port.health_check()
        except Exception as e:
            self.logger.error(f"Vector store health check failed: {e}")
            health_status['vector_store'] = False
        
        try:
            health_status['retriever'] = await self.retriever_port.health_check()
        except Exception as e:
            self.logger.error(f"Retriever health check failed: {e}")
            health_status['retriever'] = False
        
        try:
            health_status['knowledge_base'] = await self.knowledge_base_port.health_check()
        except Exception as e:
            self.logger.error(f"Knowledge base health check failed: {e}")
            health_status['knowledge_base'] = False
        
        # Overall health
        health_status['overall'] = all(health_status.values())
        
        return health_status
    
    def get_metrics(self) -> RAGMetrics:
        """Get current RAG metrics"""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset RAG metrics"""
        self.metrics = RAGMetrics()
        self.logger.info("RAG metrics reset")
    
    def clear_cache(self) -> None:
        """Clear query cache"""
        self._query_cache.clear()
        self.logger.info("RAG query cache cleared")
    
    async def _retrieve_local_first(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        agent_type: str
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks using local-first strategy with optional external fallback.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            agent_type: Type of agent
            
        Returns:
            List of retrieved chunks
        """
        # Stage 1: Local retrieval
        local_chunks = await self.retriever_port.retrieve(
            query=query,
            top_k=top_k,
            filters=filters
        )
        
        # Check if we have enough local hits
        if len(local_chunks) >= self.config.min_local_hits:
            self.logger.info(f"Local retrieval sufficient: {len(local_chunks)} >= {self.config.min_local_hits}")
            return local_chunks
        
        # Stage 2: External fallback (if enabled and no local hits)
        if self.config.external_kb_enabled and len(local_chunks) == 0:
            self.logger.info("No local hits, trying external KB fallback")
            
            try:
                external_chunks = await self.external_kb_adapter.query_external_knowledge(
                    query=query,
                    filters=filters,
                    top_k=top_k
                )
                
                if external_chunks:
                    self.logger.info(f"External KB returned {len(external_chunks)} chunks")
                    
                    # Convert external chunks to RetrievedChunk format
                    external_retrieved = []
                    for chunk in external_chunks:
                        retrieved = RetrievedChunk(
                            chunk=chunk,
                            score=0.5,  # Default score for external chunks
                            retrieval_method='external_stub'
                        )
                        external_retrieved.append(retrieved)
                    
                    return external_retrieved
                else:
                    self.logger.info("External KB returned no chunks")
                    
            except Exception as e:
                self.logger.error(f"External KB fallback failed: {e}")
        
        # Return local chunks (even if empty)
        return local_chunks
    
    def _get_cache_key(self, query: RAGQuery) -> str:
        """Generate cache key for query"""
        import hashlib
        
        cache_data = {
            'query': query.query,
            'agent_type': query.agent_type,
            'top_k': query.top_k,
            'filters': query.filters or {}
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, result: RAGResult) -> None:
        """Cache a query result"""
        if len(self._query_cache) >= self.config.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = result
    
    def _log_rag_result(self, result: RAGResult) -> None:
        """Log structured RAG result with mandatory fields and empty context reasoning"""
        import uuid
        
        request_id = f"rag_req_{uuid.uuid4().hex[:8]}"
        trace_id = f"rag_trace_{uuid.uuid4().hex[:8]}"
        
        # Enhanced log data with empty context reasoning
        log_data = {
            'event': 'rag_retrieval_end',
            'request_id': request_id,
            'trace_id': trace_id,
            'agent_selected': result.metadata.get('agent_type', 'unknown'),
            'rag_used': result.rag_used,
            'rag_top_k': result.metadata.get('top_k', 0),
            'retrieved_chunk_ids': result.metadata.get('retrieved_chunk_ids', []),
            'retrieved_breadcrumbs': result.metadata.get('retrieved_breadcrumbs', []),
            'rag_hits': result.rag_hits,
            'rag_sources': result.metadata.get('sources', []),
            'rag_domains': result.metadata.get('domains', []),
            'avg_score': result.metadata.get('avg_score', 0.0),
            'query_preview': result.metadata.get('query', '')[:50],
            'agent_type': result.metadata.get('agent_type', 'unknown'),
            'model_used': 'unknown',  # Will be updated by integration layer
            'latency_ms': result.rag_latency_ms,
            'timestamp': result.timestamp,
            'rag_always_on': result.metadata.get('rag_always_on', False),
            'local_first': result.metadata.get('local_first', False),
            'local_hits': result.metadata.get('local_hits', 0),
            'external_hits': result.metadata.get('external_hits', 0),
            'filters_applied': result.metadata.get('filters', {}),
            'retrieval_latency_ms': result.metadata.get('retrieval_latency_ms', 0.0)
        }
        
        # Add empty context reasoning if applicable
        if result.empty_context_reason and result.rag_hits == 0:
            log_data.update({
                'empty_context_reason': result.empty_context_reason.value,
                'empty_context_details': result.empty_context_details.__dict__ if result.empty_context_details else None
            })
        
        self.logger.info(json.dumps(log_data, ensure_ascii=False))
