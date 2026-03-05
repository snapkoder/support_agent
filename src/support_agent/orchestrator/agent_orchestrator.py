"""
AGENT ORCHESTRATOR - JOTA SUPPORT AGENT
Orquestrador inteligente de agentes para escala enterprise
"""

import asyncio
import logging
import re
import time
import uuid
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import json
import hashlib
import unicodedata

from support_agent.llm.llm_manager import get_llm_manager
from support_agent.prompts.prompt_manager import get_prompt_manager
from support_agent.memory.simple_memory_adapter import get_simple_memory_adapter
from support_agent.memory import get_memory_orchestrator, MemoryUpdate
from support_agent.policy.policy_engine import JotaPolicyEngine
from support_agent.security.redact import get_secure_logger
from support_agent.cache.cache_store import get_classify_cache, get_decision_cache, get_rag_cache
from support_agent.config.settings import _get_env_fallback

logger = logging.getLogger(__name__)

# Portuguese stopwords — allocated once at module level
_PT_STOPWORDS = frozenset({
    'o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na',
    'nos', 'nas', 'por', 'para', 'com', 'sem', 'como', 'onde', 'quando', 'que',
    'qual', 'é', 'são', 'foi', 'tem', 'tenho', 'ter', 'pode', 'poder', 'ser',
    'estar', 'está', 'estou', 'vai', 'ir', 'faz', 'fazer', 'já', 'também',
    'mais', 'muito', 'muita', 'bem', 'meu', 'minha', 'seu', 'sua', 'nosso',
    'nossa', 'este', 'esta', 'isto', 'isso', 'aquilo', 'um', 'uma', 'uns', 'umas',
})


# ============================================================================
# RAG CONTRACT — data classes and system (previously in core/rag_system.py)
# ============================================================================

@dataclass
class RAGDocument:
    """Documento para o sistema RAG"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    chunk_id: str = ""
    source: str = ""
    embedding: Optional[List[float]] = None
    score: float = 0.0


@dataclass
class RAGQuery:
    """Consulta RAG estruturada"""
    query: str
    agent_type: str
    user_context: Dict[str, Any]
    top_k: int = field(default_factory=lambda: int(_get_env_fallback("RETRIEVAL_TOP_K", "8")))
    filters: Dict[str, Any] = None


@dataclass
class RAGResult:
    """Resultado da consulta RAG"""
    documents: List[RAGDocument]
    query: RAGQuery
    confidence: float
    processing_time: float
    source: str = "knowledge_base"


class JotaRAGSystem:
    """
    RAG system that delegates all retrieval to core.rag.RAGService via ports/adapters.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.JotaRAGSystem")
        self._rag_service = None
        self._embeddings_adapter = None
        self._vector_store = None
        self.metrics = {
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.logger.info("JotaRAGSystem initialized")

    async def initialize(self) -> bool:
        """Inicializa o sistema RAG usando core/rag/ adapters."""
        try:
            from support_agent.rag.rag_facade import RAGFacade
            from support_agent.rag.adapters import (
                InMemoryVectorStoreAdapter,
                RetrieverAdapter,
                KnowledgeBaseAdapter,
            )
            from support_agent.rag.rag_service import RAGService
            from support_agent.rag.models import RAGConfig, DocumentChunk

            self._embeddings_adapter = RAGFacade.build_embeddings_adapter_from_config()
            sig = self._embeddings_adapter.get_signature()
            self.logger.info(f"[RAG_INIT] Embedding adapter: {sig}")

            self._vector_store = InMemoryVectorStoreAdapter()

            base_path = Path(__file__).resolve().parents[3]
            kb_dir = str(base_path / "assets" / "knowledge_base")
            kb_adapter = KnowledgeBaseAdapter(kb_dir=kb_dir)

            doc_files = await kb_adapter.list_documents()
            if not doc_files:
                self.logger.warning("[RAG_INIT] No KB documents found")
                return True

            raw_contents = await kb_adapter.load_documents(doc_files)
            self.logger.info(f"[RAG_INIT] Loaded {len(raw_contents)} KB files")

            all_chunks: List[DocumentChunk] = []
            for file_idx, (file_path, content) in enumerate(zip(doc_files, raw_contents)):
                sections = self._parse_markdown(content)
                for sec_idx, (title, sec_content) in enumerate(sections):
                    agent_type = self._infer_agent_type(title, sec_content)
                    clean_title = title.replace('#', '').strip()
                    chunk = DocumentChunk(
                        chunk_id=f"kb_section_{file_idx}_{sec_idx}_chunk_0",
                        content=f"{title}\n{sec_content}" if title else sec_content,
                        source_file=str(file_path),
                        section_title=clean_title,
                        domain=agent_type,
                        agent_type=agent_type,
                        breadcrumb=title,
                        chunk_index=sec_idx,
                        total_chunks=len(sections),
                    )
                    all_chunks.append(chunk)

            self.logger.info(f"[RAG_INIT] Generated {len(all_chunks)} chunks")

            texts = [c.content for c in all_chunks]
            embed_result = self._embeddings_adapter.embed(texts)
            embeddings = (await embed_result) if asyncio.iscoroutine(embed_result) else embed_result
            self.logger.info(f"[RAG_INIT] Embeddings generated ({len(embeddings)})")

            await self._vector_store.upsert(all_chunks, embeddings)
            self.logger.info(f"[RAG_INIT] Chunks stored in vector store")

            retriever = RetrieverAdapter(
                embeddings_port=self._embeddings_adapter,
                vector_store_port=self._vector_store,
                skip_compatibility_check=True,
            )
            self._rag_service = RAGService(
                embeddings_port=self._embeddings_adapter,
                vector_store_port=self._vector_store,
                retriever_port=retriever,
                knowledge_base_port=kb_adapter,
            )
            self.logger.info(f"✅ RAG System initialized via core/rag/ — {len(all_chunks)} chunks indexed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error initializing RAG system: {e}")
            self.logger.error(traceback.format_exc())
            return False

    async def query(self, rag_query: RAGQuery) -> RAGResult:
        """Executa consulta RAG delegando para core/rag/RAGService."""
        start_time = datetime.now()

        try:
            self.logger.info(
                f"[RAG_IMPL_PROOF] event=rag_impl_selected impl=new_core_rag "
                f"file=core/rag/rag_service.py method=RAGService.process_query"
            )
            self.logger.info(
                f"🔍 RAG Query - query='{rag_query.query}', "
                f"agent={rag_query.agent_type}, top_k={rag_query.top_k}"
            )

            new_result = await self._rag_service.process_query(
                query=rag_query.query,
                agent_type=rag_query.agent_type,
                requires_rag=True,
                top_k=rag_query.top_k,
                filters=rag_query.filters,
            )

            legacy_docs = []
            for rc in new_result.chunks:
                doc = RAGDocument(
                    content=rc.chunk.content,
                    metadata={
                        "title": rc.chunk.section_title,
                        "h_path": rc.chunk.breadcrumb,
                        "section_title": rc.chunk.section_title,
                        "agent_type": rc.chunk.agent_type,
                        "domain": rc.chunk.domain,
                        "source": rc.chunk.source_file,
                        "chunk_id": rc.chunk.chunk_id,
                        "breadcrumb": rc.chunk.breadcrumb,
                        "score_cosine": rc.score,
                        "score_lexical": 0.0,
                        "coverage": 0.0,
                        "final_score": rc.score,
                    },
                    doc_id=rc.chunk.chunk_id,
                    chunk_id=rc.chunk.chunk_id,
                    source=rc.chunk.source_file,
                    score=rc.score,
                )
                legacy_docs.append(doc)

            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics["queries_processed"] += 1
            self.metrics["total_processing_time"] += processing_time

            confidence = max((d.score for d in legacy_docs), default=0.0)

            result = RAGResult(
                documents=legacy_docs,
                query=rag_query,
                confidence=min(confidence, 1.0),
                processing_time=processing_time,
                source="new_core_rag",
            )
            self.logger.info(
                f"RAG query processed in {processing_time:.3f}s — "
                f"{len(legacy_docs)} results via core/rag/"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error processing RAG query: {e}")
            self.logger.error(traceback.format_exc())
            return RAGResult(
                documents=[],
                query=rag_query,
                confidence=0.0,
                processing_time=0.0,
                source="error",
            )

    async def get_stats(self) -> Dict[str, Any]:
        """Estatísticas do sistema RAG"""
        vs_stats = {}
        if self._vector_store:
            vs_stats = await self._vector_store.get_stats()
        return {
            "vector_store": vs_stats,
            "metrics": self.metrics.copy(),
            "impl": "new_core_rag",
            "config": self.config,
            "status": "operational",
        }

    @staticmethod
    def _parse_markdown(content: str) -> List[Tuple[str, str]]:
        """Parseia documento Markdown em seções"""
        sections = []
        lines = content.split('\n')
        current_title = ""
        current_content = []

        for line in lines:
            if line.startswith('#'):
                if current_title and current_content:
                    sections.append((current_title, '\n'.join(current_content)))
                current_title = line.strip()
                current_content = []
            else:
                if line.strip():
                    current_content.append(line.strip())

        if current_title and current_content:
            sections.append((current_title, '\n'.join(current_content)))

        return sections

    @staticmethod
    def _infer_agent_type(title: str, content: str) -> str:
        """Infere tipo de agente baseado no conteúdo"""
        title_lower = title.lower()
        content_lower = content.lower()

        if any(kw in title_lower or kw in content_lower for kw in ["horário", "atendimento", "contato", "whatsapp", "email", "suporte"]):
            return "atendimento_geral"
        elif any(kw in title_lower or kw in content_lower for kw in ["conta", "criação", "cadastro", "abrir"]):
            return "criacao_conta"
        elif any(kw in title_lower or kw in content_lower for kw in ["golpe", "fraude", "segurança", "med"]):
            return "golpe_med"
        elif any(kw in title_lower or kw in content_lower for kw in ["banco", "conectar", "open finance", "financeiro"]):
            return "open_finance"
        else:
            return "atendimento_geral"


# ============================================================================
# RAG SINGLETON
# ============================================================================

_rag_system_instance: Optional[JotaRAGSystem] = None
_rag_lock = asyncio.Lock()


async def get_rag_system(config: Dict[str, Any] = None) -> JotaRAGSystem:
    """Obtém instância do sistema RAG (singleton)"""
    global _rag_system_instance

    if _rag_system_instance is None:
        async with _rag_lock:
            if _rag_system_instance is None:
                rag = JotaRAGSystem(config)
                await rag.initialize()
                _rag_system_instance = rag

    return _rag_system_instance


async def reset_rag_system():
    """Reseta a instância do sistema RAG (para testes)"""
    global _rag_system_instance

    async with _rag_lock:
        _rag_system_instance = None


# ============================================================================


class EscalationReason(Enum):
    """Categorias obrigatórias para escalation_reason"""
    USER_FRUSTRATION = "user_frustration"
    FRAUD_RISK = "fraud_risk"
    POLICY_BLOCK = "policy_block"
    TOOL_FAILURE = "tool_failure"
    MISSING_INFORMATION = "missing_information"
    COMPLIANCE_REQUIREMENT = "compliance_requirement"
    UNKNOWN = "unknown"

class AgentAction(Enum):
    """Ações que o agente pode tomar"""
    RESPOND = "respond"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"
    EXECUTE_ACTION = "execute_action"
    TRANSFER = "transfer"

class AgentPriority(Enum):
    """Prioridade das mensagens"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class AgentMessage:
    """Mensagem do cliente processada"""
    content: str
    user_id: str
    session_id: str
    timestamp: datetime
    context: Dict[str, Any]
    priority: AgentPriority = AgentPriority.MEDIUM
    metadata: Dict[str, Any] = None

@dataclass
class AgentDecision:
    """Decisão do agente"""
    action: AgentAction
    response: str
    confidence: float
    agent_type: str
    reasoning: str
    processing_time: float
    rag_used: bool
    should_escalate: bool = False
    escalation_reason: str = ""  # 🆕 Obrigatório se should_escalate=True
    evidence_pack: Dict[str, Any] = field(default_factory=dict)  # 🆕 Evidence pack para compatibilidade
    trace: Dict[str, Any] = field(default_factory=dict)  # 🆕 Rastreabilidade mínima
    actions: List[str] = field(default_factory=list)  # 🆕 Lista de ações executadas
    # 🆕 Grounding obrigatório
    grounded: bool = False
    grounding_score: float = 0.0
    citation_coverage: float = 0.0
    # 🆕 Risk-based routing (para golpe_med)
    risk_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    risk_factors: List[str] = field(default_factory=list)
    user_impact: str = "UNKNOWN"  # LOW, MEDIUM, HIGH, CRITICAL
    recommended_next_action: str = "RESPOND"  # RESPOND, ASK_CLARIFY, ESCALATE

# 🆕 Constantes de versionamento para cache determinístico
POLICY_VERSION = "1.0.0"
KB_VERSION_CACHE = {}  # Cache para hash da KB
RAG_SIGNATURE_CACHE = {}  # Cache para assinatura estável do RAG (embeddings)

# 🆕 ENTREGA B: Constantes de orquestração por fluxo
GENERAL = "atendimento_geral"
SPECIALISTS = {"criacao_conta", "open_finance", "golpe_med"}
VALID_AGENTS = {GENERAL} | SPECIALISTS

class DelegationReason(Enum):
    """Razões canônicas de delegação do atendimento_geral para especialistas"""
    FRAUD_SIGNAL = "fraud_signal"
    OPEN_FINANCE_SIGNAL = "open_finance_signal"
    ONBOARDING_SIGNAL = "onboarding_signal"
    UNKNOWN_GENERAL = "unknown_general"

def _get_kb_version() -> str:
    """Obtém hash da KB para versionamento de cache com mtime"""
    global KB_VERSION_CACHE
    
    try:
        kb_path = Path(__file__).resolve().parents[3] / "assets" / "knowledge_base" / "jota_kb_restructured.md"
        
        if not kb_path.exists():
            KB_VERSION_CACHE["version"] = "unknown"
            return KB_VERSION_CACHE["version"]

        # 🆕 Verificar mtime para invalidação automática
        current_mtime = kb_path.stat().st_mtime
        
        # Se cache existe e mtime não mudou, usar cache
        if "version" in KB_VERSION_CACHE and "mtime" in KB_VERSION_CACHE:
            if KB_VERSION_CACHE["mtime"] == current_mtime:
                return KB_VERSION_CACHE["version"]
        
        # Recalcular hash se mtime mudou
        with open(kb_path, 'r', encoding='utf-8') as f:
            content = f.read()
            version_hash = hashlib.md5(content.encode()).hexdigest()[:16]
            
            # Atualizar cache com novo hash e mtime
            KB_VERSION_CACHE["version"] = version_hash
            KB_VERSION_CACHE["mtime"] = current_mtime
            
            return version_hash
            
    except Exception as e:
        logger.warning(f"Error calculating KB version: {e}")
        KB_VERSION_CACHE["version"] = "error"
        return KB_VERSION_CACHE["version"]


def _normalize_query_text(text: str) -> str:
    """Normalize query text for cache keys (PII-safe: never log raw text)."""
    if text is None:
        return ""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _hash_md5_16(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

def _get_conversation_hash(message_history: list = None, max_messages: int = 3) -> str:
    """Gera hash determinístico do histórico relevante"""
    if not message_history or len(message_history) == 0:
        return "no_history"
    
    # Pegar últimas N mensagens para hash (configurável)
    recent_history = message_history[-max_messages:] if len(message_history) > max_messages else message_history
    
    # Ordenar por timestamp para determinismo (se disponível)
    if recent_history and isinstance(recent_history[0], dict) and 'timestamp' in recent_history[0]:
        recent_history.sort(key=lambda x: x.get('timestamp', ''))
    
    # Extrair conteúdo de forma determinística
    history_content = []
    for msg in recent_history:
        if isinstance(msg, dict):
            # Excluir campos voláteis
            content = msg.get('content', str(msg))
            # Remover request_id e timestamps dinâmicos
            if isinstance(content, str):
                content = content.replace('request_id', '').replace('timestamp', '')
        else:
            content = str(msg)
        # Truncar para hash consistente
        history_content.append(content[:100])
    
    history_text = "|".join(history_content)
    return hashlib.md5(history_text.encode()).hexdigest()[:12]


class JotaAgentOrchestrator:
    """Orquestrador principal de agentes do Jota"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_secure_logger(f"{__name__}.JotaAgentOrchestrator")
        
        # Componentes
        self.rag_system = None
        self.llm_manager = None
        self.prompt_manager = None
        self.memory_adapter = None
        self.policy_engine = None
        self.memory_orchestrator = None  # 🆕 Orquestrador de memória
        
        # 🆕 CORREÇÃO: Usar cache stores singleton persistentes
        self.decision_cache = get_decision_cache()
        self._classification_cache = get_classify_cache()
        self._rag_cache = get_rag_cache()

        # Métricas
        self.metrics = {
            "messages_processed": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "rag_queries": 0,
            "classification_cache_hits": 0,
            "agent_distribution": {},
            # 🆕 Métricas de Grounding por agente
            "grounding_metrics": {
                "total_queries": 0,
                "grounded_count": 0,
                "avg_grounding_score": 0.0,
                "avg_citation_coverage": 0.0,
                "by_agent": {}
            },
            # 🆕 Métricas de negócio críticas
            "auto_resolution_rate": 0.0,
            "escalation_rate": 0.0,
            "tool_error_rate": 0.0,
            "rag_usage_rate": 0.0,
            "empty_context_rate": 0.0
        }
        
        self.logger.info("JotaAgentOrchestrator initialized")

    def _get_rag_signature(self) -> str:
        """Get a stable signature for current RAG embedding configuration (cached, no PII)."""
        global RAG_SIGNATURE_CACHE
        try:
            cached = RAG_SIGNATURE_CACHE.get("signature")
            if cached:
                return cached

            adapter = getattr(self.rag_system, "_embeddings_adapter", None) if self.rag_system else None
            if adapter and hasattr(adapter, "get_signature"):
                sig = adapter.get_signature()
                if hasattr(sig, "stable_hash"):
                    signature = sig.stable_hash()
                else:
                    signature = _hash_md5_16(str(sig))
            else:
                signature = "unknown"

            RAG_SIGNATURE_CACHE["signature"] = signature
            return signature
        except Exception as e:
            self.logger.warning(f"Error calculating RAG signature: {e}")
            RAG_SIGNATURE_CACHE["signature"] = "error"
            return RAG_SIGNATURE_CACHE["signature"]

    def _make_rag_cache_key(self, query_text: str) -> Tuple[str, str]:
        """Return (cache_key, key_hash) for rag_cache. Key does not depend on agent_type."""
        kb_version = _get_kb_version()
        rag_signature = self._get_rag_signature()
        normalized_query = _normalize_query_text(query_text)
        key_material = f"{normalized_query}|{kb_version}|{rag_signature}"
        key_hash = _hash_md5_16(key_material)
        return f"rag_{key_hash}", key_hash
    
    async def initialize(self) -> bool:
        """Inicializa o orquestrador"""
        try:
            # Inicializar componentes
            self.rag_system = await get_rag_system(self.config.get("rag", {}))
            self.llm_manager = await get_llm_manager(self.config.get("llm", {}).get("preferred_provider", "fallback"))
            self.prompt_manager = get_prompt_manager()  # Sem parâmetros
            self.memory_adapter = await get_simple_memory_adapter()  # Com await
            self.policy_engine = JotaPolicyEngine(self.config.get("policy", {}))
            self.memory_orchestrator = get_memory_orchestrator()  # Inicializar orquestrador de memória
            
            self.logger.info("JotaAgentOrchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing orchestrator: {e}")
            return False
    
    # ── ENTREGA B: Flow Orchestration Methods ──────────────────────────
    
    def _is_new_session(self, memory_context: Dict[str, Any]) -> Tuple[bool, str]:
        """Deterministic check: is this a new session?
        Returns (is_new, reason) for auditability.
        """
        stage = memory_context.get("memory_stage", "initial")
        turn_count = memory_context.get("memory_turn_count", 0)
        summary = memory_context.get("memory_summary", "")
        
        if not summary and stage == "initial" and turn_count == 0:
            return True, "no_memory_context"
        if stage == "initial":
            return True, f"stage_initial_turn_{turn_count}"
        if turn_count == 0:
            return True, "turn_count_zero"
        return False, f"existing_session_stage_{stage}_turn_{turn_count}"
    
    @staticmethod
    def _map_agent_to_delegation_reason(agent_type: str) -> str:
        """Maps a specialist agent to its canonical delegation reason."""
        mapping = {
            "golpe_med": DelegationReason.FRAUD_SIGNAL.value,
            "open_finance": DelegationReason.OPEN_FINANCE_SIGNAL.value,
            "criacao_conta": DelegationReason.ONBOARDING_SIGNAL.value,
        }
        return mapping.get(agent_type, DelegationReason.UNKNOWN_GENERAL.value)
    
    def _determine_delegation(self, message_content: str) -> Dict[str, Any]:
        """Triage by atendimento_geral: decide respond / delegate / escalate.
        Uses the same deterministic keyword rules already in the pipeline.
        Returns a structured decision dict.
        """
        keyword_agent, keyword_reason = self._classify_by_keywords(message_content)
        
        if keyword_agent and keyword_agent in SPECIALISTS:
            return {
                "action_type": "delegate",
                "agent_target": keyword_agent,
                "delegation_reason": self._map_agent_to_delegation_reason(keyword_agent),
                "routing_source": "keyword_rule",
                "routing_detail": keyword_reason,
            }
        
        # No specialist keyword match → atendimento_geral responds itself
        return {
            "action_type": "respond",
            "agent_target": GENERAL,
            "delegation_reason": DelegationReason.UNKNOWN_GENERAL.value,
            "routing_source": "no_specialist_match",
            "routing_detail": None,
        }
    
    async def process_message_flow(self, message: AgentMessage) -> Dict[str, Any]:
        """Flow-aware pipeline: enforces atendimento_geral as entry point for new sessions.
        Returns an enriched dict with flow metadata + the AgentDecision.
        """
        request_id = str(uuid.uuid4())
        flow_meta: Dict[str, Any] = {
            "request_id": request_id,
            "entry_agent": GENERAL,
            "delegated_to": None,
            "delegation_reason": None,
            "final_agent": None,
            "is_new_session": None,
            "new_session_reason": None,
            "logs_emitted": False,
        }
        
        try:
            # 1. Load memory
            memory_context = self.memory_orchestrator.load_memory_context(message.session_id)
            message.context.update(memory_context)
            
            # 2. Determine if new session
            is_new, new_reason = self._is_new_session(memory_context)
            flow_meta["is_new_session"] = is_new
            flow_meta["new_session_reason"] = new_reason
            
            if is_new:
                # 3. Triage via atendimento_geral keywords
                delegation = self._determine_delegation(message.content)
                
                if delegation["action_type"] == "delegate":
                    # Delegate to specialist
                    flow_meta["delegated_to"] = delegation["agent_target"]
                    flow_meta["delegation_reason"] = delegation["delegation_reason"]
                    flow_meta["final_agent"] = delegation["agent_target"]
                    
                    # Log structured audit event
                    self.logger.info(
                        "flow_delegation",
                        extra={
                            "event": "flow_delegation",
                            "request_id": request_id,
                            "entry_agent": GENERAL,
                            "delegated_to": delegation["agent_target"],
                            "delegation_reason": delegation["delegation_reason"],
                            "routing_source": delegation["routing_source"],
                            "routing_detail": delegation["routing_detail"],
                            "session_id_hash": message.session_id[:8] + "***",
                        }
                    )
                    flow_meta["logs_emitted"] = True
                else:
                    # atendimento_geral responds
                    flow_meta["final_agent"] = GENERAL
                    self.logger.info(
                        "flow_general_response",
                        extra={
                            "event": "flow_general_response",
                            "request_id": request_id,
                            "entry_agent": GENERAL,
                            "delegated_to": None,
                            "delegation_reason": DelegationReason.UNKNOWN_GENERAL.value,
                            "session_id_hash": message.session_id[:8] + "***",
                        }
                    )
                    flow_meta["logs_emitted"] = True
            else:
                # Existing session: use classified agent (may already be specialist)
                agent_type = await self._classify_with_cache(message)
                flow_meta["final_agent"] = agent_type
                flow_meta["delegated_to"] = agent_type if agent_type in SPECIALISTS else None
                flow_meta["delegation_reason"] = self._map_agent_to_delegation_reason(agent_type) if agent_type in SPECIALISTS else None
                
                self.logger.info(
                    "flow_existing_session",
                    extra={
                        "event": "flow_existing_session",
                        "request_id": request_id,
                        "entry_agent": GENERAL,
                        "resumed_agent": agent_type,
                        "session_id_hash": message.session_id[:8] + "***",
                        "new_session_reason": new_reason,
                    }
                )
                flow_meta["logs_emitted"] = True
            
            # 4. Execute the pipeline with the determined agent
            decision = await self.process_message(message)
            flow_meta["final_agent"] = decision.agent_type
            
            # 5. Persist memory (closes load→process→save cycle)
            try:
                self.memory_orchestrator.update_memory(
                    MemoryUpdate(
                        conversation_id=message.session_id,
                        agent_type=decision.agent_type,
                        summary_delta=decision.agent_type,
                        confidence=decision.confidence,
                    )
                )
            except Exception as e:
                self.logger.warning(f"[MEMORY_UPDATE_FAILED] {e}")
            
            # Strip internal citations [C#] from user-facing response
            decision.response = re.sub(r'\s*\[C\d+\]', '', decision.response).strip()
            
            return {
                "decision": decision,
                "flow": flow_meta,
            }
            
        except Exception as e:
            self.logger.error(f"Error in process_message_flow: {e}")
            flow_meta["error"] = str(e)
            error_decision = self._create_error_decision(str(e))
            flow_meta["final_agent"] = error_decision.agent_type
            return {
                "decision": error_decision,
                "flow": flow_meta,
            }
    
    async def process_message(self, message: AgentMessage) -> AgentDecision:
        """Pipeline otimizado para escala enterprise com profiling detalhado"""
        start_time = time.perf_counter()  # 🆕 CORREÇÃO: Usar perf_counter
        
        # FASE 0: Criar trace/request_id IMEDIATAMENTE no início
        request_id = str(uuid.uuid4())
        processing_step = "start"
        
        # PERFILING: Estrutura para coletar tempos por etapa
        step_timings = {}  # {step_name: elapsed_ms}
        cache_status = {}  # {cache_name: hit/miss}
        
        # Log de início (DEBUG; detalhes consolidados no REQUEST_EVENT final)
        self.logger.debug(
            "processing_start",
            extra={
                "event": "processing_start",
                "request_id": request_id,
                "message_len": len(message.content),
            }
        )
        
        agent_selected = None
        _final_decision = None
        _evt_error_info = None
        
        try:
            # FASE 0.5: Carregar contexto de memória estruturada
            processing_step = "memory_load"
            
            # CORREÇÃO: Usar perf_counter para timing preciso
            step_start = time.perf_counter()
            self.logger.debug(
                "pipeline_step_start",
                extra={
                    "event": "pipeline_step_start",
                    "request_id": request_id,
                    "step_name": processing_step,
                    "agent_type_current": "unknown",
                    "llm_migration_enabled": os.getenv("LLM_MIGRATION_ENABLED", "false")
                }
            )
            
            memory_context = self.memory_orchestrator.load_memory_context(message.session_id)
            
            # Integrar memória ao contexto da mensagem
            message.context.update(memory_context)
            
            # Log estruturado de fim da etapa com profiling preciso
            step_end = time.perf_counter()
            step_elapsed_ms = (step_end - step_start) * 1000  # CORREÇÃO: Calcular ms corretamente
            memory_keys_count = len(memory_context) if memory_context else 0
            step_timings[processing_step] = step_elapsed_ms
            
            self.logger.debug(
                "pipeline_step_end",
                extra={
                    "event": "pipeline_step_end",
                    "request_id": request_id,
                    "step_name": processing_step,
                    "elapsed_ms": round(step_elapsed_ms, 2),  # CORREÇÃO: Sempre em ms
                    "memory_keys_loaded": memory_keys_count,
                    "memory_loaded": memory_keys_count > 0
                }
            )
            
            # CACHE MULTI-NÍVEL: Verificar todos os caches em paralelo (ANTES da classificação)
            processing_step = "cache_check"
            # OTIMIZAÇÃO: Cache key melhorada com hash do conteúdo
            content_hash = hashlib.md5(f"{message.content}_{agent_selected}".encode()).hexdigest()[:8]
            cache_key = f"decision_{content_hash}"
            
            # Log estruturado de início da etapa
            step_start = time.perf_counter()
            self.logger.debug(
                "pipeline_step_start",
                extra={
                    "event": "pipeline_step_start",
                    "request_id": request_id,
                    "step_name": processing_step,
                    "agent_type_current": agent_selected,
                    "llm_migration_enabled": os.getenv("LLM_MIGRATION_ENABLED", "false")
                }
            )
            
            # Corrigir: _check_decision_cache é síncrono, não usar create_task
            decision_cache_task = self._check_decision_cache(cache_key)
            rag_cache_task = await self._check_rag_cache(message.content)
            
            # Executar cache em paralelo (decision_cache é síncrono)
            cached_decision = decision_cache_task
            cached_rag = rag_cache_task
            
            # CORREÇÃO: Registrar cache status
            cache_status["decision_cache_hit"] = cached_decision is not None
            cache_status["rag_cache_hit"] = cached_rag is not None
            cache_status["cache_key_hash"] = content_hash
            
            # CORREÇÃO: Verificar classify cache hit usando a mesma chave que _classify_with_cache
            classify_content_hash = hashlib.md5(message.content.lower().strip().encode()).hexdigest()[:8]
            classify_cache_key = f"classify_{classify_content_hash}"
            classify_cached = self._classification_cache.get(classify_cache_key)
            cache_status["classify_cache_hit"] = classify_cached is not None
            
            # DIAGNÓSTICO: Log cache store info
            self.logger.debug(
                "cache_get",
                extra={
                    "event": "cache_get",
                    "cache_name": "decision_cache",
                    "key_hash": content_hash,
                    "hit": cached_decision is not None,
                    "store_type": "singleton",
                    "store_id": id(self.decision_cache),
                    "store_size": self.decision_cache.size()
                }
            )

            # Log PII-safe do cache_get RAG (key_hash apenas)
            rag_cache_key, rag_key_hash = self._make_rag_cache_key(message.content)
            self.logger.debug(
                "cache_get",
                extra={
                    "event": "cache_get",
                    "cache_name": "rag_cache",
                    "key_hash": rag_key_hash,
                    "hit": cached_rag is not None,
                    "store_type": "singleton",
                    "store_id": id(self._rag_cache),
                    "store_size": self._rag_cache.size(),
                    "ttl_s": 300
                }
            )
            
            # Log estruturado de fim da etapa
            step_end = time.perf_counter()
            step_elapsed_ms = (step_end - step_start) * 1000
            step_timings[processing_step] = step_elapsed_ms
            
            self.logger.debug(
                "pipeline_step_end",
                extra={
                    "event": "pipeline_step_end",
                    "request_id": request_id,
                    "step_name": processing_step,
                    "elapsed_ms": round(step_elapsed_ms, 2),
                    "cache_hit_decision": cached_decision is not None,
                    "cache_hit_rag": cached_rag is not None,
                    "cache_key_hash": content_hash
                }
            )
            
            # CACHE HIT: Retornar decisão cacheada
            if cached_decision:
                self.metrics["cache_hits"] += 1
                self.logger.debug(f"DECISION_CACHE_HIT key={cache_key}")
                _final_decision = cached_decision
                return _final_decision
            
            # EXECUÇÃO PARALELA: Contexto + Classificação
            processing_step = "classify_llm"
            
            # Log estruturado de início da etapa
            step_start = time.perf_counter()
            self.logger.debug(
                "pipeline_step_start",
                extra={
                    "event": "pipeline_step_start",
                    "request_id": request_id,
                    "step_name": processing_step,
                    "agent_type_current": "unknown",
                    "llm_migration_enabled": os.getenv("LLM_MIGRATION_ENABLED", "false")
                }
            )
            
            # PARTE 2: Verificar memory_adapter antes de usar
            if self.memory_adapter is not None:
                context_task = self.memory_adapter.get_client_context(message.user_id)
            else:
                self.logger.warning("Memory adapter not initialized, using None context")
                context_task = None
            
            classification_task = self._classify_with_cache(message)
            
            # Executar em paralelo se context_task existir
            if context_task is not None:
                client_context, agent_selected = await asyncio.gather(
                    context_task, classification_task
                )
            else:
                client_context = None
                agent_selected = await classification_task
            
            # Log após classificação
            step_end = time.perf_counter()
            step_elapsed_ms = (step_end - step_start) * 1000
            step_timings[processing_step] = step_elapsed_ms
            
            self.logger.debug(
                "classification_complete",
                extra={
                    "event": "classification_complete",
                    "request_id": request_id,
                    "agent_selected": agent_selected,
                    "step": processing_step,
                    "elapsed_ms": round(step_elapsed_ms, 2)
                }
            )
            
            # Log estruturado de fim da etapa
            self.logger.debug(
                "pipeline_step_end",
                extra={
                    "event": "pipeline_step_end",
                    "request_id": request_id,
                    "step_name": processing_step,
                    "elapsed_ms": round(step_elapsed_ms, 2),
                    "agent_selected": agent_selected,
                    "client_context_loaded": client_context is not None
                }
            )
            
            # RAG OTIMIZADO: Buscar contexto se necessário
            processing_step = "rag_retrieve"
            rag_result = cached_rag
            
            # Log estruturado de início da etapa RAG
            step_start = time.perf_counter()
            self.logger.debug(
                "pipeline_step_start",
                extra={
                    "event": "pipeline_step_start",
                    "request_id": request_id,
                    "step_name": processing_step,
                    "agent_type_current": agent_selected,
                    "llm_migration_enabled": os.getenv("LLM_MIGRATION_ENABLED", "false")
                }
            )
            
            if not rag_result or not rag_result.documents:
                rag_task = await self._query_rag_optimized(message, agent_selected)
                rag_result = rag_task
                # Instrumentação: contabilizar RAG query do orchestrator
                try:
                    from support_agent.agents.base_agent import _rag_metrics
                    _rag_metrics["orchestrator_rag_queries"] += 1
                except Exception:
                    pass
            
            # Log estruturado de fim da etapa RAG com profiling detalhado
            step_end = time.perf_counter()
            step_elapsed_ms = (step_end - step_start) * 1000
            rag_docs_count = len(rag_result.documents) if rag_result else 0
            rag_context_chars = sum(len(doc.content) for doc in rag_result.documents) if rag_result else 0
            step_timings[processing_step] = step_elapsed_ms
            
            self.logger.debug(
                "pipeline_step_end",
                extra={
                    "event": "pipeline_step_end",
                    "request_id": request_id,
                    "step_name": processing_step,
                    "elapsed_ms": round(step_elapsed_ms, 2),
                    "rag_documents_found": rag_docs_count,
                    "rag_used": rag_docs_count > 0,
                    "rag_context_chars": rag_context_chars,
                    "rag_top_k": 3  # Fixo no código atual
                }
            )
            
            # ANSWERABILITY GATE: Verificar se base suporta a pergunta
            processing_step = "answerability_gate"
            
            # Log estruturado de início da etapa
            step_start = time.perf_counter()
            self.logger.debug(
                "pipeline_step_start",
                extra={
                    "event": "pipeline_step_start",
                    "request_id": request_id,
                    "step_name": processing_step,
                    "agent_type_current": agent_selected,
                    "llm_migration_enabled": os.getenv("LLM_MIGRATION_ENABLED", "false")
                }
            )
            
            answerability_result = await self._check_answerability_gate(message, rag_result, agent_selected)
            
            # Log estruturado de fim da etapa com profiling
            step_end = time.perf_counter()
            step_elapsed_ms = (step_end - step_start) * 1000
            step_timings[processing_step] = step_elapsed_ms
            
            self.logger.debug(
                "pipeline_step_end",
                extra={
                    "event": "pipeline_step_end",
                    "request_id": request_id,
                    "step_name": processing_step,
                    "elapsed_ms": round(step_elapsed_ms, 2),
                    "answerable": answerability_result["answerable"],
                    "answerability_reason": answerability_result.get("reason", "unknown")
                }
            )
            
            if not answerability_result["answerable"]:
                # Criar decisão de recusa controlada
                _final_decision = AgentDecision(
                    action=AgentAction.RESPOND,
                    response=answerability_result["message"],
                    confidence=0.9,  # Alta confiança na recusa
                    agent_type=agent_selected,
                    reasoning=f"Answerability Gate: {answerability_result['reason']}",
                    processing_time=0.001,
                    rag_used=False,
                    should_escalate=answerability_result.get("should_escalate", False),
                    escalation_reason="Answerability Gate failed",
                    evidence_pack={},  # FASE 2: Sempre incluir evidence_pack vazio
                    trace={"request_id": request_id, "processing_step": processing_step, "answerability_gate": "failed"}  # FASE 2: Trace completo
                )
                return _final_decision
            
            # EXECUÇÃO DO AGENTE ESPECIALISTA
            try:
                processing_step = "response_generate_llm"
                
                # Log estruturado de início da etapa
                step_start = time.perf_counter()
                self.logger.debug(
                    "pipeline_step_start",
                    extra={
                        "event": "pipeline_step_start",
                        "request_id": request_id,
                        "step_name": processing_step,
                        "agent_type_current": agent_selected,
                        "llm_migration_enabled": os.getenv("LLM_MIGRATION_ENABLED", "false")
                    }
                )
                
                decision = await self._execute_agent_specialist(message, rag_result, client_context, agent_selected)
                
                # Ajustar confiança baseado no soft gate
                if answerability_result.get("confidence_adjusted", False):
                    # Reduz confiança se critérios baixos mas docs existem
                    decision.confidence = max(decision.confidence * 0.7, 0.45)
                    self.logger.info(f" Confidence adjusted to {decision.confidence:.2f} (soft gate)")
                
                if answerability_result.get("needs_clarification", False):
                    # Adicionar pedido de clarificação se muito fraco
                    decision.response += "\n\nPara te ajudar melhor, poderia me dar mais detalhes sobre sua dúvida?"
                    self.logger.info(f"🔄 Added clarification request (very weak evidence)")
                
                # 🆕 Garantir que decision tenha request_id no trace
                if not hasattr(decision, 'trace') or not decision.trace:
                    decision.trace = {}
                decision.trace["request_id"] = request_id
                decision.trace["processing_step"] = "success"
                
                # Log estruturado de fim da etapa com profiling detalhado
                step_end = time.perf_counter()
                step_elapsed_ms = (step_end - step_start) * 1000
                step_timings[processing_step] = step_elapsed_ms
                
                # 🆕 PERFILING: Calcular tempo total preciso
                total_elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                # 🆕 PERFILING: Adicionar breakdown completo ao trace
                decision.trace = decision.trace or {}
                decision.trace.update({
                    "request_id": request_id,
                    "processing_step": "success",
                    "step_timings_ms": {k: round(v, 2) for k, v in step_timings.items()},
                    "total_processing_time_ms": round(total_elapsed_ms, 2),
                    "cache_status": cache_status.copy()
                })
                
                self.logger.debug(
                    "pipeline_step_end",
                    extra={
                        "event": "pipeline_step_end",
                        "request_id": request_id,
                        "step_name": processing_step,
                        "elapsed_ms": round(step_elapsed_ms, 2),
                        "agent_type_final": decision.agent_type,
                        "confidence": decision.confidence,
                        "rag_used": decision.rag_used,
                        "should_escalate": decision.should_escalate,
                        "llm_provider": getattr(decision, 'llm_provider', 'unknown')
                    }
                )
                
                # 🚀 CACHE: Salvar decision cache de forma síncrona (não fire-and-forget)
                await self._save_cache_async(cache_key, decision, rag_result)
                
            except Exception as agent_error:
                # Log estruturado de erro na etapa
                step_elapsed_ms = (time.perf_counter() - step_start) * 1000
                self.logger.error(
                    "pipeline_step_error",
                    extra={
                        "event": "pipeline_step_error",
                        "request_id": request_id,
                        "step_name": processing_step,
                        "elapsed_ms": round(step_elapsed_ms, 2),
                        "exception_type": type(agent_error).__name__,
                        "exception_message": str(agent_error),
                        "stacktrace": traceback.format_exc(),
                        "agent_type_current": agent_selected
                    }
                )
                
                self.logger.error(f"❌ Error executing agent {agent_selected}: {agent_error}")
                # 🆕 PARTE 1: Usar traceback do módulo global (já importado no topo)
                self.logger.error(f"❌ Agent Error Traceback: {traceback.format_exc()}")
                # Fallback para erro com trace
                error_decision = self._create_error_decision(str(agent_error))
                error_decision.trace = {"request_id": request_id, "processing_step": "agent_error", "error_type": "agent_execution_error"}
                _final_decision = error_decision
                _evt_error_info = {"type": type(agent_error).__name__, "msg": str(agent_error)[:200]}
                return _final_decision
            
            # Métricas
            processing_time = (time.perf_counter() - start_time) * 1000 / 1000  # 🆕 CORREÇÃO: Converter para segundos
            self._update_metrics(decision, processing_time, agent_selected)
            
            # 🆕 Atualizar métricas de negócio
            self._update_business_metrics(decision, agent_selected)
            
            # 🆕 Validar escalation_reason obrigatório
            decision = self._validate_escalation_reason(decision)
            
            # Log de sucesso (DEBUG; superseded by REQUEST_EVENT)
            self.logger.debug(
                "processing_end",
                extra={
                    "event": "processing_end",
                    "request_id": request_id,
                    "agent_selected": agent_selected,
                    "step": "success",
                    "total_processing_time_ms": round(total_elapsed_ms, 2),  # 🆕 CORREÇÃO: Tempo total em ms
                    "confidence": decision.confidence,
                    "escalation_reason": decision.escalation_reason if decision.should_escalate else None
                }
            )
            
            _final_decision = decision
            return _final_decision
            
        except Exception as e:
            # 🆕 FASE 0: Except principal com stacktrace completo e log estruturado
            error_type = type(e).__name__
            error_message = str(e)
            full_stacktrace = traceback.format_exc()
            
            # Log estruturado de erro com stacktrace completo
            self.logger.error(
                "processing_error",
                extra={
                    "event": "message_processing_error",
                    "request_id": request_id,
                    "agent_selected": agent_selected,
                    "step": processing_step,
                    "error_type": error_type,
                    "error_message": error_message,
                    "stacktrace": full_stacktrace,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Log tradicional com stacktrace completo
            self.logger.exception(f"❌ CRITICAL ERROR in process_message: {error_type}: {error_message}")
            
            # Criar decisão de erro com trace
            error_decision = self._create_error_decision(error_message)
            error_decision.trace = {
                "request_id": request_id,
                "processing_step": processing_step,
                "error_type": error_type,
                "error_message": error_message
            }
            
            _final_decision = error_decision
            _evt_error_info = {"type": error_type, "msg": error_message[:200]}
            return _final_decision

        finally:
            # ── SINGLE STRUCTURED EVENT PER REQUEST (observability) ──
            try:
                _total_ms = (time.perf_counter() - start_time) * 1000
                _d = _final_decision
                _tr = (getattr(_d, "trace", None) or {}) if _d else {}
                _evt = {
                    "event": "REQUEST_EVENT",
                    "request_id": request_id,
                    "ts_ms": int(time.time() * 1000),
                    "user_id_hash": hashlib.md5(message.user_id.encode()).hexdigest()[:8],
                    "session_id_hash": hashlib.md5(message.session_id.encode()).hexdigest()[:8],
                    "message_len": len(message.content),
                    "agent_type_final": getattr(_d, "agent_type", None) if _d else agent_selected,
                    "should_escalate": getattr(_d, "should_escalate", None) if _d else None,
                    "confidence": round(getattr(_d, "confidence", 0.0), 3) if _d else None,
                    "rag_used": getattr(_d, "rag_used", None) if _d else None,
                    "success": _d is not None and bool(getattr(_d, "response", "")) and _evt_error_info is None,
                    "total_ms": round(_total_ms, 1),
                }
                # Step timings
                if step_timings:
                    _evt["timings_ms"] = {k: round(v, 1) for k, v in step_timings.items()}
                # Cache status
                _cache_flags = {k: v for k, v in cache_status.items() if k.endswith("_hit")}
                if _cache_flags:
                    _evt["cache"] = _cache_flags
                # RAG guard flags
                if _tr.get("rag_reuse_guard_passed") is not None or _tr.get("rag_reuse_guard_rejected"):
                    _evt["rag_guard"] = {
                        "passed": _tr.get("rag_reuse_guard_passed"),
                        "rejected_reason": _tr.get("rag_reuse_invalid_reason"),
                    }
                # KB / embedding info
                _meta_snap = _tr.get("_meta") or {}
                _kb_v = _meta_snap.get("kb_version") or cache_status.get("kb_version")
                if _kb_v:
                    _evt["kb_version"] = _kb_v
                # Model info (safe subset, no prompt)
                _model_info = {}
                for _mk in ("provider_used", "model_used"):
                    if _tr.get(_mk):
                        _model_info[_mk] = _tr[_mk]
                if _model_info:
                    _evt["model"] = _model_info
                # Error
                if _evt_error_info:
                    _evt["error"] = _evt_error_info
                # ── PII GUARD (last-resort safety net) ──
                # DO NOT ADD PII FIELDS TO REQUEST_EVENT. This guard is a last-resort safety net.
                _json_str = json.dumps(_evt, ensure_ascii=False, default=str)
                try:
                    _pii_patterns = {
                        "email": r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
                        "cpf": r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',
                        "cnpj": r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b',
                        "api_key": r'\bsk-[A-Za-z0-9]{20,}\b',
                    }
                    _pii_hits = [k for k, p in _pii_patterns.items() if re.search(p, _json_str)]
                    if _pii_hits:
                        _evt["pii_guard_tripped"] = True
                        _evt["pii_guard_matches"] = _pii_hits
                        # Sanitize textual fields that could carry PII
                        if isinstance(_evt.get("error"), dict) and _evt["error"].get("msg"):
                            _evt["error"]["msg"] = "<redacted>"
                        for _tk in ("reasoning", "debug", "notes"):
                            if _tk in _evt:
                                _evt[_tk] = "<redacted>"
                        _json_str = json.dumps(_evt, ensure_ascii=False, default=str)
                except Exception:
                    # Fail-safe: emit minimal event if guard itself crashes
                    _evt_safe = {k: _evt[k] for k in ("event", "request_id", "ts_ms", "success", "total_ms") if k in _evt}
                    _evt_safe["pii_guard_tripped"] = True
                    _evt_safe["pii_guard_matches"] = ["guard_internal_error"]
                    _json_str = json.dumps(_evt_safe, ensure_ascii=False, default=str)

                # Emit via underlying logger to avoid SecureLogger redaction
                # (the event is PII-safe by construction; redact_secrets corrupts JSON digit sequences)
                _raw_logger = getattr(self.logger, "logger", self.logger)
                _raw_logger.info("REQUEST_EVENT " + _json_str)
            except Exception:
                pass  # never let event emission break the pipeline

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Remove diacritical marks for keyword matching."""
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    def _classify_by_keywords(self, text: str) -> tuple:
        """Deterministic keyword-based priority routing.
        Returns (agent_type, reason) or (None, None) if no rule matches.
        Rules are high-precision only — ambiguous cases fall through to LLM.
        """
        t = text.lower().strip()
        t_norm = re.sub(r'[^\w\s]', ' ', t)
        t_norm = re.sub(r'\s+', ' ', t_norm).strip()
        t_norm = self._strip_accents(t_norm)

        # ── GOLPE_MED: fraud, scam, MED, BO, suspicious transactions ──
        golpe_strong = [
            r"\bgolpe\b", r"\bfraude\b", r"\bmed\b",
            r"mecanismo especial", r"\bbo\b", r"boletim de ocorr",
            r"\bvitim[ae]\b", r"transacao suspeita", r"atividade suspeita",
            r"\bhacke\w*\b", r"\broub\w*\b",
        ]
        for pattern in golpe_strong:
            if re.search(pattern, t_norm):
                return ("golpe_med", f"strong_pattern_{pattern}")

        golpe_keywords = [
            "contestar", "pix errado", "chave errada",
            "produto que nao chegou",
            "invadi", "estorno", "estornar", "devolu",
            "emprestei dinheiro", "nao devolve",
            "bloquear minha conta",
        ]
        for kw in golpe_keywords:
            if kw in t_norm:
                return ("golpe_med", f"keyword_rule_{kw}")

        golpe_compounds = [
            ("pix", "errad"), ("mandei", "errad"),
            ("pix", "estorn"), ("pix", "contestar"),
            ("pix", "nao reconhe"),
            ("comprei", "nao chegou"),
            ("recebi", "nao reconhe"),
            ("tempo", "med"), ("garantido", "med"),
            ("obrigat", "med"), ("obrigat", "contestar"),
        ]
        for a, b in golpe_compounds:
            if a in t_norm and b in t_norm:
                return ("golpe_med", f"compound_rule_{a}+{b}")

        # ── OPEN_FINANCE: bank connection, Open Finance ──
        of_strong = [
            r"\bopen finance\b",
            r"conectar (meu|minha)? ?banco",
            r"vincular (meu|minha)? ?banco",
            r"\bconsentimento\b",
            r"\binvalid_request_uri\b",
            r"\berr_unknown_url_scheme\b",
        ]
        for pattern in of_strong:
            if re.search(pattern, t_norm):
                return ("open_finance", f"strong_pattern_{pattern}")

        of_keywords = [
            "banco conectado", "saldo da minha conta",
            "extrato do meu banco", "extrato da minha conta",
        ]
        for kw in of_keywords:
            if kw in t_norm:
                return ("open_finance", f"keyword_rule_{kw}")

        of_compounds = [
            ("conta bancaria", "acessar"), ("conta bancaria", "erro"),
            ("conta bancaria", "conectar"), ("conta bancaria", "saldo"),
            ("conectar", "banco"),
        ]
        for a, b in of_compounds:
            if a in t_norm and b in t_norm:
                return ("open_finance", f"compound_rule_{a}+{b}")

        bank_names = [
            "nubank", "itau", "santander", "bradesco",
            "caixa", "banco do brasil", "inter", "c6", "btg",
            "xp", "sicredi", "sicoob", "mercado pago", "pagbank", "stone",
        ]
        of_intents = ["conectar", "conexao", "integr", "vincul", "acessar"]
        for bank in bank_names:
            if bank in t_norm:
                for intent in of_intents:
                    if intent in t_norm:
                        return ("open_finance", f"bank_rule_{bank}+{intent}")
                if any(w in t_norm for w in ["nao consigo", "erro", "problema"]):
                    return ("open_finance", f"bank_problem_{bank}")

        # ── CRIACAO_CONTA: account creation, onboarding ──
        cc_strong = [
            r"abrir (uma )?conta",
            r"criar (uma )?conta",
            r"\bselfie\b",
            r"cpf irregular",
            r"\bcnpj\b",
            r"conta p[fj]\b",
            r"conta pessoa",
        ]
        for pattern in cc_strong:
            if re.search(pattern, t_norm):
                return ("criacao_conta", f"strong_pattern_{pattern}")

        cc_compounds = [
            ("documento", "aprovad"), ("documento", "envi"),
            ("passaporte", "nao aceitou"),
            ("email", "cadastro"),
            ("camera", "cadastro"),
            ("filho", "conta"), ("menor", "conta"),
            ("outro cpf", "conta"),
        ]
        for a, b in cc_compounds:
            if a in t_norm and b in t_norm:
                return ("criacao_conta", f"compound_rule_{a}+{b}")

        return (None, None)

    async def _classify_with_cache(self, message: AgentMessage) -> str:
        """Classificação com cache otimizado e TTL reduzido"""
        # Agentes válidos (constante de módulo)
        valid_agents = VALID_AGENTS
        
        try:
            # 🚀 CACHE DE CLASSIFICAÇÃO COM CHAVE MELHORADA
            # Incluir hash do conteúdo para melhor cache hit
            content_hash = hashlib.md5(message.content.lower().strip().encode()).hexdigest()[:8]
            cache_key = f"classify_{content_hash}"
            
            # Verificar cache de classificação usando o método get()
            cached_value = self._classification_cache.get(cache_key)
            if cached_value:
                    cached_agent, cached_timestamp = cached_value
                    # 🆕 OTIMIZAÇÃO: TTL reduzido de 24h para 1h para frescor
                    cache_age_seconds = (datetime.now() - cached_timestamp).total_seconds()
                    if cache_age_seconds < 3600:  # 1 hora TTL
                        self.metrics["classification_cache_hits"] = self.metrics.get("classification_cache_hits", 0) + 1
                        # 🆕 FASE 2: Validação de sanidade do agente cacheado
                        if cached_agent in valid_agents:
                            # 🆕 DIAGNÓSTICO: Log cache hit
                            self.logger.debug(
                                "cache_hit",
                                extra={
                                    "event": "cache_hit",
                                    "cache_name": "classify_cache",
                                    "key_hash": content_hash,
                                    "hit": True,
                                    "store_type": "singleton",
                                    "store_id": id(self._classification_cache),
                                    "store_size": self._classification_cache.size(),
                                    "ttl_s": 3600
                                }
                            )
                            return cached_agent
                        else:
                            self.logger.warning(f"🚫 Invalid cached agent: {cached_agent} - removing cache")
                            self._classification_cache.delete(cache_key)
                    else:
                        # Cache expirado
                        self._classification_cache.delete(cache_key)
            
            # 🚀 PRIORITY ROUTING: Deterministic keyword rules BEFORE LLM
            keyword_agent, keyword_reason = self._classify_by_keywords(message.content)
            if keyword_agent:
                self.logger.info(
                    f"[ROUTING] event=routing_override agent={keyword_agent} "
                    f"reason={keyword_reason} source=keyword_rule"
                )
                agent_type = keyword_agent
            else:
                # Fallback to LLM classification
                agent_type = await self._classify_with_llm(message.content)
            
            # 🆕 FASE 2: Validação de sanidade do agent_name
            if agent_type not in valid_agents:
                self.logger.warning(f"🚫 Invalid agent from LLM: '{agent_type}' - using atendimento_geral fallback")
                agent_type = "atendimento_geral"
            
            # 🚀 CACHE RESULTADO com timestamp (apenas se for válido)
            try:
                dynamic_agents = self.prompt_manager.get_available_agents()
                all_valid_agents = VALID_AGENTS | set(dynamic_agents)
                
                if agent_type in all_valid_agents:
                    self._classification_cache.set(cache_key, (agent_type, datetime.now()), ttl_seconds=3600)
                    # 🆕 DIAGNÓSTICO: Log cache set
                    self.logger.info(
                        "cache_set",
                        extra={
                            "event": "cache_set",
                            "cache_name": "classify_cache",
                            "key_hash": content_hash,
                            "hit": False,
                            "store_type": "singleton",
                            "store_id": id(self._classification_cache),
                            "store_size": self._classification_cache.size(),
                            "ttl_s": 3600
                        }
                    )
                else:
                    self.logger.warning(f"❌ Not caching invalid agent: {agent_type}")
                
                return agent_type
                
            except Exception as cache_error:
                # 🆕 Log estruturado para falha de dynamic_agents
                self.logger.warning(
                    "prompt_manager_agents_failed",
                    extra={
                        "event": "prompt_manager_agents_failed",
                        "exception_type": type(cache_error).__name__,
                        "exception_message": str(cache_error),
                        "fallback_used": "static_agents_only"
                    }
                )
                # Retornar mesmo sem cache se a validação do cache falhar
                return agent_type
            
        except Exception as e:
            # 🆕 Fallback seguro para erros de classificação recuperáveis
            self.logger.warning(f"LLM classification failed: {e}")
            return "atendimento_geral"  # valid_agents disponível no escopo
    
    async def _classify_with_llm(self, message_content: str) -> str:
        """Classificação usando LLM Manager - sem fallback"""
        try:
            # Prompt simples e direto
            prompt = f"""
Você é um classificador de roteamento estrito.

Sua tarefa:
Escolher EXATAMENTE UM dos agentes abaixo.

Agentes permitidos (sem variações, sem explicações):
- atendimento_geral
- criacao_conta
- open_finance
- golpe_med

Definição de cada agente:

atendimento_geral:
Dúvidas gerais sobre o Jota, funcionamento, limites, taxas, Pix, rendimento, horário, funcionalidades, saldo, extrato, problemas técnicos ou perguntas gerais.

criacao_conta:
Mensagens sobre abertura de conta, cadastro, envio de CPF/CNPJ, documentos, aprovação, ativação, selfie, problemas durante o cadastro inicial.

open_finance:
Mensagens sobre conectar bancos, integração com outros bancos, visualizar saldo de outro banco, pagar com outro banco conectado, problemas ao conectar banco.

golpe_med:
Mensagens sobre golpe, fraude, roubo, conta invadida, transação suspeita, segurança urgente.

Regras obrigatórias:
1. Retorne SOMENTE o nome do agente.
2. Não explique.
3. Não use pontuação.
4. Não adicione texto extra.
5. Se houver dúvida, escolha atendimento_geral.
6. A resposta deve estar em letras minúsculas.
7. A resposta deve corresponder EXATAMENTE a uma das quatro opções.

Mensagem:
\"\"\"{message_content}\"\"\"

Resposta:
"""

            # Delegar toda configuração para o LLM Manager
            llm_response = await asyncio.wait_for(
                self.llm_manager.generate_classification(prompt),
                timeout=float(os.getenv("LLM_CLASSIFICATION_TIMEOUT", "15.0"))
            )
            
            # Extrair e validar resposta com correção de prefixos
            raw_response = llm_response.content.strip() if hasattr(llm_response, 'content') else str(llm_response).strip()
            # 🆕 Extrair apenas primeira palavra
            agent_type = raw_response.split()[0].lower() if raw_response else ""
            
            # Debug: mostrar resposta bruta do LLM
            self.logger.info(f"🤖 LLM Raw Response: '{llm_response.content if hasattr(llm_response, 'content') else str(llm_response)}'")
            self.logger.info(f"🎯 Processed agent_type: '{agent_type}'")
            self.logger.info(f"🔄 Provider used: {getattr(llm_response, 'provider_used', 'unknown')}")
            self.logger.info(f"🆘 Fallback used: {getattr(llm_response, 'fallback_used', False)}")
            
            # Se LLM retornar inválido, usar fallback padrão
            if agent_type not in VALID_AGENTS:
                self.logger.warning(f"❌ Invalid agent from LLM: '{agent_type}' - using atendimento_geral fallback")
                agent_type = "atendimento_geral"
            
            # 🆕 RAG PURO: Remover correção por prefixo - validar agente diretamente
            dynamic_agents = self.prompt_manager.get_available_agents()
            all_valid_agents = VALID_AGENTS | set(dynamic_agents)
            
            if agent_type in all_valid_agents:
                self.logger.info(f"✅ LLM Classification: '{message_content[:30]}...' → {agent_type}")
                return agent_type
            else:
                self.logger.error(f"❌ Invalid agent from LLM: '{agent_type}' - expected one of {all_valid_agents}")
                return "atendimento_geral"  # Apenas fallback crítico
                
        except asyncio.TimeoutError:
            self.logger.error("❌ LLM classification timeout")
            return "atendimento_geral"
        except Exception as e:
            self.logger.error(f"❌ Error in LLM classification: {e}")
            return "atendimento_geral"
    
    async def _check_answerability_gate(self, message: AgentMessage, rag_result: Optional[RAGResult], agent_type: str) -> Dict[str, Any]:
        """
        Answerability Gate 2.0 com verificação de relevância semântica
        Verifica se a base de conhecimento suporta a pergunta antes de processar
        """
        try:
            # Critérios de answerability
            if not rag_result or not rag_result.documents:
                self.logger.info(f"🚫 ANSWERABILITY GATE: Sem documentos RAG para {agent_type}")
                return {
                    "answerable": False,
                    "reason": "no_documents",
                    "message": "Não encontrei informações específicas sobre isso na base do conhecimento do Jota. Posso te ajudar com outra dúvida?",
                    "should_escalate": False
                }
            
            # Critério 1: Score médio dos documentos
            scores = [doc.score for doc in rag_result.documents[:3]]
            avg_score = sum(scores) / len(scores)
            
            # Critério 2: Score do melhor documento
            best_score = max(scores)
            
            # Critério 3: Verificação de relevância semântica (NOVO)
            query_lower = message.content.lower()
            top_chunk_content = rag_result.documents[0].content.lower()
            
            # Verificar se termos-chave da query aparecem no top chunk
            query_terms = set(query_lower.split())
            chunk_terms = set(top_chunk_content.split())
            
            # Remover stopwords (constante de módulo _PT_STOPWORDS)
            meaningful_query_terms = query_terms - _PT_STOPWORDS
            meaningful_chunk_terms = chunk_terms - _PT_STOPWORDS
            
            # Relevância semântica: pelo menos 1 termo significativo em comum OU 70% de similaridade
            semantic_overlap = len(meaningful_query_terms.intersection(meaningful_chunk_terms))
            semantic_relevance = semantic_overlap >= 1 or (len(meaningful_query_terms) > 0 and semantic_overlap / len(meaningful_query_terms) >= 0.7)
            
            # 🆕 RAG PURO: Remover keywords hardcoded - usar apenas validação estrutural
            # Critério 4: Removido - não usar keywords específicas hardcoded
            
            # Critério 5: Comprimento do conteúdo dos chunks
            total_content_length = sum(len(doc.content) for doc in rag_result.documents[:3])
            
            # 🆕 SOFT GATE: Nunca bloquear quando há documentos com score razoável
            min_docs_threshold = 1
            min_score_threshold = 0.10
            
            has_docs = rag_result and len(rag_result.documents) >= min_docs_threshold
            has_min_score = best_score >= min_score_threshold
            
            # Decisão baseada em critérios mais permissivos
            score_threshold = 0.15      # Reduzido de 0.4
            avg_score_threshold = 0.1   # Reduzido de 0.3
            content_length_threshold = 50  # Reduzido de 100
            keyword_threshold = 0       # Reduzido de 1
            semantic_threshold = False   # Desativado
            
            # Log detalhado para debugging
            self.logger.info(f"🔍 ANSWERABILITY ANALYSIS 2.0:")
            self.logger.info(f"   - Best score: {best_score:.3f} (threshold: {score_threshold})")
            self.logger.info(f"   - Avg score: {avg_score:.3f} (threshold: {avg_score_threshold})")
            self.logger.info(f"   - Content length: {total_content_length} (threshold: {content_length_threshold})")
            self.logger.info(f"   - Semantic relevance: {semantic_relevance} (overlap: {semantic_overlap})")
            self.logger.info(f"   - Documents: {len(rag_result.documents)}")
            self.logger.info(f"   - Query terms: {meaningful_query_terms}")
            self.logger.info(f"   - Chunk terms preview: {list(meaningful_chunk_terms)[:10]}")
            
            # Avaliação final - ajustada sem keyword_matches
            criteria_met = 0
            if best_score >= score_threshold:
                criteria_met += 1
            if avg_score >= avg_score_threshold:
                criteria_met += 1
            if total_content_length >= content_length_threshold:
                criteria_met += 1
            if semantic_relevance:
                criteria_met += 1
            
            # Se temos docs com score mínimo, sempre permitir (soft gate)
            if has_docs and has_min_score:
                self.logger.info(f"✅ SOFT GATE: APROVADO (docs={len(rag_result.documents)}, best_score={best_score:.3f})")
                return {
                    "answerable": True,
                    "reason": "soft_gate_approved",
                    "criteria_met": criteria_met,
                    "scores": scores,
                    "avg_score": avg_score,
                    "best_score": best_score,
                    "semantic_relevance": semantic_relevance,
                    "semantic_overlap": semantic_overlap,
                    "confidence_adjusted": criteria_met < 4,  # Reduz confiança se critérios baixos
                    "needs_clarification": criteria_met < 2  # Pede clarificação se muito fraco
                }
            
            # Fallback original só se não tiver docs ou score muito baixo
            # 🆕 PARTE 1: Garantir que answerable seja sempre definida
            answerable = True  # Valor padrão seguro
            if criteria_met >= 3:
                answerable = True
                self.logger.info(f"✅ ANSWERABILITY GATE: APROVADO ({criteria_met}/5 critérios)")
                return {
                    "answerable": answerable,
                    "reason": "criteria_met",
                    "criteria_met": criteria_met,
                    "scores": scores,
                    "avg_score": avg_score,
                    "best_score": best_score,
                    "semantic_relevance": semantic_relevance,
                    "semantic_overlap": semantic_overlap
                }
            else:
                answerable = False
                self.logger.info(f"🚫 ANSWERABILITY GATE: REPROVADO ({criteria_met}/5 critérios) - SEM DOCS OU SCORE MUITO BAIXO")
                
                # Tentar retry com top_k maior (NOVO)
                if criteria_met >= 2 and not semantic_relevance:
                    self.logger.info("🔄 ATTEMPTING RETRY with larger top_k...")
                    return await self._retry_with_larger_topk(message, agent_type, criteria_met)
                
                # Mensagem específica baseada no que falhou
                if not semantic_relevance:
                    user_message = "Essa pergunta parece estar fora do meu escopo de atuação. Posso te ajudar com assuntos relacionados ao Jota?"
                elif best_score < score_threshold:
                    user_message = "Não encontrei informações suficientemente relevantes sobre isso. Você poderia reformular sua pergunta?"
                else:
                    user_message = "Não encontrei informações específicas sobre isso na base do conhecimento do Jota. Posso te ajudar com outra dúvida?"
                
                return {
                    "answerable": answerable,
                    "reason": "insufficient_criteria",
                    "criteria_met": criteria_met,
                    "message": user_message,
                    "should_escalate": agent_type == "golpe_med" and rag_result and len(rag_result.documents) > 0
                }
                
        except Exception as e:
            self.logger.error(f"Error in answerability gate: {e}")
            # Em caso de erro, permitir processamento com log
            self.logger.warning("⚠️ ANSWERABILITY GATE: Erro, permitindo processamento")
            return {
                "answerable": True,
                "reason": "error_fallback",
                "error": str(e)
            }
    
    async def _retry_with_larger_topk(self, message: AgentMessage, agent_type: str, original_criteria: int) -> Dict[str, Any]:
        """Tenta novamente com top_k maior antes de recusar"""
        try:
            initial_top_k = 2 if agent_type in ["golpe_med", "criacao_conta"] else 3
            retry_top_k = 8
            self.logger.info(
                f"[RAG_RETRY] event=rag_answerability_retry agent_type={agent_type} "
                f"initial_top_k={initial_top_k} retry_top_k={retry_top_k} "
                f"criteria_met={original_criteria} reason=low_criteria_no_semantic"
            )
            
            # Buscar com top_k maior
            rag_query = RAGQuery(
                query=message.content,
                agent_type=agent_type,
                user_context={},
                top_k=retry_top_k,
                filters={"agent_type": agent_type}
            )
            
            rag_result = await self.rag_system.query(rag_query)
            
            if not rag_result or not rag_result.documents:
                return {
                    "answerable": False,
                    "reason": "retry_no_documents",
                    "message": "Não encontrei informações específicas sobre isso na base do conhecimento do Jota. Posso te ajudar com outra dúvida?",
                    "should_escalate": False
                }
            
            # Verificar se retry melhorou a relevância
            scores = [doc.score for doc in rag_result.documents[:3]]
            best_score = max(scores)
            avg_score = sum(scores) / len(scores)
            
            query_lower = message.content.lower()
            top_chunk_content = rag_result.documents[0].content.lower()
            
            query_terms = set(query_lower.split())
            chunk_terms = set(top_chunk_content.split())
            
            stopwords = {'o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'com', 'sem', 'como', 'onde', 'quando', 'que', 'qual', 'qual', 'é', 'são', 'foi', 'tem', 'tenho', 'ter', 'pode', 'poder', 'ser', 'estar', 'está', 'estou', 'vai', 'ir', 'faz', 'fazer', 'já', 'também', 'mais', 'muito', 'muita', 'bem', 'meu', 'minha', 'seu', 'sua', 'nosso', 'nossa', 'este', 'esta', 'isto', 'isso', 'aquilo', 'um', 'uma', 'uns', 'umas'}
            
            meaningful_query_terms = query_terms - stopwords
            meaningful_chunk_terms = chunk_terms - stopwords
            semantic_overlap = len(meaningful_query_terms.intersection(meaningful_chunk_terms))
            semantic_relevance = semantic_overlap >= 1
            
            self.logger.info(f"🔄 RETRY RESULTS:")
            self.logger.info(f"   - New best score: {best_score:.3f}")
            self.logger.info(f"   - New semantic relevance: {semantic_relevance}")
            self.logger.info(f"   - Semantic overlap: {semantic_overlap}")
            
            # Se retry melhorou significativamente, aprovar
            if best_score >= 0.5 and semantic_relevance:
                self.logger.info("✅ RETRY SUCCESS: Melhoria significativa detectada")
                return {
                    "answerable": True,
                    "reason": "retry_success",
                    "criteria_met": 5,
                    "scores": scores,
                    "avg_score": avg_score,
                    "best_score": best_score,
                    "semantic_relevance": semantic_relevance,
                    "semantic_overlap": semantic_overlap,
                    "retry_used": True
                }
            else:
                self.logger.info("🚫 RETRY FAILED: Sem melhoria significativa")
                return {
                    "answerable": False,
                    "reason": "retry_failed",
                    "message": "Não encontrei informações suficientemente relevantes sobre isso. Você poderia reformular sua pergunta?",
                    "should_escalate": False,
                    "retry_used": True
                }
                
        except Exception as e:
            self.logger.error(f"Error in retry: {e}")
            return {
                "answerable": False,
                "reason": "retry_error",
                "message": "Não encontrei informações específicas sobre isso na base do conhecimento do Jota. Posso te ajudar com outra dúvida?",
                "should_escalate": False
            }
    
    async def _execute_agent_specialist(self, message: AgentMessage, rag_result: Optional[RAGResult], 
                                     client_context, agent_type: str) -> AgentDecision:
        """Executa agente especialista usando classes existentes"""
        try:
            self.logger.info(f"🔧 Executing agent: {agent_type}")
            
            # DEFINIR VARIÁVEIS NECESSÁRIAS NO INÍCIO
            false_negative_detected = False
            false_negative_prevented = False
            
            # Importar agente especialista CORRETO
            if agent_type == "atendimento_geral":
                from support_agent.agents.atendimento_geral import OptimizedAgentAtendimentoGeral
                agent_class = OptimizedAgentAtendimentoGeral
            elif agent_type == "golpe_med":
                from support_agent.agents.golpe_med import AgentGolpeMed
                agent_class = AgentGolpeMed
            elif agent_type == "criacao_conta":
                from support_agent.agents.criacao_conta import AgentCriacaoConta
                agent_class = AgentCriacaoConta
            elif agent_type == "open_finance":
                from support_agent.agents.open_finance import AgentOpenFinance
                agent_class = AgentOpenFinance
            else:
                # Agente não encontrado - erro
                self.logger.error(f"❌ Agent {agent_type} not found")
                return self._create_error_decision(f"Agente {agent_type} não encontrado")
            
            self.logger.info(f"✅ Agent class imported: {agent_class.__name__}")
            
            # Inicializar agente especialista
            agent = agent_class(
                rag_system=self.rag_system,
                llm_manager=self.llm_manager,
                prompt_manager=self.prompt_manager
            )
            
            self.logger.info(f"✅ Agent initialized: {agent_type}")
            
            # Preparar contexto para o agente
            agent_context = {
                "user_name": getattr(client_context, 'user_name', 'Cliente') if client_context else 'Cliente',
                "message_history": getattr(client_context, 'message_history', []) if client_context else [],
                "user_id": message.user_id,
                "session_id": message.session_id,
                "_query_norm": _normalize_query_text(message.content),
            }
            
            # Injetar RAG result do orchestrator para evitar query duplicada no agent
            if rag_result and rag_result.documents:
                rag_payload = {
                    "query": rag_result.query.query if rag_result.query else message.content,
                    "top_k": rag_result.query.top_k if rag_result.query else 8,
                    "kb_version": _get_kb_version(),
                    "rag_used": True,
                    "chunks": [
                        {
                            "content": doc.content,
                            "source_id": doc.doc_id,
                            "title": doc.metadata.get("title"),
                            "domain": doc.metadata.get("domain"),
                            "score": doc.score,
                            "chunk_id": doc.chunk_id,
                            "metadata": {k: v for k, v in doc.metadata.items()
                                         if k not in ("embedding",)},
                        }
                        for doc in rag_result.documents
                    ],
                    "confidence": rag_result.confidence,
                    "processing_time": rag_result.processing_time,
                    "source": rag_result.source,
                    "_meta": {
                        "query_norm": _normalize_query_text(message.content),
                        "top_k": rag_result.query.top_k if rag_result.query else 8,
                        "kb_version": _get_kb_version(),
                        "embedding_signature": self._get_rag_signature(),
                        "created_at_ms": int(time.time() * 1000),
                    },
                }
                agent_context["_orchestrator_rag_v1"] = rag_payload
                self.logger.debug(
                    "rag_payload_injected",
                    extra={
                        "chunks_count": len(rag_payload["chunks"]),
                        "top_score": rag_payload["chunks"][0]["score"] if rag_payload["chunks"] else 0,
                    }
                )
            
            self.logger.info(f"🔧 Calling agent.process_message for: {message.content[:50]}...")
            
            # Executar agente especialista
            agent_response = await agent.process_message(message.content, agent_context)
            
            self.logger.info(f"✅ Agent response received: {type(agent_response)}")
            self.logger.info(f"🔍 Agent response content: {str(agent_response)[:200]}...")
            
            # 🆕 PARTE 1: Log detalhado do evidence_pack do agente
            if hasattr(agent_response, 'evidence_pack') and agent_response.evidence_pack:
                evidence_pack_from_agent = agent_response.evidence_pack
                if isinstance(evidence_pack_from_agent, dict):
                    docs_count = len(evidence_pack_from_agent.get('documents', []))
                    citations = evidence_pack_from_agent.get('citations', [])
                    self.logger.info(f"📚 AGENT EVIDENCE_PACK: docs={docs_count}, citations={citations}")
                    self.logger.info(f"📚 AGENT EVIDENCE_PACK TYPE: {type(evidence_pack_from_agent)}")
                else:
                    self.logger.info(f"📚 AGENT EVIDENCE_PACK: {type(evidence_pack_from_agent)} - {str(evidence_pack_from_agent)[:100]}")
            else:
                self.logger.warning(f"⚠️ AGENT NO EVIDENCE_PACK: {type(agent_response)}")
            
            # 🆕 PATCH CIRÚRGICO: Evitar re-empacotamento desnecessário
            if isinstance(agent_response, AgentDecision):
                # ✅ Agent já retornou AgentDecision - apenas normalizar
                decision = agent_response
                
                # Normalizações obrigatórias
                decision.agent_type = agent_type  # Garantir agent_type correto
                
                # Normalizar trace
                if not hasattr(decision, 'trace') or decision.trace is None:
                    decision.trace = {}
                elif not isinstance(decision.trace, dict):
                    decision.trace = dict(decision.trace) if decision.trace else {}
                
                # 🆕 PARTE 1: Preservar evidence_pack original do agente
                if hasattr(agent_response, 'evidence_pack') and agent_response.evidence_pack:
                    decision.evidence_pack = agent_response.evidence_pack
                    self.logger.info(f"🔄 PRESERVED EVIDENCE_PACK from agent: {type(decision.evidence_pack)}")
                elif not hasattr(decision, 'evidence_pack') or decision.evidence_pack is None:
                    decision.evidence_pack = {}
                    self.logger.warning(f"⚠️ CREATED EMPTY EVIDENCE_PACK (agent had none)")
                elif not isinstance(decision.evidence_pack, dict):
                    try:
                        decision.evidence_pack = dict(decision.evidence_pack)
                        self.logger.info(f"🔄 CONVERTED EVIDENCE_PACK to dict")
                    except Exception:
                        decision.evidence_pack = {}
                        self.logger.warning(f"⚠️ FAILED TO CONVERT EVIDENCE_PACK, using empty dict")
                
                # Preservar rag_used do agente, só inferir se não existir
                if not hasattr(decision, 'rag_used'):
                    decision.rag_used = rag_result is not None and len(rag_result.documents) > 0
                
                # Aceitar ambos nomes de escalonamento
                if hasattr(decision, 'needs_escalation') and not hasattr(decision, 'should_escalate'):
                    decision.should_escalate = decision.needs_escalation
                elif not hasattr(decision, 'should_escalate'):
                    decision.should_escalate = False
                
                self.logger.info(f"🔄 Using original AgentDecision from agent: {agent_type}")
                
                # 🆕 PARTE 1: Log final do evidence_pack no orquestrador
                if hasattr(decision, 'evidence_pack') and decision.evidence_pack:
                    final_evidence = decision.evidence_pack
                    if isinstance(final_evidence, dict):
                        final_docs = len(final_evidence.get('documents', []))
                        final_citations = final_evidence.get('citations', [])
                        self.logger.info(f"📚 FINAL EVIDENCE_PACK: docs={final_docs}, citations={final_citations}")
                    else:
                        self.logger.info(f"📚 FINAL EVIDENCE_PACK: {type(final_evidence)}")
                else:
                    self.logger.warning(f"⚠️ FINAL NO EVIDENCE_PACK")
                
            else:
                # Fallback para resposta não-AgentDecision (compatibilidade)
                self.logger.warning(f"⚠️ Agent returned non-AgentDecision: {type(agent_response)} - adapting")
                
                # Tentar extrair campos de forma compatível
                response_text = getattr(agent_response, 'response', str(agent_response))
                confidence = getattr(agent_response, 'confidence', 0.5)
                needs_escalation = getattr(agent_response, 'needs_escalation', getattr(agent_response, 'should_escalate', False))
                reasoning = getattr(agent_response, 'reasoning', 'Adapted from non-AgentDecision response')
                
                # Criar AgentDecision adaptado
                decision = AgentDecision(
                    action=AgentAction.RESPOND if not needs_escalation else AgentAction.ESCALATE,
                    response=response_text,
                    confidence=confidence,
                    agent_type=agent_type,
                    reasoning=reasoning,
                    processing_time=getattr(agent_response, 'processing_time', 0.001),
                    rag_used=rag_result is not None and len(rag_result.documents) > 0,
                    should_escalate=needs_escalation,
                    escalation_reason=getattr(agent_response, 'escalation_reason', ''),
                    evidence_pack=getattr(agent_response, 'evidence_pack', {}),
                    trace=getattr(agent_response, 'trace', {})
                )
            
            # 🆕 PARTE 2: Grounding Obrigatório e Mensurável com Thresholds Ajustados
            grounding_result = await self._verify_grounding_systematic(decision, rag_result, agent_type)
            
            # 🆕 PARTE 3: Thresholds ajustados - soft-signals em vez de hard-fails
            rag_docs_count = len(rag_result.documents) if rag_result and rag_result.documents else 0
            
            # Se há RAG docs mas nenhuma citação, reduzir confidence (soft-signal)
            if (rag_docs_count > 0 and grounding_result["citation_coverage"] == 0):
                decision.confidence = max(decision.confidence * 0.6, 0.2)
                self.logger.warning(f"🔽 Confidence reduced: no citations with {rag_docs_count} RAG docs")
            
            # Se grounding muito baixo com evidência, reduzir confidence (soft-signal)
            elif (rag_docs_count > 0 and grounding_result["grounding_score"] < 0.4):
                decision.confidence = max(decision.confidence * 0.7, 0.25)
                self.logger.warning(f"🔽 Confidence reduced: low grounding score {grounding_result['grounding_score']:.2f}")
            
            # Grounding failure: log structured event but do NOT override should_escalate
            # (escalation must come from deterministic structured signals, not LLM text parsing)
            if grounding_result["citation_coverage"] < 0.1 and rag_docs_count > 0:
                decision.confidence = max(decision.confidence * 0.5, 0.2)
                self.logger.info(
                    f"[ESCALATION_DECISION] event=grounding_escalation_suppressed "
                    f"agent_type={agent_type} citation_coverage={grounding_result['citation_coverage']:.2f} "
                    f"rag_docs={rag_docs_count} action=confidence_reduced_only"
                )
            
            # 🆕 Validar escalation_reason centralizadamente
            decision = self._validate_escalation_reason(decision)
            
            # Atualizar campos de grounding na decision
            decision.grounded = grounding_result["grounded"]
            decision.grounding_score = grounding_result["grounding_score"]
            decision.citation_coverage = grounding_result["citation_coverage"]
            
            # 🆕 PARTE 4: OTIMIZAÇÃO - Desabilitar regenerações automáticas para performance
            regeneration_count = 0
            max_regenerations = 0  # 🆕 OTIMIZAÇÃO: Desabilitar regenerações para reduzir latência
            
            # 🆕 PARTE 1: Gatilhos rebalanceados para regeneração
            rag_docs_count = len(rag_result.documents) if rag_result and rag_result.documents else 0
            should_regenerate = False
            regeneration_reason = ""
            
            # Verificar se existe pelo menos um padrão [C#] na resposta
            has_citations = bool(re.search(r'\[C\d+\]', decision.response))
            
            # 🆕 PARTE 4: Ajustar condição de regeneração - só se rag_docs_count > 0
            if (rag_docs_count > 0 and not has_citations):
                # 🆕 PARTE 4: Política por agente
                if agent_type in ["criacao_conta", "golpe_med", "open_finance"]:
                    # Para estes agentes, sempre regenerar se não há citação com RAG
                    should_regenerate = True
                    regeneration_reason = f"missing_citations_with_rag_docs_{rag_docs_count}"
                    self.logger.warning(f"🔄 REGENERATION TRIGGER: No [C#] found with {rag_docs_count} RAG docs (agent: {agent_type})")
                elif agent_type == "atendimento_geral":
                    # Para atendimento_geral, manter política atual (mais flexível)
                    if grounding_result["citation_coverage"] == 0:
                        should_regenerate = True
                        regeneration_reason = f"missing_citations_with_rag_docs_{rag_docs_count}"
                        self.logger.warning(f"🔄 REGENERATION TRIGGER: No [C#] found with {rag_docs_count} RAG docs (agent: {agent_type})")
                    else:
                        self.logger.info(f"🚫 SKIPPING REGENERATION: atendimento_geral has some citations")
                else:
                    should_regenerate = True
                    regeneration_reason = f"missing_citations_with_rag_docs_{rag_docs_count}"
                    self.logger.warning(f"🔄 REGENERATION TRIGGER: No [C#] found with {rag_docs_count} RAG docs (agent: {agent_type})")
            
            # 🆕 GATILHO 2: Grounding False com evidência forte
            elif (not grounding_result["grounded"] and rag_docs_count > 0):
                should_regenerate = True
                regeneration_reason = f"grounded_false_with_rag_docs_{rag_docs_count}"
                self.logger.warning(f"🔄 REGENERATION TRIGGER: grounded=False with {rag_docs_count} RAG docs")
            
            # 🆕 GATILHO 3: Falsos negativos com evidência forte (PARTE 2)
            elif (false_negative_detected and rag_docs_count > 0):
                # Verificar se há evidência forte via similarity
                similarity_avg = getattr(decision, 'similarity_avg', 0.0)
                top_similarity = getattr(decision, 'top_similarity', 0.0)
                
                # 🆕 PARTE 3: Ajustar false negative threshold de 0.05 para 0.25
                if similarity_avg >= 0.25: 
                    if agent_type == "golpe_med":
                        threshold_strong = 0.3  # Threshold mais baixo para golpe_med
                    else:
                        threshold_strong = 0.4  # Threshold padrão para outros agentes
                    
                if similarity_avg >= threshold_strong or top_similarity >= threshold_strong:
                    should_regenerate = True
                    regeneration_reason = f"false_negative_with_strong_evidence_sim_{similarity_avg:.2f}"
                    self.logger.warning(f"🔄 REGENERATION TRIGGER: False negative with strong evidence (sim={similarity_avg:.2f}, agent: {agent_type})")
                else:
                    self.logger.info(f"🚫 SKIPPING REGENERATION: False negative with weak evidence (sim={similarity_avg:.2f}, agent: {agent_type})")
            else:
                self.logger.info(f"🚫 SKIPPING REGENERATION: No trigger met (has_citations={has_citations}, rag_docs={rag_docs_count})")
            
            # 🆕 PARTE 3: Thresholds de aceitação ajustados
            # Não usar thresholds hard-fail, apenas soft-signals para confidence
            # citation_coverage >= 0.2 é aceitável (não bloqueia)
            # grounding_score >= 0.4 é aceitável (não bloqueia)
            
            while (regeneration_count < max_regenerations and should_regenerate):
                
                regeneration_count += 1
                self.logger.warning(f"🔄 REGENERATION {regeneration_count}/{max_regenerations}: "
                                  f"coverage={grounding_result['citation_coverage']:.2f}, "
                                  f"grounded={grounding_result['grounded']}, "
                                  f"false_negative={false_negative_detected}")
                
                # 🆕 PARTE 4: Instrução específica por gatilho
                if regeneration_reason.startswith("missing_citations"):
                    instruction = ("Responda utilizando exclusivamente os documentos do contexto e inclua citações [C#] "
                                 "para cada afirmação factual. É OBRIGATÓRIO usar pelo menos uma citação.")
                elif regeneration_reason.startswith("false_negative"):
                    instruction = ("Há evidências relevantes no contexto. Responda usando exclusivamente o contexto "
                                 "e inclua citações [C#]. NÃO responda que não encontrou informações.")
                else:  # grounded_false
                    instruction = ("Utilize exclusivamente os documentos do contexto e inclua citações [C#] "
                                 "para cada afirmação factual.")
                
                decision = await self._regenerate_with_context_instruction(
                    decision, rag_result, agent_type, instruction
                )
                
                # Recalcular grounding após regeneração
                grounding_result = await self._verify_grounding_systematic(decision, rag_result, agent_type)
                
                # Atualizar campos
                decision.grounded = grounding_result["grounded"]
                decision.grounding_score = grounding_result["grounding_score"]
                decision.citation_coverage = grounding_result["citation_coverage"]
                
                # Se atingiu máximo de regenerações (1), avaliar se aplica fallback
                if regeneration_count >= max_regenerations:
                    self.logger.warning(f"🔽 Max regenerations reached ({max_regenerations}) - evaluating fallback")
                    
                    # 🆕 PARTE 2: Desativar SAFE FALLBACK durante avaliação A/B
                    # Permitir que o modelo final seja medido mesmo sem citações
                    if grounding_result["citation_coverage"] < 0.1 and not grounding_result["grounded"]:
                        # 🚨 SAFE FALLBACK DESATIVADO PARA A/B - manter resposta original
                        decision.confidence = max(decision.confidence * 0.8, 0.3)  # Reduzir confidence levemente
                        # 🆕 PARTE 2: NÃO substituir resposta durante A/B
                        # decision.response = "Não encontrei informações específicas nos documentos disponíveis."
                        # decision.reasoning = "Regeneration failed - safe fallback applied"
                        self.logger.warning(f"🚨 SAFE FALLBACK DESATIVADO: citation_coverage={grounding_result['citation_coverage']:.2f}, grounded={grounding_result['grounded']} - mantendo resposta original")
                    else:
                        # 🆕 PARTE 2: Permitir resposta original se não for crítico
                        self.logger.info(f"✅ KEEPING ORIGINAL RESPONSE: citation_coverage={grounding_result['citation_coverage']:.2f}, grounded={grounding_result['grounded']}")
                        # Manter resposta original, apenas reduzir confidence levemente
                        if grounding_result["citation_coverage"] < 0.5:
                            decision.confidence = max(decision.confidence * 0.8, 0.3)
                        
                        # Considerar escalonamento apenas se grounding muito ruim
                        if grounding_result["citation_coverage"] < 0.1:
                            decision.should_escalate = True
                            decision.escalation_reason = "Critical grounding failure after regeneration"
                            
                            # 🆕 Validar escalation_reason centralizadamente
                            decision = self._validate_escalation_reason(decision)
            
            # 🆕 Adicionar métricas de regeneração ao trace
            if hasattr(decision, 'trace') and decision.trace:
                decision.trace["regeneration_count"] = regeneration_count
                decision.trace["max_regenerations_reached"] = regeneration_count >= max_regenerations
            
            # 🆕 Adicionar métricas de falsos negativos ao trace
            if hasattr(decision, 'trace') and decision.trace:
                decision.trace["false_negative_detected"] = false_negative_detected
                decision.trace["false_negative_prevented"] = false_negative_prevented
            
            # 🆕 PARTE 3: Log final antes de retornar
            self.logger.info(f"🏁 FINAL EVIDENCE_PACK: {decision.evidence_pack}")
            self.logger.info(f"🏁 FINAL MODEL_USED: {decision.trace.get('model_used') if hasattr(decision, 'trace') and decision.trace else 'NO_TRACE'}")
            self.logger.info(f"🏁 FINAL MODEL_TYPE: {decision.trace.get('model_tier') if hasattr(decision, 'trace') and decision.trace else 'NO_TRACE'}")
            
            # 🆕 CALIBRAÇÃO DE FALLBACK PESSIMISTA (movida para o final - antes do return)
            self.logger.info(f"🔍 INICIANDO CALIBRAÇÃO DE FALLBACK PESSIMISTA")
            
            evidence_available = False
            fallback_pessimista_applied = False
            
            # 🆕 DEFINIR negative_patterns
            negative_patterns = ["não encontrei", "não há informação", "não disponível", "não sei"]
            
            # Verificar disponibilidade de evidências em múltiplas fontes
            if hasattr(decision, 'evidence_pack') and decision.evidence_pack:
                evidence_pack = decision.evidence_pack
                if isinstance(evidence_pack, dict):
                    citations_count = len(evidence_pack.get('citations', []))
                    chunks_count = len(evidence_pack.get('chunks_used', []))
                    docs_count = len(evidence_pack.get('documents', []))
                    
                    # Considerar evidência disponível se qualquer fonte tiver conteúdo
                    evidence_available = citations_count > 0 or chunks_count > 0 or docs_count > 0
                    
                    self.logger.info(f"🔍 EVIDENCE_CHECK: citations={citations_count}, chunks={chunks_count}, docs={docs_count}, available={evidence_available}")
            
            # Verificar também rag_result diretamente
            if not evidence_available and rag_result and len(rag_result.documents) > 0:
                evidence_available = True
                self.logger.info(f"🔍 EVIDENCE_CHECK: rag_docs={len(rag_result.documents)}, available={evidence_available}")
            
            # 🆕 REGRA: Só aplicar fallback pessimista se NÃO há evidências disponíveis
            is_negative_response = any(pattern in decision.response.lower() for pattern in negative_patterns)
            
            self.logger.info(f"🔍 FALLBACK_CHECK: is_negative={is_negative_response}, evidence_available={evidence_available}")
            
            if is_negative_response and evidence_available:
                # Há evidências mas resposta é negativa -> possível falso negativo
                false_negative_detected = True
                fallback_pessimista_applied = True
                
                self.logger.warning(f"🚨 FALLBACK PESSIMISTA BLOQUEADO: evidências disponíveis mas resposta negativa")
                self.logger.info(f"🔄 RESPOSTA ORIGINAL: {decision.response[:100]}...")
                
                # 🆕 CORREÇÃO: Como já estamos no final, não podemos regenerar.
                # Apenas aplicar penalidade e marcar que o fallback foi bloqueado
                decision.confidence = max(decision.confidence * 0.3, 0.1)
                decision.reasoning = "Fallback pessimista bloqueado - evidências disponíveis mas resposta negativa"
                
                self.logger.warning("🚫 CALIBRAÇÃO APLICADA: penalidade aplicada (não há como regenerar no final)")
                
            elif is_negative_response and not evidence_available:
                # Sem evidências e resposta negativa -> fallback aceitável
                fallback_pessimista_applied = True
                self.logger.info(f"✅ FALLBACK PESSIMISTA ACEITO: sem evidências disponíveis")
                decision.reasoning = "Fallback pessimista aplicado - sem evidências disponíveis"
            
            # 🆕 MANTER COMPATIBILIDADE: Adicionar métricas ao trace
            if not hasattr(decision, 'trace') or decision.trace is None:
                decision.trace = {}
            
            decision.trace['fallback_pessimista_applied'] = fallback_pessimista_applied
            decision.trace['evidence_available'] = evidence_available
            decision.trace['false_negative_detected'] = false_negative_detected
            
            # Log estruturado final
            self.logger.info(
                "agent_specialist_executed",
                extra={
                    "event": "agent_specialist_executed",
                    "agent_type": agent_type,
                    "returned_type": type(agent_response).__name__,
                    "rag_used": decision.rag_used,
                    "should_escalate": decision.should_escalate,
                    "has_evidence_pack": bool(decision.evidence_pack),
                    "has_trace": bool(decision.trace),
                    "confidence": decision.confidence,
                    "processing_time": getattr(decision, 'processing_time', None),
                    # 🆕 Grounding metrics
                    "grounded": decision.grounded,
                    "grounding_score": decision.grounding_score,
                    "citation_coverage": decision.citation_coverage,
                    # 🆕 False negative metrics
                    "false_negative_detected": false_negative_detected,
                    "false_negative_prevented": false_negative_prevented,
                    # 🆕 Regeneration metrics
                    "regeneration_count": regeneration_count,
                    "max_regenerations_reached": regeneration_count >= max_regenerations
                }
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error executing agent {agent_type}: {e}")
            # Fallback para erro
            return self._create_error_decision(str(e))
    
    async def _regenerate_with_context_instruction(self, decision: AgentDecision, rag_result, agent_type: str, instruction: str) -> AgentDecision:
        """
        Regenera resposta com instrução explícita para usar contexto
        """
        try:
            # 🆕 PARTE 2: Obter citações disponíveis para bloqueio real
            available_citations = []
            if decision.evidence_pack and "citations" in decision.evidence_pack:
                available_citations = decision.evidence_pack["citations"]
            
            # Obter o agente especialista para regeneração
            agent = self._get_agent_instance(agent_type)
            
            # 🆕 PARTE 4: Construir prompt com lista explícita e frase mandatória
            context_prompt = f"""
{instruction}

Citações disponíveis: {', '.join(available_citations) if available_citations else 'Nenhuma'}
Use obrigatoriamente uma dessas citações. Respostas sem [C#] serão descartadas.

CONTEXTO DISPONÍVEL:
"""
            
            # Adicionar documentos do RAG ao prompt
            if rag_result and rag_result.documents:
                for i, doc in enumerate(rag_result.documents, 1):
                    context_prompt += f"\nDocumento {i}: {doc.content.strip()}\n"
            
            context_prompt += f"\nResponda utilizando APENAS as informações acima e incluindo citações [C#] para cada afirmação factual."
            
            # Chamar o agente com contexto reforçado
            agent_message = AgentMessage(
                content=decision.response,  # Usar resposta original como base
                user_id="regeneration",
                session_id="regeneration",
                timestamp=datetime.now(),
                context={"regeneration_instruction": instruction}
            )
            
            # Tentar regenerar usando o mesmo agente
            regenerated_response = await agent.process_message(agent_message.content, agent_message.context)
            
            # 🆕 PARTE 2: Bloqueio real - verificar se há citações após regeneração
            if isinstance(regenerated_response, AgentDecision):
                response_text = regenerated_response.response
            else:
                response_text = str(regenerated_response)
            citation_pattern = r'\[C(\d+)\]'
            found_citations = re.findall(citation_pattern, response_text)
            
            # 🆕 BLOQUEIO: Se ainda não houver citações, forçar fallback seguro
            if not found_citations and available_citations:
                self.logger.warning(f"🚫 REGENERATION FAILED: No citations found after regeneration. Forcing safe fallback.")
                
                # Fallback seguro
                safe_response = "Não encontrei evidências suficientes nos documentos disponíveis."
                
                # Criar decision com fallback
                fallback_decision = AgentDecision(
                    action=AgentAction.RESPOND,
                    response=safe_response,
                    confidence=decision.confidence * 0.5,  # Reduzir confidence significativamente
                    agent_type=agent_type,
                    reasoning="Regeneration failed - no citations provided, using safe fallback",
                    processing_time=decision.processing_time,
                    rag_used=decision.rag_used,
                    should_escalate=decision.should_escalate,  # Manter política original
                    escalation_reason=decision.escalation_reason,
                    evidence_pack=decision.evidence_pack,
                    trace=decision.trace
                )
                
                # 🆕 Avaliar escalonamento para fallback
                if hasattr(decision, 'risk_level') and decision.risk_level in ['MEDIUM', 'HIGH']:
                    fallback_decision.should_escalate = True
                    fallback_decision.escalation_reason = f"High risk ({decision.risk_level}) + regeneration failure"
                    
                    # 🆕 Validar escalation_reason centralizadamente
                    fallback_decision = self._validate_escalation_reason(fallback_decision)
                
                return fallback_decision
            
            # Se regeneração funcionou, usar resposta
            if isinstance(regenerated_response, AgentDecision):
                regenerated_response.agent_type = agent_type
                regenerated_response.reasoning = f"Regenerated due to false negative: {instruction}"
                return regenerated_response
            
            # Se for string, criar AgentDecision
            return AgentDecision(
                action=AgentAction.RESPOND,
                response=response_text,
                confidence=decision.confidence * 0.9,  # Reduzir confidence um pouco
                agent_type=agent_type,
                reasoning=f"Regenerated due to false negative: {instruction}",
                processing_time=decision.processing_time,
                rag_used=decision.rag_used,
                should_escalate=decision.should_escalate,
                escalation_reason=decision.escalation_reason,
                evidence_pack=decision.evidence_pack,
                trace=decision.trace
            )
            
        except Exception as e:
            self.logger.error(f"Error in regeneration: {e}")
            # Fallback para decision original
            return decision
    
    def _get_agent_instance(self, agent_type: str):
        """
        Obtém instância do agente especialista para regeneração
        """
        from support_agent.agents.atendimento_geral import OptimizedAgentAtendimentoGeral
        from support_agent.agents.criacao_conta import AgentCriacaoConta
        from support_agent.agents.open_finance import AgentOpenFinance
        from support_agent.agents.golpe_med import AgentGolpeMed
        
        agent_map = {
            "atendimento_geral": OptimizedAgentAtendimentoGeral,
            "criacao_conta": AgentCriacaoConta,
            "open_finance": AgentOpenFinance,
            "golpe_med": AgentGolpeMed
        }
        
        agent_class = agent_map.get(agent_type)
        if agent_class:
            return agent_class(
                rag_system=self.rag_system,
                llm_manager=self.llm_manager,
                prompt_manager=self.prompt_manager
            )
        
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    async def _verify_grounding_systematic(self, decision: AgentDecision, rag_result: Optional[RAGResult], agent_type: str) -> Dict[str, Any]:
        """
        Verificação sistemática de grounding obrigatória para todas as respostas
        """
        try:
            # 1. Verificar presença de citações [C#]
            citation_pattern = r'\[C(\d+)\]'
            found_citations = re.findall(citation_pattern, decision.response)
            
            # 2. Obter citações disponíveis no evidence_pack
            available_citations = []
            if decision.evidence_pack and "citations" in decision.evidence_pack:
                available_citations = decision.evidence_pack["citations"]
            
            # 3. Validar IDs das citações
            available_ids = []
            for citation in available_citations:
                match = re.search(r'C(\d+)', citation)
                if match:
                    available_ids.append(match.group(1))
            
            # 4. Verificar validade das citações
            invalid_citations = [c for c in found_citations if c not in available_ids]
            valid_citations = [c for c in found_citations if c in available_ids]
            
            # 5. Calcular métricas de grounding
            total_sentences = len([s for s in decision.response.split('.') if s.strip()])
            factual_sentences = 0
            cited_sentences = 0
            
            # 🆕 CORREÇÃO: Tratar citações que podem estar em sentenças separadas
            sentences = decision.response.split('.')
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                next_sentence = sentences[i + 1].strip() if i + 1 < len(sentences) else ""
                
                if len(sentence) > 15:  # Reduzido de 20 para 15 para capturar mais sentenças
                    factual_sentences += 1
                    
                    # Verificar se a sentença atual tem citação
                    if re.search(citation_pattern, sentence):
                        cited_sentences += 1
                    # 🆕 CORREÇÃO: Verificar se a próxima sentença é apenas uma citação
                    elif re.fullmatch(r'\[C\d+\]', next_sentence):
                        cited_sentences += 1
            
            # 6. Calcular scores com proteções
            if factual_sentences == 0:
                citation_coverage = 0.0
            else:
                citation_coverage = cited_sentences / factual_sentences
                
                # 🆕 CORREÇÃO: Proteger contra valores inválidos
                if citation_coverage > 1.0:
                    self.logger.error(f"🚨 CITATION_COVERAGE BUG: {citation_coverage} > 1.0 - factual={factual_sentences}, cited={cited_sentences}")
                    citation_coverage = 1.0  # Limitar ao máximo
                elif citation_coverage < 0.0:
                    self.logger.error(f"🚨 CITATION_COVERAGE BUG: {citation_coverage} < 0.0 - factual={factual_sentences}, cited={cited_sentences}")
                    citation_coverage = 0.0
            
            # 7. Verificar se há afirmações factuais sem citação
            has_uncited_factual = factual_sentences > cited_sentences
            has_invalid_citations = len(invalid_citations) > 0
            has_valid_citations = len(valid_citations) > 0
            
            # 8. Critérios de grounded
            grounded = (
                not has_invalid_citations and
                not has_uncited_factual and
                citation_coverage >= 0.5  # Mínimo 50% de cobertura
            )
            
            # 9. Calcular grounding score (0-1)
            grounding_score = 0.0
            if not has_invalid_citations:
                grounding_score += 0.4  # Sem citações inválidas
            if not has_uncited_factual:
                grounding_score += 0.3  # Sem afirmações sem citação
            grounding_score += citation_coverage * 0.3  # Cobertura de citações
            
            # 10. Log detalhado
            self.logger.info(f"🔍 SYSTEMATIC GROUNDING CHECK:")
            self.logger.info(f"   - Found citations: {found_citations}")
            self.logger.info(f"   - Available IDs: {available_ids}")
            self.logger.info(f"   - Valid citations: {valid_citations}")
            self.logger.info(f"   - Invalid citations: {invalid_citations}")
            self.logger.info(f"   - Factual sentences: {factual_sentences}")
            self.logger.info(f"   - Cited sentences: {cited_sentences}")
            self.logger.info(f"   - Citation coverage: {citation_coverage:.2f}")
            self.logger.info(f"   - Grounding score: {grounding_score:.2f}")
            self.logger.info(f"   - Grounded: {grounded}")
            
            return {
                "grounded": grounded,
                "grounding_score": grounding_score,
                "citation_coverage": citation_coverage,
                "found_citations": found_citations,
                "valid_citations": valid_citations,
                "invalid_citations": invalid_citations,
                "factual_sentences": factual_sentences,
                "cited_sentences": cited_sentences,
                "has_uncited_factual": has_uncited_factual,
                "has_invalid_citations": has_invalid_citations
            }
            
        except Exception as e:
            self.logger.error(f"Error in systematic grounding verification: {e}")
            # Fallback conservador
            return {
                "grounded": False,
                "grounding_score": 0.0,
                "citation_coverage": 0.0,
                "error": str(e)
            }
    
    def _check_decision_cache(self, cache_key: str) -> Optional[AgentDecision]:
        """Verifica cache de decisões com TTL otimizado"""
        cached_value = self.decision_cache.get(cache_key)
        
        if not cached_value:
            return None
        
        # 🆕 CORREÇÃO: CacheStore retorna o valor diretamente, não dict
        if isinstance(cached_value, AgentDecision):
            # Formato do CacheStore (valor direto)
            cached_decision = cached_value
            
            # 🆕 OTIMIZAÇÃO: Verificar confiança mínima para reuso
            if hasattr(cached_decision, 'confidence') and cached_decision.confidence < 0.3:
                self.logger.info(f"🚫 CACHE BLOQUEADO: Confiança muito baixa ({cached_decision.confidence})")
                return None
            
            # 🆕 DIAGNÓSTICO: Log cache hit
            self.logger.info(
                "cache_hit",
                extra={
                    "event": "cache_hit",
                    "cache_name": "decision_cache",
                    "key_hash": cache_key[:8] if len(cache_key) > 8 else cache_key,
                    "hit": True,
                    "store_type": "singleton",
                    "store_id": id(self.decision_cache),
                    "store_size": self.decision_cache.size(),
                    "ttl_s": "default"
                }
            )
            
            return cached_decision
        
        # Formato antigo (legado) - não deve mais ocorrer
        self.logger.warning("🔄 LEGACY CACHE FORMAT: Atualizando...")
        # Converter para novo formato ou descartar
        return None
    
    async def _check_rag_cache(self, query: str) -> Optional[RAGResult]:
        """Verifica cache RAG"""
        cache_key, key_hash = self._make_rag_cache_key(query)
        cached = self._rag_cache.get(cache_key)
        if cached:
            self.logger.info(
                "cache_hit",
                extra={
                    "event": "cache_hit",
                    "cache_name": "rag_cache",
                    "key_hash": key_hash,
                    "hit": True,
                    "store_id": id(self._rag_cache),
                    "store_size": self._rag_cache.size(),
                    "ttl_s": 300
                }
            )
            return cached
        return None
    
    async def _query_rag_optimized(self, message: AgentMessage, agent_type: str) -> RAGResult:
        """RAG otimizado com cache e top_k dinâmico"""
        try:
            cache_key, key_hash = self._make_rag_cache_key(message.content)
            
            # Verificar cache de RAG (TTL de 5 minutos)
            cached_rag = self._rag_cache.get(cache_key)
            if cached_rag:
                # 🆕 DIAGNÓSTICO: Log cache hit
                self.logger.info(
                    "cache_hit",
                    extra={
                        "event": "cache_hit",
                        "cache_name": "rag_cache",
                        "key_hash": key_hash,
                        "hit": True,
                        "store_type": "singleton",
                        "store_id": id(self._rag_cache),
                        "store_size": self._rag_cache.size(),
                        "ttl_s": 300
                    }
                )
                return cached_rag
            
            # 🆕 OTIMIZAÇÃO: top_k dinâmico baseado no agente
            top_k = 2 if agent_type in ["golpe_med", "criacao_conta"] else 3
            
            rag_query = RAGQuery(
                query=message.content,
                agent_type=agent_type,
                user_context=message.context,
                top_k=top_k,  # Reduzido para queries mais rápidas
                filters={"agent_type": agent_type}
            )
            
            step_start_time = datetime.now()
            result = await self.rag_system.query(rag_query)
            rag_elapsed_ms = (datetime.now() - step_start_time).total_seconds() * 1000
            self.metrics["rag_queries"] += 1
            
            # 🆕 FASE 2: Validação de sanidade do RAG result
            if result is None:
                self.logger.warning("🚫 RAG result is None - creating empty result")
                result = RAGResult(
                    documents=[],
                    query=rag_query,
                    confidence=0.0,
                    processing_time=0.0,
                    source="empty_fallback"
                )
            elif not hasattr(result, 'documents') or result.documents is None:
                self.logger.warning("🚫 RAG result has no documents - creating empty list")
                result.documents = []
            elif not isinstance(result.documents, list):
                self.logger.warning(f"🚫 RAG documents is not list: {type(result.documents)} - converting")
                result.documents = list(result.documents) if result.documents else []
            
            # 🆕 OTIMIZAÇÃO: Cache resultado se tiver documentos
            if result and result.documents:
                self._rag_cache.set(cache_key, result, ttl_seconds=300)
                self.logger.info(
                    "cache_set",
                    extra={
                        "event": "cache_set",
                        "cache_name": "rag_cache",
                        "key_hash": key_hash,
                        "hit": False,
                        "store_type": "singleton",
                        "store_id": id(self._rag_cache),
                        "store_size": self._rag_cache.size(),
                        "ttl_s": 300
                    }
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error querying optimized RAG: {e}")
            return RAGResult(
                documents=[],
                query=rag_query if 'rag_query' in locals() else RAGQuery(
                    query=message.content,
                    agent_type=agent_type,
                    user_context=message.context,
                    top_k=3,
                    filters={"agent_type": agent_type}
                ),
                confidence=0.0,
                processing_time=0.0,
                source="error"
            )
    
    async def _save_cache_async(self, cache_key: str, decision: AgentDecision, rag_result: Optional[RAGResult]):
        """Cache inteligente: só cachear respostas com evidência válida e confidence alta"""
        try:
            # Critérios para cache inteligente
            should_cache = True
            cache_reason = []
            
            # 1. Verificar confidence >= 0.5 (relaxado de 0.7)
            if decision.confidence < 0.5:
                should_cache = False
                cache_reason.append(f"confidence_baixo_{decision.confidence:.2f}")
            
            # 2. Evidence pack opcional para debugging (removido bloqueio)
            evidence_pack = getattr(decision, 'evidence_pack', None)
            # if not evidence_pack or not evidence_pack.get('citations', []):
            #     should_cache = False
            #     cache_reason.append("evidence_pack_vazio")
            
            # 3. Grounding status opcional para debugging (removido bloqueio)
            grounding_status = getattr(decision, 'grounding_status', None)
            # if grounding_status and grounding_status not in ["grounded", "conversational"]:
            #     should_cache = False
            #     cache_reason.append(f"grounding_falhou_{grounding_status}")
            
            # 4. Permitir cache mesmo com escalation (para debugging)
            # if decision.should_escalate:
            #     should_cache = False
            #     cache_reason.append("escalation_ativo")
            
            # 5. Verificar se não é erro
            if decision.agent_type == "error":
                should_cache = False
                cache_reason.append("resposta_erro")
            
            # 🆕 DEBUG: Log decisão
            self.logger.info(f"🔍 DEBUG: should_cache={should_cache}, reasons={cache_reason}")
            
            # Log da decisão
            if should_cache:
                # Determinar TTL baseado no tipo de resposta
                ttl = self._calculate_cache_ttl(decision, evidence_pack)
                
                # Metadados para cache
                cache_metadata = {
                    "cached_at": datetime.now().isoformat(),
                    "ttl": ttl,
                    "confidence": decision.confidence,
                    "agent_type": decision.agent_type,
                    "chunks_used": evidence_pack.get("chunks_used", []) if evidence_pack else [],
                    "citations": evidence_pack.get("citations", []) if evidence_pack else [],
                    "grounding_status": grounding_status,
                    "cache_reason": "approved"
                }
                
                # Salvar usando método set() do CacheStore
                self.decision_cache.set(cache_key, decision, ttl_seconds=ttl)
                
                self.logger.info(f"💾 SMART CACHE: Salvo com TTL={ttl}s, confidence={decision.confidence:.2f}")
            else:
                self.logger.info(f"🚫 SMART CACHE: Bloqueado - {', '.join(cache_reason)}")
                
        except Exception as e:
            self.logger.error(f"Error in smart cache: {e}")
    
    def _calculate_cache_ttl(self, decision: AgentDecision, evidence_pack: Optional[Dict[str, Any]]) -> int:
        """
        Calcula TTL baseado no tipo de resposta
        - Factual estável → TTL maior (3600s = 1h)
        - Troubleshooting/contextual → TTL menor (900s = 15min)
        """
        try:
            # Base TTL
            base_ttl = 1800  # 30 minutos
            
            # Ajustar baseado no tipo de agente
            agent_ttl_adjustments = {
                "atendimento_geral": 900,   # 15min - pode mudar
                "criacao_conta": 3600,     # 1h - processo estável
                "open_finance": 1800,      # 30min - pode ter atualizações
                "golpe_med": 600          # 10min - urgência e mudanças
            }
            
            agent_ttl = agent_ttl_adjustments.get(decision.agent_type, base_ttl)
            
            # Ajustar baseado no confidence
            if decision.confidence >= 0.9:
                confidence_multiplier = 1.5
            elif decision.confidence >= 0.8:
                confidence_multiplier = 1.0
            else:
                confidence_multiplier = 0.5
            
            # Ajustar baseado no número de chunks (mais chunks = mais estável)
            chunk_count = len(evidence_pack.get("chunks_used", [])) if evidence_pack else 0
            if chunk_count >= 3:
                chunk_multiplier = 1.2
            elif chunk_count >= 2:
                chunk_multiplier = 1.0
            else:
                chunk_multiplier = 0.8
            
            # Calcular TTL final
            final_ttl = int(agent_ttl * confidence_multiplier * chunk_multiplier)
            
            # Limites: mínimo 5min, máximo 2h
            final_ttl = max(300, min(7200, final_ttl))
            
            self.logger.debug(f"TTL Calculation: agent={decision.agent_type}, base={agent_ttl}, "
                            f"conf_mult={confidence_multiplier}, chunk_mult={chunk_multiplier}, "
                            f"final={final_ttl}s")
            
            return final_ttl
            
        except Exception as e:
            self.logger.error(f"Error calculating TTL: {e}")
            return 1800  # Fallback 30min
    
    def _generate_cache_key(self, message: AgentMessage, agent_type: str) -> str:
        """Gera chave para cache versionado e determinístico"""
        # 🆕 Componentes versionados para cache determinístico
        user_hash = hashlib.md5(message.user_id.encode()).hexdigest()[:8]
        conversation_hash = _get_conversation_hash(message.context.get('message_history', []), max_messages=3)
        kb_version = _get_kb_version()
        
        # 🆕 Incluir hash do contexto de memória para consistência
        memory_context = message.context.get('memory_context', {})
        
        # 🆕 Incluir entities no memory_hash
        entities = memory_context.get('memory_entities', {})
        entities_hash = ""
        if entities:
            entities_hash = hashlib.md5(
                json.dumps(entities, sort_keys=True).encode()
            ).hexdigest()[:8]
        
        memory_hash = hashlib.md5(
            f"{memory_context.get('memory_summary', '')}"
            f"{memory_context.get('memory_stage', '')}"
            f"{memory_context.get('memory_turn_count', 0)}"
            f"{entities_hash}"
        .encode()
        ).hexdigest()[:8]
        
        # Montar chave com todos os componentes críticos
        cache_components = [
            message.content,           # Conteúdo da mensagem
            agent_type,               # Tipo de agente
            str(message.priority.value),  # Prioridade
            user_hash,               # Hash do usuário (PII protected)
            conversation_hash,       # Hash do histórico
            kb_version,              # Versão da KB
            POLICY_VERSION,           # Versão da policy
            memory_hash              # 🆕 Hash do contexto de memória
        ]
        
        cache_string = "|".join(cache_components)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _update_metrics(self, decision: AgentDecision, processing_time: float, agent_type: str):
        """Atualiza métricas"""
        self.metrics["messages_processed"] += 1
        self.metrics["total_processing_time"] += processing_time
        
        # Distribuição de agentes
        if agent_type not in self.metrics["agent_distribution"]:
            self.metrics["agent_distribution"][agent_type] = 0
        self.metrics["agent_distribution"][agent_type] += 1
        
        # 🆕 Métricas de Grounding por agente
        gm = self.metrics["grounding_metrics"]
        gm["total_queries"] += 1
        
        # Atualizar métricas agregadas
        if decision.grounded:
            gm["grounded_count"] += 1
        
        # Médias móveis
        total_queries = gm["total_queries"]
        gm["avg_grounding_score"] = ((gm["avg_grounding_score"] * (total_queries - 1)) + decision.grounding_score) / total_queries
        gm["avg_citation_coverage"] = ((gm["avg_citation_coverage"] * (total_queries - 1)) + decision.citation_coverage) / total_queries
        
        # Métricas por agente específico
        if agent_type not in gm["by_agent"]:
            gm["by_agent"][agent_type] = {
                "total_queries": 0,
                "grounded_count": 0,
                "avg_grounding_score": 0.0,
                "avg_citation_coverage": 0.0,
                "avg_confidence": 0.0,
                "rag_hit_rate": 0.0,
                "false_negative_rate": 0.0,
                "regeneration_rate": 0.0,
                "avg_regeneration_count": 0.0,
                "avg_risk_level": "UNKNOWN",
                "escalation_rate": 0.0
            }
        
        agent_metrics = gm["by_agent"][agent_type]
        agent_metrics["total_queries"] += 1
        
        if decision.grounded:
            agent_metrics["grounded_count"] += 1
        
        # Atualizar médias por agente
        agent_total = agent_metrics["total_queries"]
        agent_metrics["avg_grounding_score"] = ((agent_metrics["avg_grounding_score"] * (agent_total - 1)) + decision.grounding_score) / agent_total
        agent_metrics["avg_citation_coverage"] = ((agent_metrics["avg_citation_coverage"] * (agent_total - 1)) + decision.citation_coverage) / agent_total
        agent_metrics["avg_confidence"] = ((agent_metrics["avg_confidence"] * (agent_total - 1)) + decision.confidence) / agent_total
        
        # RAG hit rate
        if decision.rag_used:
            agent_metrics["rag_hit_rate"] = ((agent_metrics["rag_hit_rate"] * (agent_total - 1)) + 1.0) / agent_total
        else:
            agent_metrics["rag_hit_rate"] = ((agent_metrics["rag_hit_rate"] * (agent_total - 1)) + 0.0) / agent_total
        
        # False negative rate
        is_false_negative = "Não encontrei" in decision.response and decision.rag_used
        if is_false_negative:
            agent_metrics["false_negative_rate"] = ((agent_metrics["false_negative_rate"] * (agent_total - 1)) + 1.0) / agent_total
        else:
            agent_metrics["false_negative_rate"] = ((agent_metrics["false_negative_rate"] * (agent_total - 1)) + 0.0) / agent_total
        
        # Regeneration rate
        regeneration_count = getattr(decision, 'trace', {}).get('regeneration_count', 0)
        if regeneration_count > 0:
            agent_metrics["regeneration_rate"] = ((agent_metrics["regeneration_rate"] * (agent_total - 1)) + 1.0) / agent_total
            agent_metrics["avg_regeneration_count"] = ((agent_metrics["avg_regeneration_count"] * (agent_total - 1)) + regeneration_count) / agent_total
        else:
            agent_metrics["regeneration_rate"] = ((agent_metrics["regeneration_rate"] * (agent_total - 1)) + 0.0) / agent_total
            agent_metrics["avg_regeneration_count"] = ((agent_metrics["avg_regeneration_count"] * (agent_total - 1)) + 0.0) / agent_total
        
        # Risk metrics (para golpe_med)
        if hasattr(decision, 'risk_level'):
            agent_metrics["avg_risk_level"] = decision.risk_level
        
        # Escalation rate
        if decision.should_escalate:
            agent_metrics["escalation_rate"] = ((agent_metrics["escalation_rate"] * (agent_total - 1)) + 1.0) / agent_total
        else:
            agent_metrics["escalation_rate"] = ((agent_metrics["escalation_rate"] * (agent_total - 1)) + 0.0) / agent_total
    
    def _validate_escalation_reason(self, decision: AgentDecision) -> AgentDecision:
        """Valida que escalation_reason é obrigatório e válido"""
        if decision.should_escalate:
            if not decision.escalation_reason:
                # 🆕 Inferir reason baseado no contexto
                if decision.agent_type == "golpe_med":
                    decision.escalation_reason = EscalationReason.FRAUD_RISK.value
                elif decision.confidence < 0.3:
                    decision.escalation_reason = EscalationReason.MISSING_INFORMATION.value
                elif "tool" in decision.reasoning.lower() or "error" in decision.reasoning.lower():
                    decision.escalation_reason = EscalationReason.TOOL_FAILURE.value
                else:
                    decision.escalation_reason = EscalationReason.UNKNOWN.value
                
                self.logger.warning(f"🚨 Escalation without reason - inferred: {decision.escalation_reason}")
            
            # Validar que reason é válido
            valid_reasons = [reason.value for reason in EscalationReason]
            if decision.escalation_reason not in valid_reasons:
                self.logger.error(f"🚨 Invalid escalation_reason: {decision.escalation_reason}")
                decision.escalation_reason = EscalationReason.UNKNOWN.value
        
        return decision
    
    def _update_business_metrics(self, decision: AgentDecision, agent_type: str):
        """Atualiza métricas de negócio críticas"""
        try:
            total = self.metrics["messages_processed"]
            
            # Auto-resolution rate
            if not decision.should_escalate and decision.confidence > 0.7:
                self.metrics["auto_resolution_rate"] = ((self.metrics["auto_resolution_rate"] * (total - 1)) + 1.0) / total
            
            # Escalation rate
            if decision.should_escalate:
                self.metrics["escalation_rate"] = ((self.metrics["escalation_rate"] * (total - 1)) + 1.0) / total
            
            # RAG usage rate
            if decision.rag_used:
                self.metrics["rag_usage_rate"] = ((self.metrics["rag_usage_rate"] * (total - 1)) + 1.0) / total
            
            # Empty context rate (simulado - verificar se RAG retornou vazio)
            if hasattr(decision, 'evidence_pack') and not decision.evidence_pack.get('selected_docs'):
                self.metrics["empty_context_rate"] = ((self.metrics["empty_context_rate"] * (total - 1)) + 1.0) / total
                
        except Exception as e:
            self.logger.error(f"Error updating business metrics: {e}")
    
    def _create_error_decision(self, error_message: str) -> AgentDecision:
        """Cria decisão de erro padronizada"""
        return AgentDecision(
            action=AgentAction.RESPOND,
            response="Desculpe, ocorreu um erro ao processar sua mensagem. Por favor, tente novamente.",
            confidence=0.0,
            agent_type="error",
            reasoning=f"Erro no processamento: {error_message}",
            processing_time=0.0,
            rag_used=False,
            should_escalate=True,
            escalation_reason="tool_failure",
            evidence_pack={}
        )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Obtém métricas do orquestrador"""
        # Build llm_metrics expected by health check endpoint
        llm_metrics: Dict[str, Any] = {}
        if self.llm_manager:
            llm_metrics = self.llm_manager.get_metrics()
            primary = getattr(self.llm_manager, 'primary_provider', None)
            if primary:
                provider_name = primary.config.provider.value
                api_key_present = False
                try:
                    cfg_key = getattr(getattr(primary, "config", None), "api_key", None)
                    cli_key = getattr(getattr(primary, "client", None), "api_key", None)
                    api_key_present = bool(
                        cfg_key
                        or cli_key
                        or os.getenv("OPENAI_API_KEY")
                    )
                except Exception:
                    api_key_present = False
                llm_metrics["primary_provider"] = provider_name
                llm_metrics["providers"] = {
                    provider_name: {
                        "configured": True,
                        "api_key_present": api_key_present,
                        "healthy": api_key_present,
                    }
                }
        return {
            "orchestrator": self.metrics,
            "cache_size": self.decision_cache.size(),
            "status": "operational",
            "llm_metrics": llm_metrics,
        }

# Singleton para o orquestrador com thread safety
_orchestrator_instance: Optional[JotaAgentOrchestrator] = None
_orchestrator_lock = asyncio.Lock()

async def get_agent_orchestrator(config: Dict[str, Any] = None) -> JotaAgentOrchestrator:
    """Obtém instância do orquestrador (singleton com thread safety)"""
    global _orchestrator_instance
    
    async with _orchestrator_lock:
        if _orchestrator_instance is None:
            _orchestrator_instance = JotaAgentOrchestrator(config)
            await _orchestrator_instance.initialize()
        
        return _orchestrator_instance
