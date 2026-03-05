"""
Simple Memory Adapter - Versão simplificada e direta
Conecta o SimpleSessionMemory ao pipeline do agente
Estendido com orquestração de memória estruturada (hardening)
"""

import logging
import time
import json
import os
import threading
import weakref
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path

from .simple_session_memory import get_simple_session_memory, SessionFact
from ..security import redact_secrets

logger = logging.getLogger(__name__)

@dataclass
class ClientContext:
    """Contexto estruturado do cliente (simplificado)"""
    client_id: str
    last_interactions: List[Dict[str, Any]]
    preferences: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class ConversationMemory:
    """Memória conversacional simplificada"""
    conversation_id: str
    messages: List[Dict[str, Any]]
    summary: str
    timestamp: datetime

@dataclass
class MemorySnapshot:
    """Estrutura leve para snapshot de memória (hardening)"""
    conversation_id: str
    agent_type_current: str
    stage: str = "initial"
    entities: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, bool] = field(default_factory=dict)
    summary: str = ""
    schema_version: str = "1.0.0"
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    turn_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict com PII protection"""
        return {
            "conversation_id": self.conversation_id,
            "agent_type_current": self.agent_type_current,
            "stage": self.stage,
            "entities": self._sanitize_entities(self.entities),
            "constraints": self.constraints,
            "summary": redact_secrets(self.summary),
            "schema_version": self.schema_version,
            "last_updated": self.last_updated,
            "turn_count": self.turn_count
        }
    
    def _sanitize_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Remove PII de entities"""
        sanitized = {}
        for key, value in entities.items():
            if isinstance(value, str):
                sanitized[key] = redact_secrets(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_entities(value)
            else:
                sanitized[key] = value
        return sanitized
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemorySnapshot':
        """Cria snapshot do dict"""
        return cls(
            conversation_id=data["conversation_id"],
            agent_type_current=data["agent_type_current"],
            stage=data.get("stage", "initial"),
            entities=data.get("entities", {}),
            constraints=data.get("constraints", {}),
            summary=data.get("summary", ""),
            schema_version=data.get("schema_version", "1.0.0"),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            turn_count=data.get("turn_count", 0)
        )

@dataclass
class MemoryUpdate:
    """Update incremental para memória (hardening)"""
    conversation_id: str
    agent_type: str
    stage: str = ""
    new_entities: Dict[str, Any] = field(default_factory=dict)
    new_constraints: Dict[str, bool] = field(default_factory=dict)
    summary_delta: str = ""
    confidence: float = 0.0
    
    def validate(self) -> bool:
        """Valida update básico"""
        return (
            self.conversation_id and
            self.agent_type and
            0.0 <= self.confidence <= 1.0
        )

class MemoryStore:
    """Store simples, local, extensível (hardening)"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or str(Path(__file__).parent / "memory_store.json")
        self._cache: Dict[str, MemorySnapshot] = {}
        self._locks: Dict[str, threading.Lock] = {}  # 🆕 Lock por conversation_id
        self._global_lock = threading.Lock()  # Lock para o dicionário de locks
        self._lock_refs: Dict[str, int] = {}  # 🆕 Contador de referências para cleanup
        self._load_store()
    
    async def initialize(self):
        """Inicializa o SimpleMemoryAdapter (compatibilidade)"""
        # Nada a inicializar, já foi feito no __init__
        pass
    
    def _get_lock(self, conversation_id: str) -> threading.Lock:
        """Obtém lock específico para conversation_id (thread-safe)"""
        with self._global_lock:
            if conversation_id not in self._locks:
                self._locks[conversation_id] = threading.Lock()
                self._lock_refs[conversation_id] = 0
            self._lock_refs[conversation_id] += 1
            return self._locks[conversation_id]
    
    def _release_lock(self, conversation_id: str):
        """Libera referência ao lock e remove se não houver mais referências"""
        with self._global_lock:
            if conversation_id in self._lock_refs:
                self._lock_refs[conversation_id] -= 1
                if self._lock_refs[conversation_id] <= 0:
                    # Remover lock do dicionário
                    del self._locks[conversation_id]
                    del self._lock_refs[conversation_id]
    
    def _load_store(self):
        """Carrega store do disco (se existir)"""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for conv_id, snapshot_data in data.items():
                        self._cache[conv_id] = MemorySnapshot.from_dict(snapshot_data)
                logger.info(f"Loaded {len(self._cache)} memory snapshots")
        except Exception as e:
            logger.warning(f"Error loading memory store: {e}")
            self._cache = {}
    
    def _save_store(self):
        """Salva store no disco com atomicidade"""
        try:
            data = {}
            for conv_id, snapshot in self._cache.items():
                data[conv_id] = snapshot.to_dict()
            
            # Escrita atômica: arquivo temporário + rename
            temp_path = f"{self.storage_path}.tmp"
            
            # Remover arquivo temporário se existir
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Rename atômico cross-platform
            os.replace(temp_path, self.storage_path)
            return True
            
        except Exception as e:
            logger.error(f"Error saving memory store: {e}")
            # Limpar arquivo temporário em caso de erro
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False
    
    def load(self, conversation_id: str) -> Optional[MemorySnapshot]:
        """Carrega snapshot da conversa"""
        return self._cache.get(conversation_id)
    
    def save(self, snapshot: MemorySnapshot):
        """Salva snapshot da conversa com lock por conversation_id"""
        lock = self._get_lock(snapshot.conversation_id)
        
        with lock:
            self._cache[snapshot.conversation_id] = snapshot
            success = self._save_store()
            
            if success:
                # 🆕 Evento de observabilidade
                logger.info(
                    "memory_updated",
                    extra={
                        "event": "memory_updated",
                        "conversation_id": snapshot.conversation_id,
                        "agent_type": snapshot.agent_type_current,
                        "stage": snapshot.stage,
                        "turn_count": snapshot.turn_count
                    }
                )
            
            # 🆕 Liberar referência ao lock
            self._release_lock(snapshot.conversation_id)
            
            return success
    
    def list_conversations(self) -> List[str]:
        """Lista IDs de conversas com memória"""
        return list(self._cache.keys())

class SimpleMemoryAdapterWrapper:
    """
    Adaptador simplificado para orquestração de memória
    Responsabilidade: Conectar SimpleSessionMemory ao pipeline
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SimpleMemoryAdapter")
        self.session_memory = None
    
    async def initialize(self):
        """Inicializa conexão com SimpleSessionMemory"""
        try:
            self.session_memory = await get_simple_session_memory()
            self.logger.info("SimpleMemoryAdapter initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize SimpleMemoryAdapter: {e}")
            # Continua sem memória - não quebra pipeline
            self.session_memory = None
    
    async def get_client_context(self, client_id: str) -> ClientContext:
        """
        Busca contexto estruturado do cliente
        Se não houver histórico, retorna contexto vazio
        """
        try:
            if not self.session_memory:
                return self._empty_context(client_id)
            
            # Buscar sessões do usuário
            user_sessions = await self.session_memory.get_user_history(client_id, limit=1)
            
            if not user_sessions:
                return self._empty_context(client_id)
            
            session = user_sessions[0]
            
            # Extrair interações recentes
            last_interactions = []
            
            # Buscar fatos de mensagens do usuário
            user_facts = await self.session_memory.search_facts(
                client_id, 
                fact_type="user_message", 
                limit=10
            )
            
            # Buscar fatos de respostas do agente
            agent_facts = await self.session_memory.search_facts(
                client_id, 
                fact_type="agent_response", 
                limit=10
            )
            
            # Combinar em ordem cronológica
            all_facts = user_facts + agent_facts
            all_facts.sort(key=lambda f: f.timestamp)
            
            # Criar interações
            for fact in all_facts[-5:]:  # Últimas 5 interações
                interaction = {
                    "timestamp": fact.timestamp.isoformat(),
                    "message": fact.content,
                    "role": "user" if fact.fact_type == "user_message" else "agent",
                    "agent_type": fact.agent_type
                }
                last_interactions.append(interaction)
            
            return ClientContext(
                client_id=client_id,
                last_interactions=last_interactions,
                preferences={"language": "pt-BR", "contact_preference": "whatsapp"},
                metadata={"account_type": "PF", "tier": "basic"}
            )
            
        except Exception as e:
            self.logger.error(f"Error getting client context: {e}")
            return self._empty_context(client_id)
    
    def _empty_context(self, client_id: str) -> ClientContext:
        """Retorna contexto vazio"""
        return ClientContext(
            client_id=client_id,
            last_interactions=[],
            preferences={},
            metadata={}
        )

# 🆕 Memory Orchestration (Hardening)
# Singleton global
_memory_store_instance: Optional[MemoryStore] = None

def get_memory_store() -> MemoryStore:
    """Obtém instância do MemoryStore (singleton)"""
    global _memory_store_instance
    
    if _memory_store_instance is None:
        _memory_store_instance = MemoryStore()
    
    return _memory_store_instance

class MemoryOrchestrator:
    """Orquestrador de memória integrado ao pipeline (hardening)"""
    
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.memory_store = memory_store or get_memory_store()
        self.logger = logging.getLogger(f"{__name__}.MemoryOrchestrator")
    
    def load_memory_context(self, conversation_id: str) -> Dict[str, Any]:
        """Carrega contexto de memória para o pipeline"""
        snapshot = self.memory_store.load(conversation_id)
        
        if not snapshot:
            # 🆕 Evento de observabilidade
            self.logger.info(
                "memory_loaded",
                extra={
                    "event": "memory_loaded",
                    "conversation_id": conversation_id,
                    "found": False
                }
            )
            return {
                "memory_summary": "",
                "memory_entities": {},
                "memory_constraints": {},
                "memory_stage": "initial",
                "memory_turn_count": 0
            }
        
        # 🆕 Evento de observabilidade
        self.logger.info(
            "memory_loaded",
            extra={
                "event": "memory_loaded",
                "conversation_id": conversation_id,
                "found": True,
                "stage": snapshot.stage,
                "turn_count": snapshot.turn_count
            }
        )
        
        return {
            "memory_summary": snapshot.summary,
            "memory_entities": snapshot.entities,
            "memory_constraints": snapshot.constraints,
            "memory_stage": snapshot.stage,
            "memory_turn_count": snapshot.turn_count
        }
    
    def update_memory(self, update: MemoryUpdate) -> bool:
        """Atualiza memória com novo snapshot"""
        if not update.validate():
            self.logger.error(f"Invalid memory update: {update}")
            return False
        
        try:
            # Carregar snapshot existente
            snapshot = self.memory_store.load(update.conversation_id)
            
            if not snapshot:
                # Criar novo snapshot
                snapshot = MemorySnapshot(
                    conversation_id=update.conversation_id,
                    agent_type_current=update.agent_type,
                    stage=update.stage,
                    turn_count=1
                )
            else:
                # Atualizar snapshot existente
                snapshot.agent_type_current = update.agent_type
                if update.stage:
                    snapshot.stage = update.stage
                snapshot.turn_count += 1
                
                # Atualizar entities (merge com confiança)
                for key, value in update.new_entities.items():
                    if update.confidence > 0.7 or key not in snapshot.entities:
                        snapshot.entities[key] = value
                
                # Atualizar constraints
                snapshot.constraints.update(update.new_constraints)
                
                # Atualizar summary incremental
                if update.summary_delta and len(snapshot.summary) < 2000:  # Limitar tamanho
                    snapshot.summary += f"\n{update.summary_delta}" if snapshot.summary else update.summary_delta
            
            snapshot.last_updated = datetime.now().isoformat()
            
            # Salvar snapshot
            self.memory_store.save(snapshot)
            
            # 🆕 Evento de observabilidade
            self.logger.info(
                "memory_diff",
                extra={
                    "event": "memory_diff",
                    "conversation_id": update.conversation_id,
                    "agent_type": update.agent_type,
                    "stage": update.stage,
                    "entities_added": len(update.new_entities),
                    "confidence": update.confidence
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating memory: {e}")
            
            # 🆕 Evento de observabilidade
            self.logger.error(
                "memory_write_failed",
                extra={
                    "event": "memory_write_failed",
                    "conversation_id": update.conversation_id,
                    "error": str(e)
                }
            )
            
            return False

# Singleton global
_memory_orchestrator_instance: Optional[MemoryOrchestrator] = None

def get_memory_orchestrator() -> MemoryOrchestrator:
    """Obtém instância do MemoryOrchestrator (singleton)"""
    global _memory_orchestrator_instance
    
    if _memory_orchestrator_instance is None:
        _memory_orchestrator_instance = MemoryOrchestrator()
    
    return _memory_orchestrator_instance

# Continuação da classe SimpleMemoryAdapter
class SimpleMemoryAdapter:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SimpleMemoryAdapter")
        self.session_memory = None
    
    async def get_conversation_memory(self, client_id: str, session_id: str) -> ConversationMemory:
        """
        Busca memória conversacional da sessão atual
        """
        try:
            if not self.session_memory:
                return self._empty_conversation_memory(client_id, session_id)
            
            # Buscar sessão atual
            session = await self.session_memory.get_session(client_id)
            if not session:
                return self._empty_conversation_memory(client_id, session_id)
            
            # Extrair mensagens da sessão
            messages = []
            
            # Buscar todos os fatos recentes
            all_facts = await self.session_memory.search_facts(client_id, limit=50)
            all_facts.sort(key=lambda f: f.timestamp)
            
            for fact in all_facts:
                role = "user" if fact.fact_type == "user_message" else "agent"
                messages.append({
                    "role": role,
                    "content": fact.content,
                    "timestamp": fact.timestamp.isoformat()
                })
            
            # Gerar summary simples
            summary = f"Conversa com {len(messages)} trocas de mensagens"
            
            return ConversationMemory(
                conversation_id=f"conv_{client_id}_{session_id}",
                messages=messages,
                summary=summary,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error getting conversation memory: {e}")
            return self._empty_conversation_memory(client_id, session_id)
    
    async def store_interaction(self, 
                             client_id: str, 
                             session_id: str,
                             user_message: str,
                             agent_response: str,
                             agent_type: str,
                             metadata: Dict[str, Any] = None):
        """
        Armazena interação atual na memória
        """
        try:
            if not self.session_memory:
                self.logger.warning("No session memory available - skipping storage")
                return
            
            # Garantir sessão existe
            session = await self.session_memory.get_session(client_id)
            if not session:
                session = await self.session_memory.create_session(client_id, session_id)
            
            # Adicionar fato da mensagem do usuário
            user_fact = SessionFact(
                id=f"msg_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                fact_type="user_message",
                content=user_message,
                metadata=metadata or {},
                agent_type=None
            )
            
            # Adicionar fato da resposta do agente
            agent_fact = SessionFact(
                id=f"resp_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                fact_type="agent_response",
                content=agent_response,
                metadata={"agent_type": agent_type},
                agent_type=agent_type
            )
            
            # Salvar fatos
            await self.session_memory.add_fact(client_id, session_id, user_fact)
            await self.session_memory.add_fact(client_id, session_id, agent_fact)
            
            self.logger.info(f"Stored interaction for client {client_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing interaction: {e}")
            # Não quebra pipeline se falhar armazenamento
    
    def _empty_context(self, client_id: str) -> ClientContext:
        """Retorna contexto vazio"""
        return ClientContext(
            client_id=client_id,
            last_interactions=[],
            preferences={"language": "pt-BR", "contact_preference": "whatsapp"},
            metadata={"account_type": "PF", "tier": "basic"}
        )
    
    def _empty_conversation_memory(self, client_id: str, session_id: str) -> ConversationMemory:
        """Retorna memória conversacional vazia"""
        return ConversationMemory(
            conversation_id=f"conv_{client_id}_{session_id}",
            messages=[],
            summary="Nova conversa",
            timestamp=datetime.now()
        )

# Singleton global
_simple_memory_adapter = None

async def get_simple_memory_adapter() -> SimpleMemoryAdapterWrapper:
    """Retorna instância global do Memory Adapter simplificado"""
    global _simple_memory_adapter
    
    if _simple_memory_adapter is None:
        _simple_memory_adapter = SimpleMemoryAdapterWrapper()
        await _simple_memory_adapter.initialize()
    
    return _simple_memory_adapter
