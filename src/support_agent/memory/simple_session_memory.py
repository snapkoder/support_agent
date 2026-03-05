"""
Simple Session Memory - Versão simplificada e leve
Foco no essencial: armazenar e recuperar sessões de conversa
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class SessionFact:
    """Fato de sessão simplificado"""
    id: str
    timestamp: datetime
    fact_type: str  # "user_message" | "agent_response"
    content: str
    metadata: Dict[str, Any] = None
    agent_type: str = None

@dataclass
class UserSession:
    """Sessão de usuário simplificada"""
    user_id: str
    session_id: str
    created_at: datetime
    last_activity: datetime
    facts: List[SessionFact]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict para persistência"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "facts": [
                {
                    "id": fact.id,
                    "timestamp": fact.timestamp.isoformat(),
                    "fact_type": fact.fact_type,
                    "content": fact.content,
                    "metadata": fact.metadata,
                    "agent_type": fact.agent_type
                } for fact in self.facts
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSession':
        """Cria a partir de dict"""
        facts = []
        for fact_data in data.get("facts", []):
            fact_data["timestamp"] = datetime.fromisoformat(fact_data["timestamp"])
            facts.append(SessionFact(**fact_data))
        
        return cls(
            user_id=data["user_id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            facts=facts
        )

class OptimizedSessionMemory:
    """Memória de sessão ultra-otimizada para escala enterprise"""
    
    def __init__(self, max_sessions: int = 5000, max_facts_per_session: int = 50):
        self.sessions: Dict[str, Dict] = {}  # Estrutura simplificada
        self.max_sessions = max_sessions
        self.max_facts_per_session = max_facts_per_session
        self.logger = logging.getLogger(f"{__name__}.OptimizedSessionMemory")
        self._last_cleanup = time.time()
        self._cleanup_interval = 600  # 10 minutos
    
    async def get_session(self, user_id: str) -> Optional[Dict]:
        """Obtém sessão otimizada"""
        session = self.sessions.get(user_id)
        
        if session:
            # Verificar TTL (24 horas)
            if time.time() - session['last_activity'] > 86400:
                await self._cleanup_session(user_id)
                return None
            
            # Limpar fatos antigos
            await self._cleanup_old_facts(user_id)
            return session
        
        return None
    
    async def create_session(self, user_id: str, session_id: str = None) -> Dict:
        """Cria sessão otimizada"""
        if not session_id:
            session_id = f"{user_id}_{int(time.time())}"
        
        session = {
            'user_id': user_id,
            'session_id': session_id,
            'created_at': time.time(),
            'last_activity': time.time(),
            'facts': []
        }
        
        self.sessions[user_id] = session
        return session
    
    async def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Obtém histórico de sessões do usuário (compatibilidade com adapter)"""
        session = await self.get_session(user_id)
        if not session:
            return []
        
        # Retornar sessão atual como histórico
        return [session]
    
    async def search_facts(self, user_id: str, fact_type: str = None, limit: int = 50) -> List[SessionFact]:
        """Busca fatos por tipo (compatibilidade com adapter)"""
        session = await self.get_session(user_id)
        if not session:
            return []
        
        facts = []
        for fact_data in session['facts']:
            if fact_type and fact_data.get('type') != fact_type:
                continue
            
            # Converter para SessionFact
            fact = SessionFact(
                id=f"fact_{fact_data['timestamp']}",
                timestamp=datetime.fromtimestamp(fact_data['timestamp']),
                fact_type=fact_data.get('type', 'unknown'),
                content=fact_data.get('content', ''),
                metadata=fact_data.get('metadata', {}),
                agent_type=fact_data.get('metadata', {}).get('agent_type')
            )
            facts.append(fact)
        
        # Ordenar por timestamp e limitar
        facts.sort(key=lambda f: f.timestamp, reverse=True)
        return facts[:limit]
    
    async def add_fact(self, user_id: str, session_id: str, fact: SessionFact):
        """Adiciona SessionFact (compatibilidade com adapter)"""
        session = await self.get_session(user_id)
        if not session:
            session = await self.create_session(user_id, session_id)
        
        # Limitar fatos
        if len(session['facts']) >= self.max_facts_per_session:
            session['facts'] = session['facts'][-self.max_facts_per_session + 1:]
        
        fact_data = {
            'timestamp': fact.timestamp.timestamp(),
            'type': fact.fact_type,
            'content': fact.content,
            'metadata': fact.metadata or {}
        }
        
        session['facts'].append(fact_data)
        session['last_activity'] = time.time()
    
    async def add_simple_fact(self, user_id: str, session_id: str, fact_type: str, content: str, metadata: Dict = None):
        """Adiciona fato otimizado (método simplificado)"""
        session = await self.get_session(user_id)
        if not session:
            session = await self.create_session(user_id, session_id)
        
        # Limitar fatos
        if len(session['facts']) >= self.max_facts_per_session:
            session['facts'] = session['facts'][-self.max_facts_per_session + 1:]
        
        fact = {
            'timestamp': time.time(),
            'type': fact_type,
            'content': content,
            'metadata': metadata or {}
        }
        
        session['facts'].append(fact)
        session['last_activity'] = time.time()
    
    async def get_recent_facts(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Obtém fatos recentes otimizado"""
        session = await self.get_session(user_id)
        if not session:
            return []
        
        return session['facts'][-limit:]
    
    async def _cleanup_session(self, user_id: str):
        """Remove sessão"""
        if user_id in self.sessions:
            del self.sessions[user_id]
    
    async def _cleanup_old_facts(self, user_id: str):
        """Limpa fatos antigos"""
        session = self.sessions.get(user_id)
        if session:
            # Manter apenas últimos 7 dias
            cutoff_time = time.time() - 604800  # 7 dias
            session['facts'] = [f for f in session['facts'] if f['timestamp'] > cutoff_time]
    
    async def cleanup_expired_sessions(self):
        """Limpeza periódica otimizada"""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        # Remover sessões inativas por mais de 24 horas
        expired_users = []
        for user_id, session in self.sessions.items():
            if now - session['last_activity'] > 86400:
                expired_users.append(user_id)
        
        for user_id in expired_users:
            await self._cleanup_session(user_id)
        
        # Limitar número total de sessões
        if len(self.sessions) > self.max_sessions:
            # Remover mais antigas
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1]['last_activity']
            )
            
            for user_id, _ in sorted_sessions[:len(self.sessions) - self.max_sessions]:
                await self._cleanup_session(user_id)
        
        self._last_cleanup = now
        self.logger.debug(f"🧹 Cleaned {len(expired_users)} expired sessions")

# Singleton global
_optimized_session_memory = None

async def get_simple_session_memory(storage_path: str = None) -> OptimizedSessionMemory:
    """Retorna instância global do gerenciador otimizado"""
    global _optimized_session_memory
    
    if _optimized_session_memory is None:
        _optimized_session_memory = OptimizedSessionMemory()
    
    return _optimized_session_memory
