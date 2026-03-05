#!/usr/bin/env python3
"""
Cache de Respostas Rápidas
Reduz tempo de resposta para perguntas frequentes
"""

import hashlib
import json
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CachedResponse:
    response: str
    timestamp: float
    rag_used: bool
    agent_type: str
    hit_count: int = 0

class ResponseCache:
    """Cache inteligente para respostas do Jota"""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, CachedResponse] = {}
        self.logger = logging.getLogger(f"{__name__}.ResponseCache")
    
    def _generate_key(self, message: str, agent_type: str) -> str:
        """Gera chave única para cache"""
        content = f"{message.lower().strip()}|{agent_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, message: str, agent_type: str) -> Optional[CachedResponse]:
        """Obtém resposta do cache"""
        key = self._generate_key(message, agent_type)
        
        if key not in self.cache:
            return None
        
        cached = self.cache[key]
        
        # Verificar TTL
        if time.time() - cached.timestamp > self.ttl_seconds:
            del self.cache[key]
            return None
        
        # Atualizar contador
        cached.hit_count += 1
        self.logger.info(f"🎯 CACHE HIT: {message[:50]}...")
        
        return cached
    
    def set(self, message: str, agent_type: str, response: str, rag_used: bool):
        """Armazena resposta no cache"""
        key = self._generate_key(message, agent_type)
        
        # Limitar tamanho do cache
        if len(self.cache) >= self.max_size:
            # Remover mais antigo
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
        
        self.cache[key] = CachedResponse(
            response=response,
            timestamp=time.time(),
            rag_used=rag_used,
            agent_type=agent_type
        )
        
        self.logger.info(f"💾 CACHE SET: {message[:50]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Estatísticas do cache"""
        if not self.cache:
            return {"total_cached": 0, "avg_hits": 0}
        
        total_hits = sum(c.hit_count for c in self.cache.values())
        avg_hits = total_hits / len(self.cache)
        
        return {
            "total_cached": len(self.cache),
            "total_hits": total_hits,
            "avg_hits": avg_hits,
            "cache_size_bytes": len(json.dumps({
                k: v.__dict__ for k, v in self.cache.items()
            }))
        }
    
    def clear(self):
        """Limpa cache"""
        self.cache.clear()
        self.logger.info("🗑️ Cache limpo")

# Instância global
response_cache = ResponseCache()
