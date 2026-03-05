"""
Cache Store Singleton - Persistente por processo
Thread-safe com TTL e cleanup automático
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import hashlib

class CacheStore:
    """Cache store persistente por processo com thread-safety (instância independente por tipo)"""
    
    def __init__(self, name: str = "default"):
        self._name = name
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()  # Reentrant lock para thread safety
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutos
        self._max_size = 10000  # Máximo de entradas para evitar crescimento infinito
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with TTL check"""
        with self._lock:
            if key not in self._cache:
                return None
            
            item = self._cache[key]
            cached_at = item.get('cached_at')
            ttl_seconds = item.get('ttl_seconds', 3600)  # Default 1 hora
            
            if cached_at and ttl_seconds:
                cache_age = (datetime.now() - cached_at).total_seconds()
                if cache_age > ttl_seconds:
                    # Cache expirado - remover
                    del self._cache[key]
                    return None
            
            # Retornar apenas o valor, sem metadados
            return item.get('value')
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set item in cache with TTL"""
        with self._lock:
            # Lazy cleanup para evitar crescimento infinito
            self._lazy_cleanup()
            
            self._cache[key] = {
                'value': value,
                'cached_at': datetime.now(),
                'ttl_seconds': ttl_seconds
            }
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            now = datetime.now()
            expired_count = 0
            
            for item in self._cache.values():
                cached_at = item.get('cached_at')
                ttl_seconds = item.get('ttl_seconds', 3600)
                if cached_at and ttl_seconds:
                    cache_age = (now - cached_at).total_seconds()
                    if cache_age > ttl_seconds:
                        expired_count += 1
            
            return {
                'total_entries': len(self._cache),
                'expired_entries': expired_count,
                'valid_entries': len(self._cache) - expired_count,
                'max_size': self._max_size,
                'store_id': id(self)
            }
    
    def _lazy_cleanup(self) -> None:
        """Remove expired entries if cleanup interval passed"""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        with self._lock:
            self._last_cleanup = now
            now_dt = datetime.now()
            keys_to_delete = []
            
            for key, item in self._cache.items():
                cached_at = item.get('cached_at')
                ttl_seconds = item.get('ttl_seconds', 3600)
                
                if cached_at and ttl_seconds:
                    cache_age = (now_dt - cached_at).total_seconds()
                    if cache_age > ttl_seconds:
                        keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._cache[key]
            
            # Se ainda estiver muito grande, remover entradas mais antigas
            if len(self._cache) > self._max_size:
                # Ordenar por cached_at e remover as mais antigas
                sorted_items = sorted(
                    self._cache.items(),
                    key=lambda x: x[1].get('cached_at', datetime.min)
                )
                
                excess = len(self._cache) - self._max_size
                for i in range(excess):
                    key = sorted_items[i][0]
                    del self._cache[key]


# Module-level instances — one per cache type, persistent for process lifetime
_classify_cache = CacheStore(name="classify")
_decision_cache = CacheStore(name="decision")
_rag_cache = CacheStore(name="rag")

# Funções de acesso global
def get_classify_cache() -> CacheStore:
    """Get module-level classify cache"""
    return _classify_cache

def get_decision_cache() -> CacheStore:
    """Get module-level decision cache"""
    return _decision_cache

def get_rag_cache() -> CacheStore:
    """Get module-level RAG cache"""
    return _rag_cache
