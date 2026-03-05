"""
Memory Module - Sistema simplificado de gerenciamento de memória
Implementação leve e focada no essencial para o Jota Support Agent
Estendido com orquestração de memória estruturada (hardening)
"""

from .simple_session_memory import (
    OptimizedSessionMemory,
    SessionFact,
    get_simple_session_memory
)

from .simple_memory_adapter import (
    SimpleMemoryAdapter,
    ClientContext,
    ConversationMemory,
    get_simple_memory_adapter,
    # 🆕 Classes do hardening
    MemorySnapshot,
    MemoryUpdate,
    MemoryStore,
    MemoryOrchestrator,
    get_memory_store,
    get_memory_orchestrator
)

# Manter compatibilidade com nomes antigos
SessionMemoryManager = OptimizedSessionMemory
get_session_memory_manager = get_simple_session_memory
MemoryAdapter = SimpleMemoryAdapter
get_memory_adapter = get_simple_memory_adapter

__all__ = [
    # Optimized Session Memory
    "OptimizedSessionMemory",
    "SessionFact", 
    "get_simple_session_memory",
    
    # Compatibilidade
    "SessionMemoryManager",
    "get_session_memory_manager",
    
    # Simple Memory Adapter
    "SimpleMemoryAdapter",
    "ClientContext",
    "ConversationMemory", 
    "get_simple_memory_adapter",
    
    # Compatibilidade
    "MemoryAdapter",
    "get_memory_adapter",
    
    # 🆕 Classes do hardening
    "MemorySnapshot",
    "MemoryUpdate",
    "MemoryStore", 
    "MemoryOrchestrator",
    "get_memory_store",
    "get_memory_orchestrator"
]
