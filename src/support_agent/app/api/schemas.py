from __future__ import annotations

from typing import Dict, Any, List

from pydantic import BaseModel, Field


class CustomerMessage(BaseModel):
    content: str = Field(..., max_length=5000, description="Conteúdo da mensagem")
    user_id: str = Field(..., max_length=128, description="ID do usuário")
    session_id: str = Field(..., max_length=128, description="ID da sessão")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contexto adicional")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados")


class AgentResponse(BaseModel):
    response: str = Field(..., description="Resposta gerada")
    agent_type: str = Field(..., description="Tipo de agente")
    confidence: float = Field(..., description="Confiança da resposta")
    processing_time: float = Field(..., description="Tempo de processamento")
    rag_used: bool = Field(..., description="Se usou RAG")
    should_escalate: bool = Field(..., description="Se deve escalar")
    escalation_reason: str = Field(default="", description="Motivo da escalada")
    reasoning: str = Field(..., description="Raciocínio do agente")


class HealthCheck(BaseModel):
    status: str = Field(..., description="Status do serviço")
    timestamp: str = Field(..., description="Timestamp")
    version: str = Field(default="1.0.0", description="Versão")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Métricas")


class WebhookPayload(BaseModel):
    object: str
    entry: List[Dict[str, Any]]
