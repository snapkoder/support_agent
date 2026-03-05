"""
Agent de Open Finance - LLM-DRIVEN
Foco: Conexão bancária via Open Finance usando Inteligência Artificial
"""

import logging
from typing import Dict, Any, Optional
import asyncio
import os

from .base_agent import BaseAgent

class AgentOpenFinance(BaseAgent):
    """
    Agent de Open Finance - LLM-DRIVEN
    Objetivo: Resolver problemas de conexão bancária usando IA
    - Suporte a bancos via Open Finance com LLM
    - Troubleshooting inteligente de conexões
    """
    
    def __init__(self, rag_system=None, llm_manager=None, prompt_manager=None):
        super().__init__(rag_system, llm_manager, prompt_manager)
        self.name = "AgentOpenFinance"
        self.logger = logging.getLogger(f"{__name__}.AgentOpenFinance")
        
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa mensagens sobre Open Finance usando LLM com Evidence Pack obrigatório
        """
        try:
            # Extrair informações do contexto
            user_name = context.get('user_name', 'Cliente')
            message_history = context.get('message_history', [])
            
            # 🚀 EVIDENCE PACK OBRIGATÓRIO (reuse orchestrator RAG if available)
            rag_result = self._try_reuse_rag(context) or await self._get_rag_context(message)
            context_top_k = int(os.getenv("CONTEXT_TOP_K_FINAL", "3"))
            evidence_pack = self.prompt_manager.create_evidence_pack(rag_result, message, top_k=context_top_k)
            
            # Log do Evidence Pack
            self.logger.info(f"📚 EVIDENCE PACK [open_finance]: chunks={evidence_pack['top_k']}, citations={evidence_pack['citations']}, scores={evidence_pack['scores']}")
            
            # 🆕 PARTE 1: Diagnóstico Obrigatório - Log detalhado
            system_prompt = self.prompt_manager.get_system_prompt("open_finance")
            
            # Logar prompt efetivo aplicado (primeiras 300 chars)
            prompt_preview = system_prompt[:300] + "..." if len(system_prompt) > 300 else system_prompt
            self.logger.info(f"🔍 SYSTEM PROMPT [open_finance]: {prompt_preview}")
            
            # Logar evidence pack efetivo
            available_ids = []
            if evidence_pack and "citations" in evidence_pack:
                available_ids = [c.replace("[C", "").replace("]", "") for c in evidence_pack["citations"]]
            self.logger.info(f"🔍 AVAILABLE IDS [open_finance]: {available_ids}")
            
            # Logar contexto injetado no LLM (primeiras 200 chars)
            context_preview = evidence_pack.get('context', '')[:200] + "..." if len(evidence_pack.get('context', '')) > 200 else evidence_pack.get('context', '')
            self.logger.info(f"🔍 CONTEXT INJECTED [open_finance]: {context_preview}")
            
            examples = self.prompt_manager.format_examples_for_few_shot("open_finance", max_examples=2)
            
            # Construir prompt único para o LLM com Evidence Pack
            prompt = f"""{system_prompt}{examples}

CONTEXTO DA CONVERSA:
- Nome do cliente: {user_name}
- Histórico recente: {message_history[-3:] if message_history else 'Primeira mensagem'}

{evidence_pack['context']}

MENSAGEM DO CLIENTE:
{message}

Analise esta mensagem e gere uma resposta completa em formato JSON:

{{
  "response": "sua resposta especializada aqui",
  "bank_type": "GRANDE|PEQUENO|DIGITAL|INDEFINIDO",
  "issue_type": "CONEXAO|ERRO|DÚVIDA|PRIMEIRO_ACESSO",
  "needs_escalation": true/false
}}

REGRAS ESPECÍFICAS PARA ESTA RESPOSTA:
1. Use APENAS as informações do EVIDENCE PACK acima
2. Cite cada informação factual usando [C#]
3. Se não encontrar informação no EVIDENCE PACK, diga claramente
4. A resposta deve ser especializada em Open Finance
5. Identifique tipo de banco se aplicável
6. Forneça steps práticos para conexão
7. Explique erros comuns e soluções
8. Mantenha tom especialista em conexões bancárias

Responda apenas com o JSON, sem texto adicional."""

            # Gerar resposta e análises usando ÚNICA chamada LLM
            llm_response = await self.llm_manager.generate_response(
                prompt=prompt,
                agent_type="open_finance"
            )
            
            # Parse do JSON response
            default_fields = {
                "response": f"Olá, {user_name}! 🏦 Sou o especialista em Open Finance do Jota. Posso te ajudar a conectar seu banco. Qual banco você quer conectar?",
                "bank_type": "INDEFINIDO",
                "issue_type": "DÚVIDA",
                "needs_escalation": False
            }
            parsed_response = self._parse_json_response(llm_response.content if hasattr(llm_response, 'content') else str(llm_response), default_fields)
            
            # Valores com fallback
            response_text = parsed_response.get("response", f"Olá, {user_name}! 🏦 Sou o especialista em Open Finance do Jota. Posso te ajudar a conectar seu banco. Qual banco você quer conectar?")
            bank_type = parsed_response.get("bank_type", "INDEFINIDO")
            issue_type = parsed_response.get("issue_type", "DÚVIDA")
            # Deterministic escalation: ignore LLM's needs_escalation (non-deterministic).
            # Escalate only on structured signal: issue_type == "ERRO" or frustration detected.
            needs_escalation = False
            strong_hits = 0
            weak_hits = 0
            frustration_detected = False
            llm_wants_escalation = parsed_response.get("needs_escalation", False)
            if llm_wants_escalation:
                self.logger.info(f"[ESCALATION_DECISION] event=escalation_decision agent_type=open_finance source=llm_json llm_wants=True deterministic_override=False reason=llm_signal_ignored")
            if issue_type == "ERRO":
                needs_escalation = True
                self.logger.info(f"[ESCALATION_DECISION] event=escalation_decision agent_type=open_finance source=structured reason=issue_type_ERRO")
            
            # Frustration detection (deterministic, post-LLM)
            _frustration_strong = ["nada funciona", "desisto", "ja tentei tudo", "nao aguento mais"]
            _frustration_weak = ["cansado", "frustrado", "ja tentei", "tentei de tudo", "nao consigo mais", "estou cansado"]
            message_lower = message.lower()
            strong_hits = sum(1 for p in _frustration_strong if p in message_lower)
            weak_hits = sum(1 for p in _frustration_weak if p in message_lower)
            frustration_detected = (strong_hits >= 1 or weak_hits >= 2)
            if frustration_detected:
                needs_escalation = True
                self.logger.info(
                    f"[ESCALATION_DECISION] event=escalation_decision agent_type=open_finance "
                    f"source=frustration_detection reason=client_frustration_keywords "
                    f"strong={strong_hits} weak={weak_hits}"
                )
            
            # 🚀 DECISÃO COM EVIDENCE PACK E TRACE BÁSICO
            from support_agent.orchestrator.agent_orchestrator import AgentDecision, AgentAction
            
            # 🆕 TRACE BÁSICO PARA RASTREABILIDADE - Usar método centralizado
            trace = self._create_trace(
                rag_result=rag_result,
                evidence_pack=evidence_pack,
                llm_response=llm_response,
                prompt_mode="generative"
            )
            
            decision = AgentDecision(
                action=AgentAction.RESPOND,
                response=response_text,
                confidence=0.85,
                agent_type="open_finance",
                reasoning="Análise LLM especializada em Open Finance com Evidence Pack",
                processing_time=0.001,
                rag_used=True,
                should_escalate=needs_escalation,
                escalation_reason="user_frustration" if frustration_detected else ("missing_information" if needs_escalation else ""),
                evidence_pack=evidence_pack,
                trace=trace  # 🆕 Rastreabilidade mínima
            )
            
            # 🚀 APLICAR OVERRIDE GLOBAL (antes do grounding)
            decision = self._apply_overrides(
                query=message,
                decision=decision,
                evidence_pack=evidence_pack,
                rag_result=rag_result,
                agent_name="open_finance"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error processing message with LLM: {e}")
            
            # Fallback para resposta básica
            return {
                "agent_type": "open_finance",
                "response": f"Olá, {context.get('user_name', 'Cliente')}! 🏦 Sou o especialista em Open Finance do Jota. Posso te ajudar a conectar seu banco. Qual banco você quer conectar?",
                "confidence": 0.3,
                "should_delegate": False,
                "reasoning": "Erro no processamento LLM, usando fallback",
                "llm_generated": False,
                "evidence_pack": {"citations": [], "chunks_used": [], "answerable": False}
            }
    
    async def _get_rag_context(self, message: str):
        """Obtém contexto RAG para o agente"""
        try:
            if not self.rag_system:
                await self.initialize()
            
            from support_agent.orchestrator.agent_orchestrator import RAGQuery
            rag_query = RAGQuery(
                query=message,
                agent_type="open_finance",
                user_context={},
                top_k=int(os.getenv("CONTEXT_TOP_K_FINAL", "3")),
                filters={"agent_type": "open_finance"}
            )
            
            result = await self.rag_system.query(rag_query)
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting RAG context: {e}")
            return None
    
