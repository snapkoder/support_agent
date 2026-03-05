"""
Agent de Golpe Med - LLM-DRIVEN
Foco: Fraudes, segurança e MED - máxima prioridade para proteção do cliente
"""

import logging
from typing import Dict, Any, Optional
import asyncio
import os

from .base_agent import BaseAgent

class AgentGolpeMed(BaseAgent):
    """
    Agent de Golpe Med - LLM-DRIVEN
    Objetivo: Proteção máxima contra fraudes e golpes usando Inteligência Artificial
    - Identificar urgências imediatas com LLM
    - Coletar informações críticas para MED de forma inteligente
    """
    
    def __init__(self, rag_system=None, llm_manager=None, prompt_manager=None):
        super().__init__(rag_system, llm_manager, prompt_manager)
        self.name = "AgentGolpeMed"
        self.logger = logging.getLogger(f"{__name__}.AgentGolpeMed")
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa mensagens sobre fraudes e golpes usando LLM com Evidence Pack obrigatório
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
            self.logger.info(f"📚 EVIDENCE PACK [golpe_med]: chunks={evidence_pack['top_k']}, citations={evidence_pack['citations']}, scores={evidence_pack['scores']}")
            
            # 🆕 PARTE 1: Diagnóstico Obrigatório - Log detalhado
            system_prompt = self.prompt_manager.get_system_prompt("golpe_med")
            
            # Logar prompt efetivo aplicado (primeiras 300 chars)
            prompt_preview = system_prompt[:300] + "..." if len(system_prompt) > 300 else system_prompt
            self.logger.info(f"🔍 SYSTEM PROMPT [golpe_med]: {prompt_preview}")
            
            # Logar evidence pack efetivo
            available_ids = []
            if evidence_pack and "citations" in evidence_pack:
                available_ids = [c.replace("[C", "").replace("]", "") for c in evidence_pack["citations"]]
            self.logger.info(f"🔍 AVAILABLE IDS [golpe_med]: {available_ids}")
            
            # Logar contexto injetado no LLM (primeiras 200 chars)
            context_preview = evidence_pack.get('context', '')[:200] + "..." if len(evidence_pack.get('context', '')) > 200 else evidence_pack.get('context', '')
            self.logger.info(f"🔍 CONTEXT INJECTED [golpe_med]: {context_preview}")
            
            examples = self.prompt_manager.format_examples_for_few_shot("golpe_med", max_examples=2)
            
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
  "urgency": "URGENTE|NORMAL|EMERGENCIA",
  "response": "sua resposta empática e especializada aqui",
  "needs_escalation": true/false
}}

REGRAS ESPECÍFICAS PARA ESTA RESPOSTA:
1. Use APENAS as informações do EVIDENCE PACK acima
2. Cite cada informação factual usando [C#]
3. Se não encontrar informação no EVIDENCE PACK, diga claramente
4. A resposta deve ser extremamente empática e acolhedora
5. Identifique o tipo de fraude ou golpe se aplicável
6. Colete informações críticas para o MED
7. Explique o processo de forma clara
8. Mantenha a calma e segurança do cliente
9. Forneça orientações práticas e imediatas

Responda apenas com o JSON, sem texto adicional."""

            # Gerar resposta e análises usando ÚNICA chamada LLM
            llm_response = await self.llm_manager.generate_response(
                prompt=prompt,
                agent_type="golpe_med"
            )
            
            # Parse do JSON response
            default_fields = {
                "response": "Olá! Sou o especialista em segurança do Jota. Entendo sua situação e estou aqui para ajudar.",
                "urgency": "NORMAL",
                "needs_escalation": False
            }
            parsed_response = self._parse_json_response(llm_response.content if hasattr(llm_response, 'content') else str(llm_response), default_fields)
            
            # Valores com fallback
            urgency_level = parsed_response.get("urgency", "NORMAL")
            response_text = parsed_response.get("response", "Olá! Sou o especialista em segurança do Jota. Entendo sua situação e estou aqui para ajudar.")
            # Deterministic escalation: ignore LLM's needs_escalation (non-deterministic).
            # Escalate only on structured signal: urgency_level in URGENTE/EMERGENCIA.
            needs_escalation = False
            llm_wants_escalation = parsed_response.get("needs_escalation", False)
            if llm_wants_escalation:
                self.logger.info(f"[ESCALATION_DECISION] event=escalation_decision agent_type=golpe_med source=llm_json llm_wants=True deterministic_override=False reason=llm_signal_ignored")
            if urgency_level in ["URGENTE", "EMERGENCIA"]:
                needs_escalation = True
                self.logger.info(f"[ESCALATION_DECISION] event=escalation_decision agent_type=golpe_med source=structured reason=urgency_{urgency_level}")
            
            # 🚀 DECISÃO COM EVIDENCE PACK E TRACE BÁSICO
            from support_agent.orchestrator.agent_orchestrator import AgentDecision, AgentAction
            
            # 🆕 TRACE BÁSICO PARA RASTREABILIDADE - Usar método centralizado
            trace = self._create_trace(
                rag_result=rag_result,
                evidence_pack=evidence_pack,
                llm_response=llm_response,
                prompt_mode="generative"
            )
            
            # 🆕 PARTE 5: Risk-Based Routing Analysis
            risk_level, risk_factors, user_impact, recommended_action = self._analyze_risk_level(
                message, evidence_pack, rag_result
            )
            
            # Determinar action baseada no risco — urgency escalation is a FLOOR
            # (risk routing can only ADD escalation, never suppress urgency signal)
            should_escalate = needs_escalation  # preserve urgency-based decision
            escalation_reason = f"urgency_{urgency_level}" if needs_escalation else ""
            
            if recommended_action == "ESCALATE":
                should_escalate = True
                escalation_reason = f"High risk detected: {', '.join(risk_factors)}"
            elif recommended_action == "ASK_CLARIFY":
                # ASK_CLARIFY overrides urgency — do not escalate, ask for more info
                should_escalate = False
                escalation_reason = ""
                if "precisa de mais informações" not in response_text.lower():
                    response_text += "\n\nPara melhor ajudar, poderia me dar mais detalhes sobre o ocorrido?"
            
            action = AgentAction.ESCALATE if should_escalate else AgentAction.RESPOND
            
            decision = AgentDecision(
                action=action,
                response=response_text,
                confidence=0.85,
                agent_type="golpe_med",
                reasoning="Análise inteligente via LLM especializado em fraudes com Evidence Pack",
                processing_time=0.001,
                rag_used=True,
                should_escalate=should_escalate,
                escalation_reason=escalation_reason,
                evidence_pack=evidence_pack,
                trace=trace,  # 🆕 Rastreabilidade mínima
                # 🆕 Risk-based routing fields
                risk_level=risk_level,
                risk_factors=risk_factors,
                user_impact=user_impact,
                recommended_next_action=recommended_action
            )
            
            # 🚀 APLICAR OVERRIDE GLOBAL (antes do grounding)
            decision = self._apply_overrides(
                query=message,
                decision=decision,
                evidence_pack=evidence_pack,
                rag_result=rag_result,
                agent_name="golpe_med"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error processing message with LLM: {e}")
            
            # Fallback para resposta básica
            return AgentDecision(
                action=AgentAction.ESCALATE,
                response="Olá! Sou o especialista em segurança do Jota. Entendo sua situação e estou aqui para ajudar. Poderia me dar mais detalhes sobre o ocorrido?",
                confidence=0.3,
                agent_type="golpe_med",
                reasoning="Erro no processamento LLM, usando fallback",
                processing_time=0.001,
                rag_used=False,
                should_escalate=True,
                escalation_reason="Error - always escalate for security",
                evidence_pack=evidence_pack if evidence_pack else {"citations": [], "chunks_used": [], "answerable": False},
                # 🆕 Risk-based routing fields (fallback conservador)
                risk_level="HIGH",
                risk_factors=["error_fallback"],
                user_impact="UNKNOWN",
                recommended_next_action="ESCALATE"
            )
    
    async def _get_rag_context(self, message: str):
        """Obtém contexto RAG para o agente"""
        try:
            if not self.rag_system:
                await self.initialize()
            
            from support_agent.orchestrator.agent_orchestrator import RAGQuery
            rag_query = RAGQuery(
                query=message,
                agent_type="golpe_med",
                user_context={},
                top_k=int(os.getenv("CONTEXT_TOP_K_FINAL", "3")),
                filters={"agent_type": "golpe_med"}
            )
            
            result = await self.rag_system.query(rag_query)
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting RAG context: {e}")
            return None
    
    def _analyze_risk_level(self, message: str, evidence_pack: Dict[str, Any], rag_result) -> tuple:
        """
        Analisa o nível de risco baseado na mensagem e contexto disponível
        Retorna: (risk_level, risk_factors, user_impact, recommended_action)
        """
        import re
        
        # Inicializar variáveis
        risk_factors = []
        user_impact = "UNKNOWN"
        
        # 1. Analisar palavras-chave de alto risco
        high_risk_keywords = [
            "perdi dinheiro", "roubaram minha conta", "hackearam", "fraude", 
            "transferência não autorizada", "saque indevido", "furto", "estelionato",
            "perdi acesso", "bloquearam minha conta", "valor desapareceu"
        ]
        
        # 🆕 PARTE 4: Medium risk keywords expandidos para maior sensibilidade
        medium_risk_keywords = [
            "suspeito", "estranho", "não reconheço", "nao reconheco", "recebi mensagem",
            "clicaram em link", "pediram senha", "cpf", "dados",
            # 🆕 Adicionados para sensibilidade aumentada
            "acessaram minha conta", "transação que não reconheço", "transacao que nao reconheco",
            "fizeram pix", "não autorizei", "nao autorizei", "atividade suspeita",
            "compartilharam senha", "código", "codigo",
            # 🆕 DIV-4: Pix recebido não reconhecido
            "recebi pix", "recebi um pix", "pix recebido", "caiu pix", "caiu um pix",
            "transferencia recebida", "deposito nao reconhecido", "depósito não reconhecido",
        ]
        
        low_risk_keywords = [
            "como proteger", "segurança", "dicas", "prevenir", "cuidado",
            "duvida", "informação", "educativo"
        ]
        
        message_lower = message.lower()
        
        # Contar ocorrências de cada categoria
        high_risk_count = sum(1 for keyword in high_risk_keywords if keyword in message_lower)
        medium_risk_count = sum(1 for keyword in medium_risk_keywords if keyword in message_lower)
        low_risk_count = sum(1 for keyword in low_risk_keywords if keyword in message_lower)
        
        # Adicionar fatores de risco
        if high_risk_count > 0:
            risk_factors.extend([f"high_risk_keyword_detected_{high_risk_count}"])
            user_impact = "HIGH"
        
        if medium_risk_count > 0:
            risk_factors.extend([f"medium_risk_keyword_detected_{medium_risk_count}"])
            if user_impact == "UNKNOWN":
                user_impact = "MEDIUM"
        
        if low_risk_count > 0:
            risk_factors.extend([f"low_risk_keyword_detected_{low_risk_count}"])
            if user_impact == "UNKNOWN":
                user_impact = "LOW"
        
        # 2. Analisar evidências disponíveis no RAG
        if rag_result and rag_result.documents:
            kb_sufficient = len(rag_result.documents) > 0
            avg_similarity = sum(doc.score for doc in rag_result.documents) / len(rag_result.documents)
            
            if kb_sufficient and avg_similarity > 0.1:
                risk_factors.append("kb_sufficient_for_response")
                if user_impact == "UNKNOWN":
                    user_impact = "LOW"
            else:
                risk_factors.append("kb_insufficient")
                if user_impact == "UNKNOWN":
                    user_impact = "MEDIUM"
        else:
            risk_factors.append("no_kb_available")
            user_impact = "HIGH"
        
        # 3. 🆕 PARTE 4: Determinar nível de risco final com sensibilidade aumentada
        if high_risk_count > 0 or user_impact == "HIGH":
            risk_level = "HIGH"
            recommended_action = "ESCALATE"
        elif medium_risk_count > 0 or user_impact == "MEDIUM":
            risk_level = "MEDIUM"
            recommended_action = "ASK_CLARIFY"
        else:
            risk_level = "LOW"
            recommended_action = "RESPOND"
        
        # 🆕 PARTE 4: Casos especiais expandidos para maior sensibilidade
        if "perdi dinheiro" in message_lower or "valor desapareceu" in message_lower:
            risk_level = "HIGH"
            recommended_action = "ESCALATE"
            risk_factors.append("financial_loss_confirmed")
            user_impact = "CRITICAL"
        
        # 🆕 Adicionados: confirmação de comprometimento de conta
        if any(term in message_lower for term in ["acessaram minha conta", "comprometeram minha conta", "invadiram minha conta"]):
            risk_level = "HIGH"
            recommended_action = "ESCALATE"
            risk_factors.append("account_compromise_confirmed")
            user_impact = "CRITICAL"
        
        # 🆕 Adicionados: confirmação de compartilhamento de dados
        if any(term in message_lower for term in ["compartilharam senha", "pedi código", "enviaram token"]):
            risk_level = "HIGH"
            recommended_action = "ESCALATE"
            risk_factors.append("data_sharing_confirmed")
            user_impact = "CRITICAL"
        
        if "como proteger" in message_lower or "dicas de segurança" in message_lower:
            risk_level = "LOW"
            recommended_action = "RESPOND"
            risk_factors.append("educational_query")
            user_impact = "LOW"
        
        # Log da análise
        self.logger.info(f"🔍 RISK ANALYSIS:")
        self.logger.info(f"   - Risk level: {risk_level}")
        self.logger.info(f"   - User impact: {user_impact}")
        self.logger.info(f"   - Risk factors: {risk_factors}")
        self.logger.info(f"   - Recommended action: {recommended_action}")
        
        return risk_level, risk_factors, user_impact, recommended_action
    
