"""
Agent de Criação de Conta - LLM-DRIVEN
Foco: Onboarding e problemas de cadastro usando Inteligência Artificial
"""

import logging
from typing import Dict, Any, Optional
import asyncio
import os

from .base_agent import BaseAgent

class AgentCriacaoConta(BaseAgent):
    """
    Agent de Criação de Conta - LLM-DRIVEN
    Objetivo: Especialista em onboarding e problemas de cadastro usando IA
    - Guiar clientes através do processo de criação com LLM
    - Resolver problemas comuns de cadastro inteligentemente
    - Validar documentos e requisitos
    - Manter tom acolhedor e profissional
    """
    
    def __init__(self, rag_system=None, llm_manager=None, prompt_manager=None):
        super().__init__(rag_system, llm_manager, prompt_manager)
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa mensagens sobre criação de conta usando LLM com Evidence Pack obrigatório
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
            self.logger.info(f"📚 EVIDENCE PACK [criacao_conta]: chunks={evidence_pack['top_k']}, citations={evidence_pack['citations']}, scores={evidence_pack['scores']}")
            
            # Obter system prompt e exemplos do PromptManager
            system_prompt = self.prompt_manager.get_system_prompt("criacao_conta")
            examples = self.prompt_manager.format_examples_for_few_shot("criacao_conta", max_examples=2)
            
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
  "response": "sua resposta acolhedora e profissional aqui",
  "account_type": "PF|PJ|AMBOS|INDEFINIDO",
  "issue_type": "PROBLEMA|DUVIDA|INICIO",
  "needs_escalation": true/false
}}

REGRAS ESPECÍFICAS PARA ESTA RESPOSTA:
1. Use APENAS as informações do EVIDENCE PACK acima
2. Cite cada informação factual usando [C#]
3. Se não encontrar informação no EVIDENCE PACK, diga claramente
4. A resposta deve ser acolhedora e profissional
5. Identifique se o cliente quer PF ou PJ
6. Explique o processo de abertura de forma clara
7. Forneça informações sobre documentos necessários
8. Dê o contato (11) 4004-8006 para iniciar
9. Resolva dúvidas comuns sobre cadastro
10. Mantenha o tom especialista em criação de contas

Responda apenas com o JSON, sem texto adicional."""

            # 🆕 PARTE 2: Log do prompt final para verificar injeção do contexto
            self.logger.info(f"📝 PROMPT FINAL (primeiros 500 chars): {prompt[:500]}...")
            self.logger.info(f"📝 PROMPT CONTEM CITAÇÕES: {'[C' in prompt}")
            self.logger.info(f"📝 EVIDENCE PACK NO PROMPT: {evidence_pack['context'][:200]}...")

            # Gerar resposta e análises usando ÚNICA chamada LLM
            llm_response = await self.llm_manager.generate_response(
                prompt=prompt,
                agent_type="criacao_conta"
            )
            
            # Parse do JSON response
            default_fields = {
                "response": f"Olá, {user_name}! 🏢 Sou o especialista em criação de contas do Jota. Para abrir sua conta, entre em contato conosco pelo WhatsApp no número (11) 4004-8006 e siga o passo a passo informado durante a conversa.",
                "account_type": "INDEFINIDO",
                "issue_type": "DUVIDA",
                "needs_escalation": False
            }
            parsed_response = self._parse_json_response(llm_response.content if hasattr(llm_response, 'content') else str(llm_response), default_fields)
            
            # Valores com fallback
            response_text = parsed_response.get("response", f"Olá, {user_name}! 🏢 Sou o especialista em criação de contas do Jota. Para abrir sua conta, entre em contato conosco pelo WhatsApp no número (11) 4004-8006 e siga o passo a passo informado durante a conversa.")
            account_type = parsed_response.get("account_type", "INDEFINIDO")
            issue_type = parsed_response.get("issue_type", "DUVIDA")
            # Deterministic escalation: ignore LLM's needs_escalation (non-deterministic).
            # Escalate only on structured signal: issue_type == "PROBLEMA".
            needs_escalation = False
            llm_wants_escalation = parsed_response.get("needs_escalation", False)
            if llm_wants_escalation:
                self.logger.info(f"[ESCALATION_DECISION] event=escalation_decision agent_type=criacao_conta source=llm_json llm_wants=True deterministic_override=False reason=llm_signal_ignored")
            if issue_type == "PROBLEMA":
                needs_escalation = True
                self.logger.info(f"[ESCALATION_DECISION] event=escalation_decision agent_type=criacao_conta source=structured reason=issue_type_PROBLEMA")
            
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
                agent_type="criacao_conta",
                reasoning="Análise LLM especializada em criação de contas com Evidence Pack",
                processing_time=0.001,
                rag_used=True,
                should_escalate=needs_escalation,
                escalation_reason="missing_information" if needs_escalation else "",
                evidence_pack=evidence_pack,
                trace=trace  # 🆕 Rastreabilidade mínima
            )
            
            # 🚀 APLICAR OVERRIDE GLOBAL (antes do grounding)
            decision = self._apply_overrides(
                query=message,
                decision=decision,
                evidence_pack=evidence_pack,
                rag_result=rag_result,
                agent_name="criacao_conta"
            )
            
            # 🆕 PARTE 1: Log para verificar evidence_pack no retorno do agente
            self.logger.info(f"AGENT RETURN EVIDENCE_PACK: {evidence_pack}")
            self.logger.info(f"AGENT RETURN EVIDENCE_PACK TYPE: {type(evidence_pack)}")
            if evidence_pack and isinstance(evidence_pack, dict):
                docs = len(evidence_pack.get('documents', []))
                citations = evidence_pack.get('citations', [])
                self.logger.info(f"AGENT RETURN EVIDENCE_PACK CONTENT: docs={docs}, citations={citations}")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error processing message with LLM: {e}")
            
            # Fallback para resposta básica COM Evidence Pack se disponível
            fallback_response = f"Olá, {context.get('user_name', 'Cliente')}! 🏢 Sou o especialista em criação de contas do Jota. Para abrir sua conta, entre em contato conosco pelo WhatsApp no número (11) 4004-8006 e siga o passo a passo informado durante a conversa."
            
            # Se tivermos evidence_pack, tentar usar informações dele
            if evidence_pack and evidence_pack.get('answerable', False):
                # Tentar extrair informação relevante do evidence pack
                chunks_content = " ".join([doc.content for doc in rag_result.documents[:2]] if rag_result else [])
                if "tipos de contas" in chunks_content.lower() or "cpf" in chunks_content.lower() or "cnpj" in chunks_content.lower():
                    fallback_response = f"Olá, {context.get('user_name', 'Cliente')}! 🏢 Sou o especialista em criação de contas do Jota. Oferecemos contas para Pessoas Físicas (CPF) e Jurídicas (CNPJ), incluindo ME e MEI. Para abrir sua conta, entre em contato conosco pelo WhatsApp no número (11) 4004-8006."
            
            # 🆕 TRACE BÁSICO PARA FALLBACK - Usar método centralizado
            trace = self._create_trace(
                rag_result=rag_result,
                evidence_pack=evidence_pack if evidence_pack else {"selected_docs": [], "citations": []},
                prompt_mode="generative",
                kb_fact_override_applied=False,
                override_reasons=["fallback_error"]
            )
            
            return AgentDecision(
                action=AgentAction.RESPOND,
                response=fallback_response,
                confidence=0.3,
                agent_type="criacao_conta",
                reasoning="Erro no processamento LLM, usando fallback com Evidence Pack",
                processing_time=0.001,
                rag_used=True,
                should_escalate=False,
                escalation_reason="",
                evidence_pack=evidence_pack if evidence_pack else {"citations": [], "chunks_used": [], "answerable": False},
                trace=trace  # 🆕 Rastreabilidade no fallback
            )
    
    async def _get_rag_context(self, message: str):
        """Obtém contexto RAG para o agente"""
        try:
            if not self.rag_system:
                await self.initialize()
            
            from support_agent.orchestrator.agent_orchestrator import RAGQuery
            rag_query = RAGQuery(
                query=message,
                agent_type="criacao_conta",
                user_context={},
                top_k=int(os.getenv("CONTEXT_TOP_K_FINAL", "3")),
                filters={"agent_type": "criacao_conta"}
            )
            
            result = await self.rag_system.query(rag_query)
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting RAG context: {e}")
            return None
    
    
