"""
Agent de Atendimento Geral - LLM-DRIVEN
Foco: Triagem e direcionamento inteligente usando Inteligência Artificial
"""

import logging
import os
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseAgent
from support_agent.orchestrator.agent_orchestrator import AgentDecision, AgentAction
from support_agent.orchestrator.agent_orchestrator import RAGQuery

logger = logging.getLogger(__name__)

class OptimizedAgentAtendimentoGeral(BaseAgent):
    """Agente otimizado para atendimento geral com RAG e Policy Engine"""
    
    def __init__(self, rag_system=None, llm_manager=None, prompt_manager=None):
        super().__init__(rag_system, llm_manager, prompt_manager)
        self.agent_type = "atendimento_geral"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._response_cache = {}  # Cache de respostas
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Processamento otimizado com Evidence Pack obrigatório e Fases A+B+C"""
        try:
            # 🚀 CACHE DE RESPOSTAS
            cache_key = f"{message.lower().strip()}_{context.get('user_name', '')}"
            if cache_key in self._response_cache:
                cached = self._response_cache[cache_key]
                self.logger.info(f"🎯 AGENT CACHE HIT: {message[:30]}...")
                # Garantir que retorna Dict
                return {
                    **cached,
                    "agent_type": "atendimento_geral"
                }
            
            # EXTRAÇÃO RÁPIDA DE CONTEXTO
            user_name = context.get('user_name', 'Cliente')
            
            # 🚀 EVIDENCE PACK OBRIGATÓRIO (reuse orchestrator RAG if available)
            rag_result = self._try_reuse_rag(context) or await self._get_rag_context(message)
            context_top_k = int(os.getenv("CONTEXT_TOP_K_FINAL", "3"))
            
            # 🆕 Log debug do RAG
            self.logger.info(f"🔍 RAG Result: documents={len(rag_result.documents) if rag_result else 0}, rag_result={'None' if rag_result is None else 'OK'}")
            
            # 🆕 Log detalhado dos documentos RAG
            if rag_result and rag_result.documents:
                for i, doc in enumerate(rag_result.documents[:3]):  # Top 3
                    self.logger.info(f"🔍 RAG Doc {i+1}: score={doc.score:.4f}, content='{doc.content[:60]}...'")
            
            evidence_pack = self.prompt_manager.create_evidence_pack(rag_result, message, top_k=context_top_k)
            
            # Log do Evidence Pack
            self.logger.info(f"📚 EVIDENCE PACK: chunks={evidence_pack['top_k']}, citations={evidence_pack['citations']}, scores={evidence_pack['scores']}")
            
            # 🆕 Log do conteúdo do evidence pack
            if evidence_pack.get('selected_docs'):
                self.logger.info(f"📚 EVIDENCE CONTENT: {len(evidence_pack['selected_docs'])} docs selected")
                for i, doc in enumerate(evidence_pack['selected_docs'][:2]):  # Top 2
                    content_preview = doc.get('content', str(doc))[:80]
                    self.logger.info(f"📚 Evidence {i+1}: '{content_preview}...'")
            else:
                self.logger.warning(f" EVIDENCE PACK: No selected_docs found!")
            
            # FASE 0 - FATO EXPLÍCITO GLOBAL (Determinístico)
            explicit_fact_response = self._check_explicit_facts(message, rag_result)
            if explicit_fact_response:
                # Resposta determinística encontrada - criar trace básico
                trace = self._create_trace(
                    rag_result=rag_result,
                    evidence_pack=evidence_pack,
                    prompt_mode="extractive"
                )
                decision = AgentDecision(
                    action=AgentAction.RESPOND,
                    response=explicit_fact_response,
                    confidence=1.0,  # Máxima confiança para fato explícito
                    agent_type="atendimento_geral",
                    reasoning="Fato explícito global encontrado - resposta determinística",
                    processing_time=0.001,
                    rag_used=True,
                    should_escalate=False,
                    escalation_reason="",
                    trace=trace
                )
                
                return decision
            
            # FASE A - ANCHOR CHUNK SELECTION (usando método do BaseAgent)
            enable_anchor_selection = os.getenv("ENABLE_ANCHOR_SELECTION", "false").lower() == "true"
            anchor_info = {}
            
            if enable_anchor_selection:
                anchor_info = self._select_anchor_chunk(rag_result, evidence_pack)
            else:
                # Modo compatibilidade - anchor desativado
                anchor_info = {
                    "anchor_doc": None,
                    "anchor_chunk_id": None,
                    "anchor_section": None,
                    "anchor_score": 0.0,
                    "reason": "anchor_selection_disabled"
                }
            
            # FASE B - DETECTAR SE É PERGUNTA OBJETIVA (usando método do BaseAgent)
            enable_extractive = os.getenv("ENABLE_EXTRACTIVE_MODE", "false").lower() == "true"
            is_objective = self._is_objective_question(message) if enable_extractive else False
            
            # 🆕 PARTE 3: Logar question_type e extractive_applied
            question_type = "objective" if is_objective else "subjective"
            extractive_applied = False
            
            self.logger.info(f"🎯 Question type: {question_type} (extractive_mode={enable_extractive})")
            
            # FASE B - MODO DE RESPOSTA POR EXTRAÇÃO (apenas para objective)
            if enable_extractive and is_objective and anchor_info["anchor_doc"]:
                extractive_applied = True
                self.logger.info(f"🔧 Extractive mode applied: {extractive_applied}")
                
                # Usar apenas anchor chunk para perguntas objetivas
                anchor_doc = anchor_info["anchor_doc"]
                anchor_content = anchor_doc.content if hasattr(anchor_doc, 'content') else str(anchor_doc)
                anchor_section = anchor_info["anchor_section"]
                
                # Prompt de extração (corrigido - sem metalinguagem)
                extractive_prompt = f"""Responda apenas com a resposta final ao cliente em português natural.

TRECHO:
{anchor_content}

PERGUNTA: {message}

REGRAS OBRIGATÓRIAS:
1. Responda em português natural como se estivesse conversando com o cliente
2. Produza apenas a resposta final, não explique o trecho
3. Não faça análise estrutural do texto
4. Não use bullets ou listas
5. Se a resposta não estiver explicitamente no texto fornecido, responda exatamente: 'Não encontrei informações específicas sobre isso. {anchor_section}'
6. Inclua apenas 1 citação no formato {anchor_section}

Resposta:"""
                
                try:
                    llm_response = await asyncio.wait_for(
                        self.llm_manager.generate_response(
                            prompt=extractive_prompt,
                            agent_type="atendimento_geral"
                        ),
                        timeout=float(os.getenv("LLM_EXTRACTIVE_TIMEOUT", "30.0"))
                    )
                    
                    response_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                    
                    # 🆕 Extrair informações do LLM para trace
                    llm_provider_used = getattr(llm_response, 'provider_used', 'unknown')
                    llm_fallback_used = getattr(llm_response, 'fallback_used', False)
                    llm_request_id = getattr(llm_response, 'request_id', '')
                    
                except asyncio.TimeoutError:
                    self.logger.warning("⚠️ LLM timeout - usando fallback")
                    response_content = f"Não encontrei informações específicas sobre isso. {anchor_section}"
                
                # FASE C - KB FACT CHECK (usando método do BaseAgent)
                enable_fact_check = os.getenv("ENABLE_KB_FACT_CHECK", "false").lower() == "true"
                original_response = response_content
                
                if enable_fact_check:
                    response_content = self._kb_fact_check(message, response_content, anchor_info, anchor_content)
                
                # 🆕 ATUALIZAR TRACE se override foi aplicado
                if response_content != original_response:
                    trace["kb_fact_override_applied"] = True
                    trace["override_reasons"].append("kb_fact_check")
                
            else:
                # Modo normal para perguntas subjetivas
                prompt = self._get_optimized_prompt_with_evidence(message, user_name, evidence_pack)
                
                # Safe diagnostic log (no full prompt to avoid PII leak)
                import hashlib as _hl
                self.logger.debug(
                    "Prompt generated",
                    extra={
                        "agent": "atendimento_geral",
                        "prompt_hash": _hl.sha256(prompt.encode()).hexdigest()[:8],
                        "prompt_size": len(prompt),
                    }
                )
                
                try:
                    llm_response = await asyncio.wait_for(
                        self.llm_manager.generate_response(
                            prompt=prompt,
                            agent_type="atendimento_geral"
                        ),
                        timeout=float(os.getenv("LLM_STANDARD_TIMEOUT", "30.0"))
                    )
                    
                    response_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                    self.logger.debug(
                        "LLM response received",
                        extra={"response_length": len(response_content)}
                    )
                    
                    # 🆕 Extrair informações do LLM para trace
                    llm_provider_used = getattr(llm_response, 'provider_used', 'unknown')
                    llm_fallback_used = getattr(llm_response, 'fallback_used', False)
                    llm_request_id = getattr(llm_response, 'request_id', '')
                    
                except asyncio.TimeoutError:
                    self.logger.warning("⚠️ LLM timeout - usando fallback")
                    response_content = f"Olá, {user_name}! 🌟 Sou o assistente do Jota. Como posso te ajudar hoje?"
            
            # 🚀 GROUNDING CHECK PÓS-RESPOSTA
            evidence_pack_for_grounding = evidence_pack
            grounding_result = self._verify_grounding(response_content, evidence_pack_for_grounding)
            
            # 🆕 RASTREABILIDADE MÍNIMA - Usar método centralizado do BaseAgent
            trace = self._create_trace(
                rag_result=rag_result,
                evidence_pack=evidence_pack,
                anchor_info=anchor_info,
                llm_response=llm_response,
                prompt_mode="extractive" if (is_objective and anchor_info["anchor_doc"]) else "generative",
                kb_fact_override_applied=False,
                override_reasons=[]
            )
            
            # 🆕 Adicionar métricas de extractive mode ao trace
            trace["question_type"] = question_type
            trace["extractive_applied"] = extractive_applied
            
            # 🚀 DECISÃO OTIMIZADA COM EVIDENCE PACK E TRACE
            decision = AgentDecision(
                action=AgentAction.RESPOND,
                response=response_content,
                confidence=0.85,
                agent_type="atendimento_geral",
                reasoning="Processamento com Anchor Chunk e Fact Check",
                processing_time=0.001,
                rag_used=True,
                should_escalate=False,
                escalation_reason="",
                evidence_pack=evidence_pack,  # 🆕 Incluir evidence_pack com citações
                trace=trace  # 🆕 Rastreabilidade mínima
            )
            
            #  CACHE ASSINCRONO
            asyncio.create_task(self._cache_response_async(cache_key, decision))
            
            return decision
            
        except Exception as e:
            self.logger.error(f"❌ Error in optimized agent: {e}")
            self.logger.error(f"❌ Error type: {type(e)}")
            self.logger.error(f"❌ Error args: {e.args}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            # Criar fallback simples
            fallback = {
                "action": "RESPOND",
                "response": "Ocorreu um erro ao processar sua mensagem. Estou transferindo para um especialista.",
                "confidence": 0.1,
                "agent_type": "atendimento_geral",
                "reasoning": f"Error: {str(e)}",
                "processing_time": 0.001,
                "rag_used": False,
                "should_escalate": False,
                "escalation_reason": ""
            }
            
            self.logger.info(f"🔧 Returning fallback: {fallback}")
            return fallback
    
    async def _get_rag_context(self, message: str):
        """Obtém contexto RAG para o agente"""
        try:
            self.logger.info(f"🔍 Getting RAG context for message: '{message}'")
            self.logger.info(f"🔍 RAG System available: {self.rag_system is not None}")
            
            rag_query = RAGQuery(
                query=message,
                agent_type=self.agent_type,
                user_context={},
                top_k=8  # Retrieval bruto
            )
            
            self.logger.info(f"🔍 RAG Query created: top_k={rag_query.top_k}, agent_type={rag_query.agent_type}")
            
            result = await self.rag_system.query(rag_query)
            self.logger.info(f"🔍 RAG Query result: {len(result.documents)} docs returned")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error getting RAG context: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return None
    
    def _get_optimized_prompt_with_evidence(self, message: str, user_name: str, evidence_pack: Dict[str, Any]) -> str:
        """Gera prompt otimizado com Evidence Pack"""
        
        # Extrai contexto do evidence pack
        context = evidence_pack.get("context", "")
        citations = evidence_pack.get("citations", [])
        strong_match = evidence_pack.get("strong_match", {})
        
        # Constrói prompt claro com separação de papéis
        base_prompt = f"""Você é o assistente virtual do Jota. Responda a pergunta do cliente usando as informações abaixo.

{context}

PERGUNTA DO CLIENTE: {message}

Responda de forma clara, direta e amigável em português. Use as citações [C#] para referenciar as informações acima."""
        
        # Se há strong match, adicionar instrução específica
        if strong_match and strong_match.get("snippet_original"):
            base_prompt += f"\n\n⚠️ ATENÇÃO: Use principalmente esta informação: {strong_match['snippet_original']}"
        
        return base_prompt
    
    def _verify_grounding(self, response: str, evidence_pack: Dict[str, Any]) -> Dict[str, Any]:
        """Verifica se resposta está grounded no evidence pack"""
        
        # Verifica se há citações
        citations = evidence_pack.get("citations", [])
        has_citations = any(f"[C{num}]" in response for num in range(1, 100))
        
        # Verifica se é fallback
        fallback_patterns = ["não encontrei", "nao encontrei", "posso te ajudar com outra dúvida"]
        is_fallback = any(pattern in response.lower() for pattern in fallback_patterns)
        
        # Verifica se há afirmações factuais sem citação
        factual_keywords = ["é", "tem", "funciona", "limite", "valor", "taxa"]
        has_factual_claims = any(keyword in response.lower() for keyword in factual_keywords)
        
        grounding_issues = []
        
        if not has_citations and not is_fallback:
            grounding_issues.append("missing_citations")
        
        if has_factual_claims and not has_citations:
            grounding_issues.append("factual_without_citation")
        
        return {
            "grounded": len(grounding_issues) == 0,
            "issues": grounding_issues,
            "has_citations": has_citations,
            "is_fallback": is_fallback
        }
    
    def _create_fallback_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Cria resposta de fallback segura"""
        user_name = context.get('user_name', 'Cliente')
        
        fallback_response = f"Olá, {user_name}! 🌟 Sou o assistente do Jota. Como posso te ajudar hoje?"
        
        return {
            "action": "RESPOND",
            "response": fallback_response,
            "confidence": 0.1,
            "agent_type": "atendimento_geral",
            "reasoning": "Fallback response - error in processing",
            "processing_time": 0.001,
            "rag_used": False,
            "should_escalate": False,
            "escalation_reason": ""
        }
    
    async def _cache_response_async(self, cache_key: str, decision: AgentDecision):
        """Cache assíncrono de resposta"""
        try:
            self._response_cache[cache_key] = {
                "action": decision.action,
                "response": decision.response,
                "confidence": decision.confidence,
                "agent_type": decision.agent_type,
                "reasoning": decision.reasoning,
                "rag_used": decision.rag_used,
                "should_escalate": decision.should_escalate,
                "escalation_reason": decision.escalation_reason,
                "evidence_pack": getattr(decision, 'evidence_pack', None)
            }
        except Exception as e:
            self.logger.error(f"Error caching response: {e}")
