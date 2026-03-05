"""
Base Agent - Jota Support Agent
Classe base comum para todos os agentes especializados
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from support_agent.orchestrator.agent_orchestrator import (
    get_rag_system, RAGDocument, RAGQuery, RAGResult,
    _normalize_query_text, _get_kb_version, RAG_SIGNATURE_CACHE,
)
from support_agent.llm.llm_manager import get_llm_manager
from support_agent.prompts.prompt_manager import get_prompt_manager

logger = logging.getLogger(__name__)

# Module-level RAG dedup metrics (reset between validation runs)
_rag_metrics = {
    "orchestrator_rag_queries": 0,
    "agent_rag_queries": 0,
    "rag_reuse_success": 0,
    "rag_reuse_failed": 0,
}

def get_rag_metrics() -> Dict[str, int]:
    """Return a copy of the current RAG dedup metrics."""
    return dict(_rag_metrics)

def reset_rag_metrics():
    """Reset all RAG dedup metrics to zero."""
    for k in _rag_metrics:
        _rag_metrics[k] = 0

class BaseAgent(ABC):
    """Classe base para todos os agentes do Jota"""
    
    def __init__(self, rag_system=None, llm_manager=None, prompt_manager=None):
        self.rag_system = rag_system
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self):
        """Inicializa dependências do agente"""
        if not self.rag_system:
            self.rag_system = await get_rag_system()
        
        if not self.llm_manager:
            self.llm_manager = await get_llm_manager()
        
        if not self.prompt_manager:
            self.prompt_manager = get_prompt_manager()
        
        # Log de configuração efetiva LLM (ETAPA 0)
        import os
        model = os.getenv("OLLAMA_MODEL")
        max_tokens = os.getenv("OLLAMA_MAX_TOKENS")
        timeout = os.getenv("OLLAMA_TIMEOUT")
        
        self.logger.info("🔧 LLM CONFIG EFETIVA:")
        self.logger.info(f"   model={model}")
        self.logger.info(f"   max_tokens={max_tokens}")
        self.logger.info(f"   timeout={timeout}")
    
    def _try_reuse_rag(self, context: Dict[str, Any]) -> Optional[RAGResult]:
        """
        Attempt to reuse the RAG result already computed by the orchestrator.
        Includes a Snapshot Guard: validates _meta identity fields before reuse.
        Returns a reconstructed RAGResult if valid, or None (fail-closed).
        """
        def _reject(reason: str) -> None:
            """Record rejection in metrics and log with reason code."""
            _rag_metrics["rag_reuse_failed"] += 1
            _rag_metrics["agent_rag_queries"] += 1
            self.logger.info(
                f"🛡️ RAG snapshot guard REJECTED: reason={reason}"
            )

        # --- basic payload presence ---
        payload = context.get("_orchestrator_rag_v1")
        if not payload or not isinstance(payload, dict):
            _reject("missing_payload")
            return None

        chunks = payload.get("chunks")
        if not chunks or not isinstance(chunks, list):
            _reject("invalid_chunks_format")
            return None

        # --- _meta guard ---
        meta = payload.get("_meta")
        if not meta or not isinstance(meta, dict):
            _reject("missing_meta")
            return None

        # (a) query_norm: must match current request
        current_query_norm = context.get("_query_norm")
        if current_query_norm is None:
            # fallback: agent has no _query_norm injected (shouldn't happen)
            _reject("missing_query_norm_in_context")
            return None

        meta_query_norm = meta.get("query_norm")
        if meta_query_norm is not None and meta_query_norm != current_query_norm:
            _reject("query_mismatch")
            return None

        # (b) top_k: sanity check
        meta_top_k = meta.get("top_k")
        if meta_top_k is not None and (not isinstance(meta_top_k, int) or meta_top_k < 1):
            _reject("topk_invalid")
            return None

        # (c) kb_version: compare only if both sides available
        meta_kb = meta.get("kb_version")
        if meta_kb is not None:
            try:
                current_kb = _get_kb_version()
                if current_kb and current_kb != "error" and meta_kb != current_kb:
                    _reject("kb_version_mismatch")
                    return None
            except Exception:
                pass  # best-effort; don't reject on access failure

        # (d) embedding_signature: compare only if both sides available
        meta_sig = meta.get("embedding_signature")
        if meta_sig is not None:
            try:
                current_sig = RAG_SIGNATURE_CACHE.get("signature")
                if current_sig and current_sig not in ("unknown", "error") and meta_sig != current_sig:
                    _reject("embedding_signature_mismatch")
                    return None
            except Exception:
                pass  # best-effort

        # --- guard passed — reconstruct RAGResult ---
        try:
            documents = [
                RAGDocument(
                    content=ch["content"],
                    metadata=ch.get("metadata", {}),
                    doc_id=ch.get("source_id", ""),
                    chunk_id=ch.get("chunk_id", ""),
                    source=ch.get("source_id", ""),
                    score=ch.get("score", 0.0),
                )
                for ch in chunks
            ]
            query = RAGQuery(
                query=payload.get("query", ""),
                agent_type=getattr(self, "agent_type", "atendimento_geral"),
                user_context={},
                top_k=payload.get("top_k", 8),
            )
            result = RAGResult(
                documents=documents,
                query=query,
                confidence=payload.get("confidence", 0.0),
                processing_time=payload.get("processing_time", 0.0),
                source=payload.get("source", "knowledge_base"),
            )
            _rag_metrics["rag_reuse_success"] += 1
            self.logger.info(
                "♻️ RAG reused (guard PASSED): %d docs, top_score=%.3f",
                len(documents),
                documents[0].score if documents else 0.0,
            )
            return result
        except Exception as e:
            _reject("reconstruct_error")
            self.logger.warning(f"Failed to reconstruct RAG from orchestrator payload: {e}")
            return None

    def _apply_overrides(
        self, 
        query: str, 
        decision, 
        evidence_pack: Dict[str, Any], 
        rag_result: Optional[Any] = None, 
        agent_name: str = ""
    ):
        """
        Hook global para Strong Evidence Override com 3 camadas de blindagem semântica
        Aplicado a TODOS os agentes sem duplicar lógica
        """
        try:
            # Verificar strong_match
            has_strong_match = evidence_pack.get("strong_match", {}).get("strong_match", False)
            
            if not has_strong_match:
                return decision
            
            strong_match = evidence_pack["strong_match"]
            citation_id = strong_match.get("citation_id", "[C1]")
            snippet = strong_match.get("snippet_original", "")
            
            # Inicializar motivos de override
            override_reasons = []
            
            # CAMADA 1: Confidence Gate
            confidence = getattr(decision, 'confidence', 0.85)
            if confidence < 0.6:
                override_reasons.append("low_confidence")
            
            # CAMADA 2: Citation Presence Check
            response_text = getattr(decision, 'response', '')
            if citation_id not in response_text:
                override_reasons.append("missing_citation")
            
            # CAMADA 3: Semantic Alignment Check
            query_lower = query.lower()
            response_lower = response_text.lower()
            
            # Extrair palavra-chave principal da query
            keyword_found = False
            matched_keywords = strong_match.get("matched_keywords", [])
            
            # Tentar usar keywords do strong_match primeiro
            if matched_keywords:
                for keyword in matched_keywords:
                    if keyword.lower() in query_lower and keyword.lower() not in response_lower:
                        override_reasons.append("semantic_mismatch")
                        keyword_found = True
                        break
            
            # Se não há keywords no strong_match, extrair da query
            if not keyword_found:
                # Palavras-chave comuns para detectar
                common_keywords = ["cartão", "aplicativo", "rendimento", "pix", "limite", "horário", "atendimento"]
                for keyword in common_keywords:
                    if keyword in query_lower and keyword not in response_lower:
                        override_reasons.append("semantic_mismatch")
                        break
            
            # Detectar false fallback patterns (camada original)
            false_fallback_patterns = [
                "não encontrei",
                "não encontrei informações específicas",
                "nao encontrei",
                "nao encontrei informacoes"
            ]
            
            is_false_fallback = any(pattern in response_lower for pattern in false_fallback_patterns)
            if is_false_fallback:
                override_reasons.append("fallback_detected")
            
            # Aplicar override se qualquer motivo for acionado
            if override_reasons:
                # Log obrigatório com todos os motivos
                self.logger.info(f"🔍 STRONG EVIDENCE OVERRIDE - {agent_name}")
                self.logger.info(f"   - strong_match=True")
                self.logger.info(f"   - motivos: {override_reasons}")
                self.logger.info(f"   - confidence_anterior: {confidence}")
                self.logger.info(f"   - citation: {citation_id}")
                self.logger.info(f"   - snippet: {snippet[:100]}...")
                self.logger.info(f"   - doc_rank: {strong_match.get('doc_rank', 'N/A')}")
                self.logger.info(f"   - patterns: {strong_match.get('matched_patterns', [])}")
                
                # Override determinístico
                override_response = self._deterministic_override(query, strong_match)
                
                # Atualizar decision
                decision.response = override_response
                decision.reasoning = f"Strong evidence override applied: {', '.join(override_reasons)}"
                decision.confidence = 0.9
                
                return decision
            
            # Se nenhum motivo, retornar decision intacta
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in _apply_overrides: {e}")
            # Em caso de erro, não quebrar - retornar decision intacta
            return decision
    
    def _topic_alignment_guard(
        self, 
        query: str, 
        decision, 
        evidence_pack: Dict[str, Any], 
        agent_name: str = "",
        retry_count: int = 0
    ) -> tuple[object, bool]:
        """
        SOFT GUARD: Topic Alignment Guard - Apenas logging, sem bloqueio
        Verifica alinhamento tópico mas NÃO bloqueia respostas úteis
        """
        try:
            # Flags de verificação (apenas para logging)
            checks_failed = []
            response_text = getattr(decision, 'response', '')
            response_lower = response_text.lower()
            query_lower = query.lower()
            
            # 1) Fallback detected (apenas fallbacks verdadeiros)
            fallback_patterns = [
                "não encontrei informações específicas",
                "nao encontrei informacoes especificas"
            ]
            response_text_clean = response_text.strip()
            
            # Verificar se é um fallback real
            if any(pattern in response_text_clean for pattern in fallback_patterns):
                # Verificar se a resposta é realmente um fallback (sem conteúdo útil)
                # Se a resposta tem citações ou conteúdo específico, não é fallback
                has_citations = any(f"[C{num}]" in response_text for num in range(1, 100))
                has_useful_content = len(response_text_clean) > 150 or any(keyword in response_text_clean.lower() for keyword in ["passo", "como", "para", "deve", "pode", "funciona"])
                
                if not has_citations and not has_useful_content:
                    checks_failed.append("fallback_detected")
            
            # 2) Missing citation (apenas se não tiver nenhuma citação)
            selected_docs = evidence_pack.get("selected_docs", [])
            if selected_docs:
                expected_citations = evidence_pack.get("citations", [])
                if expected_citations and not any(citation in response_text for citation in expected_citations):
                    checks_failed.append("missing_citation")
            
            # 3) Semantic mismatch (apenas se não tiver nenhuma keyword relevante)
            strong_match = evidence_pack.get("strong_match", {})
            matched_keywords = strong_match.get("matched_keywords", [])
            
            if matched_keywords:
                target_keywords = [kw.lower() for kw in matched_keywords]
                missing_keywords = [kw for kw in target_keywords if kw not in response_lower]
                if len(missing_keywords) > len(target_keywords) * 0.8:  # 80%+ das keywords faltando
                    checks_failed.append("semantic_mismatch")
            
            # 4) Cross-topic leak (apenas se for muito grave)
            domain_keywords = self._get_domain_keywords(agent_name)
            leak_keywords = [kw for kw in domain_keywords if kw in response_lower and kw not in query_lower]
            
            if len(leak_keywords) > 3:  # Apenas se muitas palavras de outros domínios
                checks_failed.append("cross_topic_leak")
            
            # 🆕 SOFT GUARD: Nunca bloquear, apenas logar e manter resposta útil
            if checks_failed:
                self.logger.warning(f"🔍 SOFT GUARD: Issues detected but NOT blocking: {checks_failed}")
                self.logger.info(f"🔍 SOFT GUARD: Keeping useful response despite minor issues")
                
                # Log em JSON para auditoria
                guard_log = {
                    "event": "soft_guard_log",
                    "agent": agent_name,
                    "checks_failed": checks_failed,
                    "response_length": len(response_text),
                    "has_citations": any(f"[C{num}]" in response_text for num in range(1, 100)),
                    "confidence": getattr(decision, 'confidence', 0),
                    "action": "no_block_soft_guard"
                }
                self.logger.info(f"SOFT_GUARD_LOG: {json.dumps(guard_log)}")
            
            # 🆕 SOFT GUARD: Nunca retry, sempre manter resposta
            return decision, False
            
        except Exception as e:
            self.logger.error(f"Error in topic_alignment_guard: {e}")
            return decision, False
    
    def _check_explicit_facts(self, query: str, rag_result) -> Optional[str]:
        """
        FASE 0 - FATO EXPLÍCITO GLOBAL (Determinístico)
        Verifica se há fato explícito nos chunks recuperados antes de qualquer geração
        """
        if not rag_result or not hasattr(rag_result, 'documents') or not rag_result.documents:
            return None
        
        query_lower = query.lower()
        
        # Regras determinísticas globais
        fact_rules = {
            # A) PIX sem senha
            "pix_sem_senha": {
                "keywords": ["pix", "senha", "sem senha", "sem senha"],
                "required_phrases": ["não existe limite de pix sem senha", "todas as transações exigem senha", "exige senha"],
                "response_template": "Todas as transações do Jota são autorizadas com senha, independentemente do valor. Não existe Pix sem senha. {citation}",
                "check_noturno": True
            },
            # B) Aplicativo
            "aplicativo": {
                "keywords": ["aplicativo", "app", "baixar", "instalar"],
                "required_phrases": ["não existe aplicativo", "100% pelo whatsapp", "funciona 100% pelo whatsapp"],
                "response_template": "Funciona 100% no WhatsApp, não existe aplicativo separado. {citation}",
                "check_noturno": False
            },
            # C) Cartão de crédito
            "cartao": {
                "keywords": ["cartão", "cartão de crédito", "cartao"],
                "required_phrases": ["não emite cartão de crédito", "jota não emite cartão"],
                "response_template": "O Jota não emite cartão de crédito. {citation}",
                "check_noturno": False
            },
            # D) Horário
            "horario": {
                "keywords": ["horário", "funciona", "aberto", "fecha"],
                "required_phrases": ["seg-sex 8:30–20h", "fins de semana/feriados 9–20h", "sem suporte 24h"],
                "response_template": "Seg-sex 8:30–20h; fins de semana/feriados 9–20h; sem suporte 24h. {citation}",
                "check_noturno": False
            },
            # E) Rendimento
            "rendimento": {
                "keywords": ["rendimento", "rende", "cdi", "100% cdi"],
                "required_phrases": ["100% do cdi", "crédito em dias úteis", "rende 100% do cdi"],
                "response_template": "Rende 100% do CDI para todo o saldo. {citation}",
                "check_noturno": False
            }
        }
        
        # Verificar cada regra
        for rule_name, rule in fact_rules.items():
            # Verificar se a query contém as keywords
            if any(keyword in query_lower for keyword in rule["keywords"]):
                # Procurar frase obrigatória nos chunks
                for doc in rag_result.documents[:3]:  # Verificar top 3 chunks
                    content_lower = doc.content.lower() if hasattr(doc, 'content') else str(doc).lower()
                    
                    # Verificar se chunk contém pelo menos uma frase obrigatória
                    if any(phrase in content_lower for phrase in rule["required_phrases"]):
                        # Extrair citação do chunk
                        citation = getattr(doc, 'metadata', {}).get('citation', '[C#]')
                        if not citation or citation == '[C#]':
                            # Tentar extrair do conteúdo
                            import re
                            citation_match = re.search(r'\[C\d+\]', doc.content if hasattr(doc, 'content') else str(doc))
                            citation = citation_match.group(0) if citation_match else '[C#]'
                        
                        # Construir resposta
                        response = rule["response_template"].format(citation=citation)
                        
                        # Adicionar info de noturno se for PIX e mencionar horário
                        if rule["check_noturno"] and ("noturno" in query_lower or "noite" in query_lower or "22h" in query_lower):
                            if "limite noturno" in content_lower or "22h–06h" in content_lower:
                                response = response.replace(citation, f"{citation} Limite noturno 22h–06h: R$3.000.")
                        
                        # Log do fato explícito encontrado
                        self.logger.info(json.dumps({
                            "event": "explicit_fact_found",
                            "rule": rule_name,
                            "query": query,
                            "response": response,
                            "chunk_id": getattr(doc, 'doc_id', 'unknown'),
                            "citation": citation
                        }))
                        
                        return response
        
        return None
    
    def _select_anchor_chunk(self, rag_result, evidence_pack: Dict[str, Any]) -> Dict[str, Any]:
        """
        FASE A - ANCHOR CHUNK SELECTION (Pipeline RAG Estrutural)
        Seleção determinística de 1 chunk âncora para todos os agentes
        """
        selected_docs = evidence_pack.get("selected_docs", [])
        
        # 🆕 LOG OBRIGATÓRIO: Verificar se rag_result está vazio
        if not rag_result or not hasattr(rag_result, 'documents') or not rag_result.documents:
            self.logger.warning("🚫 ANCHOR SELECTION: rag_result vazio ou sem documentos")
            return {
                "anchor_doc": None,
                "anchor_chunk_id": None,
                "anchor_section": None,
                "anchor_score": 0.0,
                "reason": "rag_result_empty"
            }
        
        # 🆕 LOG OBRIGATÓRIO: Informações básicas
        self.logger.info(f"🔍 ANCHOR SELECTION: {len(selected_docs)} docs disponíveis")
        
        if not selected_docs:
            self.logger.warning("🚫 ANCHOR SELECTION: Nenhum documento para seleção")
            return {
                "anchor_doc": None,
                "anchor_chunk_id": None,
                "anchor_section": None,
                "anchor_score": 0.0,
                "reason": "no_docs_available"
            }
        
        # Critérios de contaminação (padrão para todos os agentes)
        domain_keywords = {
            'pix': ['pix', 'senha', 'limite', 'transação'],
            'cartão': ['cartão', 'crédito', 'débito'],
            'rendimento': ['rendimento', 'juros', 'rentabilidade', 'cdi'],
            'segurança': ['golpe', 'segurança', 'estorno', 'med'],
            'funcionalidades': ['aplicativo', 'funcionalidade', 'disponível'],
            'open_finance': ['banco', 'conectar', 'saldo', 'extrato']
        }
        
        def is_contaminated(doc_content):
            """Verifica se chunk tem contaminação multi-tópico"""
            content_lower = doc_content.lower()
            domains_found = []
            
            for domain, keywords in domain_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    domains_found.append(domain)
            
            # Se tem mais de 1 domínio diferente, está contaminado
            return len(domains_found) > 1, domains_found
        
        # Tentar top-1, depois top-2, depois top-3
        anchor_doc = None
        reason = ""
        
        for i, doc in enumerate(selected_docs[:3]):
            is_contaminated_result, domains = is_contaminated(doc.content if hasattr(doc, 'content') else str(doc))
            
            if not is_contaminated_result:
                anchor_doc = doc
                reason = f"top{i+1}_ok" if i == 0 else f"top{i+1}_used_top{i}_contaminated"
                break
            else:
                self.logger.info(f"Anchor selection: top{i+1} contaminated with domains: {domains}")
        
        # Se todos estiverem contaminados, usar top-1 mesmo assim
        if not anchor_doc:
            anchor_doc = selected_docs[0]
            reason = "all_contaminated_using_top1"
        
        # Extrair informações do anchor
        anchor_chunk_id = anchor_doc.doc_id if hasattr(anchor_doc, 'doc_id') else str(anchor_doc.get('doc_id', 'unknown'))
        anchor_score = anchor_doc.score if hasattr(anchor_doc, 'score') else 0.0
        
        # Encontrar seção/citação
        citations = evidence_pack.get("citations", [])
        anchor_section = citations[0] if citations else None
        
        # Log obrigatório com informações completas
        self.logger.info(json.dumps({
            "event": "anchor_chunk_selected",
            "anchor_chunk_id": anchor_chunk_id,
            "anchor_section": anchor_section,
            "anchor_score": anchor_score,
            "reason": reason,
            "total_docs": len(selected_docs),
            "retrieval_top_k": len(rag_result.documents) if rag_result and hasattr(rag_result, 'documents') else 0
        }))
        
        # 🆕 LOG OBRIGATÓRIO: Formato simplificado para debugging
        self.logger.info(f"🎯 ANCHOR SELECTED: id={anchor_chunk_id}, section={anchor_section}, reason={reason}, top_k={len(rag_result.documents) if rag_result and hasattr(rag_result, 'documents') else 0}")
        
        return {
            "anchor_doc": anchor_doc,
            "anchor_chunk_id": anchor_chunk_id,
            "anchor_section": anchor_section,
            "anchor_score": anchor_score,
            "reason": reason
        }
    
    def _is_objective_question(self, query: str) -> bool:
        """
        FASE B - Detecção de pergunta objetiva (Pipeline RAG Estrutural)
        Heurística padrão para todos os agentes
        """
        query_lower = query.lower().strip()
        
        # Keywords objetivas
        objective_keywords = [
            "qual", "quanto", "tem", "existe", "limite", "horário", "rendimento", 
            "taxa", "senha", "app", "cartão", "pix", "open finance", "como faço para", 
            "como conectar", "funciona", "disponível", "valor", "valor máximo", "mínimo"
        ]
        
        # Verificar se contém keyword objetiva
        has_objective_keyword = any(keyword in query_lower for keyword in objective_keywords)
        
        # Verificar se é curta e direta
        is_short_direct = (
            len(query_lower) < 80 and 
            (query_lower.endswith("?") or 
             any(word in query_lower for word in ["quero", "preciso", "gostaria"]))
        )
        
        return has_objective_keyword or is_short_direct
    
    def _kb_fact_check(self, query: str, response: str, anchor_info: Dict[str, Any], anchor_content: str) -> str:
        """
        FASE C - KB FACT CHECK (Pipeline RAG Estrutural)
        Verificação determinística para todos os agentes
        """
        query_lower = query.lower()
        response_lower = response.lower()
        anchor_lower = anchor_content.lower()
        anchor_section = anchor_info.get("anchor_section", "[C1]")
        
        # Keywords críticas e suas respostas esperadas (padrão para todos os agentes)
        fact_templates = {
            "pix sem senha": f"Todas as transações exigem senha; não existe Pix sem senha. {anchor_section}",
            "senha pix": f"Todas as transações exigem senha; não existe Pix sem senha. {anchor_section}",
            "limite pix sem senha": f"Todas as transações exigem senha; não existe Pix sem senha. {anchor_section}",
            "limite pix": f"Entre 22h e 06h, o limite de Pix é R$ 3.000,00 e não é possível aumentar. {anchor_section}",
            "limite noturno": f"Entre 22h e 06h, o limite de Pix é R$ 3.000,00 e não é possível aumentar. {anchor_section}",
            "rendimento": f"O saldo rende 100% do CDI, com crédito em dias úteis. {anchor_section}",
            "100% cdi": f"O saldo rende 100% do CDI, com crédito em dias úteis. {anchor_section}",
            "aplicativo": f"Não existe aplicativo; funciona 100% pelo WhatsApp. {anchor_section}",
            "app": f"Não existe aplicativo; funciona 100% pelo WhatsApp. {anchor_section}",
            "cartão": f"O Jota não emite cartão de crédito. {anchor_section}",
            "cartão de crédito": f"O Jota não emite cartão de crédito. {anchor_section}",
            "horário": f"Seg-sex 8:30–20h; fins de semana/feriados 9–20h; sem suporte 24h. {anchor_section}",
            "funciona": f"Seg-sex 8:30–20h; fins de semana/feriados 9–20h; sem suporte 24h. {anchor_section}"
        }
        
        # Verificar se há fato explícito no anchor chunk (mais restritivo)
        for keyword, expected_response in fact_templates.items():
            if keyword in query_lower:
                # 🆕 SÓ aplicar override se chunk contém o fato explicitamente
                fact_phrases = {
                    "pix sem senha": ["não existe pix sem senha", "exige senha", "todas as transações exigem senha"],
                    "senha pix": ["não existe pix sem senha", "exige senha", "todas as transações exigem senha"],
                    "limite pix sem senha": ["não existe pix sem senha", "exige senha", "todas as transações exigem senha"],
                    "limite pix": ["limite de pix é r$ 3.000,00", "entre 22h e 06h", "não é possível aumentar"],
                    "limite noturno": ["limite de pix é r$ 3.000,00", "entre 22h e 06h", "não é possível aumentar"],
                    "rendimento": ["100% do cdi", "crédito em dias úteis", "rende 100% do cdi"],
                    "100% cdi": ["100% do cdi", "crédito em dias úteis", "rende 100% do cdi"],
                    "aplicativo": ["não existe aplicativo", "100% pelo whatsapp", "funciona 100% pelo whatsapp"],
                    "app": ["não existe aplicativo", "100% pelo whatsapp", "funciona 100% pelo whatsapp"],
                    "cartão": ["não emite cartão de crédito", "jota não emite cartão"],
                    "cartão de crédito": ["não emite cartão de crédito", "jota não emite cartão"],
                    "horário": ["seg-sex 8:30–20h", "fins de semana/feriados 9–20h", "sem suporte 24h"],
                    "funciona": ["seg-sex 8:30–20h", "fins de semana/feriados 9–20h", "sem suporte 24h"]
                }
                
                # Verificar se chunk contém pelo menos uma frase do fato
                required_phrases = fact_phrases.get(keyword, [])
                chunk_has_fact = any(phrase in anchor_lower for phrase in required_phrases)
                
                if chunk_has_fact:
                    # Verificar se resposta contém keyword central e citação correta
                    if keyword not in response_lower or anchor_section not in response:
                        # Aplicar override determinístico
                        self.logger.warning(json.dumps({
                            "event": "kb_fact_check_failed",
                            "reason": "missing_keyword_or_citation",
                            "expected_keyword": keyword,
                            "anchor_section": anchor_section,
                            "chunk_has_fact": True,
                            "used_template": True
                        }))
                        return expected_response
        
        # Verificar se resposta é fallback mas o anchor tem a informação
        if "não encontrei informações" in response_lower:
            # Procurar por informações relevantes no anchor
            if any(phrase in anchor_lower for phrase in [
                "senha", "limite", "rendimento", "aplicativo", "cartão", "horário", "funciona"
            ]):
                # Tentar extrair resposta simples do anchor
                for keyword, expected_response in fact_templates.items():
                    if keyword in query_lower:
                        self.logger.warning(json.dumps({
                            "event": "kb_fact_check_failed",
                            "reason": "fallback_with_info_available",
                            "expected_keyword": keyword,
                            "anchor_section": anchor_section,
                            "used_template": True
                        }))
                        return expected_response
        
        # Se passou em todos os checks, manter resposta original
        self.logger.info("KB Fact Check passed - keeping original response")
        return response
    
    def _get_domain_keywords(self, agent_type: str) -> List[str]:
        """Retorna keywords do domínio para detecção de cross-topic leak"""
        domain_keywords = {
            "atendimento_geral": ["pix", "senha", "limite", "transação", "cartão", "crédito", "débito", "rendimento", "juros", "rentabilidade", "golpe", "segurança", "estorno", "med", "aplicativo", "funcionalidade", "disponível"],
            "criacao_conta": ["cpf", "cnpj", "documento", "cadastro", "aprovação", "conta", "abertura"],
            "open_finance": ["banco", "conectar", "saldo", "extrato", "integração"],
            "golpe_med": ["golpe", "estorno", "med", "polícia", "boletim", "segurança"]
        }
        return domain_keywords.get(agent_type, [])
    
    def _deterministic_override(self, query: str, strong_match: Dict[str, Any]) -> str:
        """Override determinístico quando async não é possível"""
        citation_id = strong_match.get("citation_id", "[C1]")
        snippet = strong_match.get("snippet_original", "")
        query_lower = query.lower()
        
        # Overrides específicos baseados no padrão atual
        if "aplicativo" in query_lower:
            return f"Não há aplicativo separado, o Jota funciona 100% pelo WhatsApp. {citation_id}"
        elif "cartão" in query_lower and "credito" in query_lower:
            return f"O Jota não emite cartão de crédito. {citation_id}"
        elif "rendimento" in query_lower:
            if "100%" in snippet:
                return f"O saldo na conta Jota rende 100% do CDI. {citation_id}"
            else:
                return f"Qualquer saldo na conta Jota rende. Para detalhes específicos sobre percentuais, entre em contato pelo WhatsApp. {citation_id}"
        else:
            # Fallback genérico com evidência
            return f"{snippet[:200]} {citation_id}"
    
    @abstractmethod
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Processa mensagem do cliente - implementação obrigatória"""
        pass
    
    def _parse_json_response(self, response: str, default_fields: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse da resposta JSON do LLM com fallback robusto"""
        if default_fields is None:
            default_fields = {"response": response}
        
        try:
            # Garantir que response é string
            if not isinstance(response, str):
                response = str(response)
            
            # Limpar a resposta
            response = response.strip()
            
            # Tentar extrair JSON da resposta
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                
                # Tentar fazer parse do JSON
                try:
                    parsed = json.loads(json_str)
                    # Validar estrutura mínima
                    if isinstance(parsed, dict):
                        # Mesclar com defaults
                        return {**default_fields, **parsed}
                except json.JSONDecodeError:
                    pass
            
            # Fallback se não encontrar JSON ou JSON inválido
            return default_fields
            
        except Exception as e:
            self.logger.warning(f"Error parsing JSON response: {e}")
            # Fallback final
            return default_fields
    
    def _get_user_name(self, context: Dict[str, Any]) -> str:
        """Extrai nome do usuário do contexto"""
        return context.get("user_name", "Cliente")
    
    def _verify_grounding(self, response: str, evidence_pack: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grounding Check pós-resposta com validação de IDs e regeneração
        Verifica se a resposta usa apenas evidências disponíveis
        """
        try:
            # 1. Verificar presença de citações [C#]
            import re
            citation_pattern = r'\[C(\d+)\]'
            found_citations = re.findall(citation_pattern, response)
            
            # 2. Validar que os IDs citados existem no Evidence Pack
            available_citations = evidence_pack.get("citations", [])
            available_ids = []
            for citation in available_citations:
                match = re.search(r'C(\d+)', citation)
                if match:
                    available_ids.append(match.group(1))
            
            # 3. Verificar se todas as citações são válidas
            invalid_citations = [c for c in found_citations if c not in available_ids]
            valid_citations = [c for c in found_citations if c in available_ids]
            
            # 4. Verificar se há afirmações factuais sem citação
            sentences = response.split('.')
            factual_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and not re.search(citation_pattern, sentence):
                    # Sentença longa sem citação pode ser factual não ancorada
                    factual_sentences.append(sentence)
            
            # Log detalhado
            self.logger.info(f"🔍 GROUNDING CHECK:")
            self.logger.info(f"   - Found citations: {found_citations}")
            self.logger.info(f"   - Available IDs: {available_ids}")
            self.logger.info(f"   - Invalid citations: {invalid_citations}")
            self.logger.info(f"   - Valid citations: {valid_citations}")
            self.logger.info(f"   - Factual sentences without citation: {len(factual_sentences)}")
            
            # 5. Decisão de grounding
            has_invalid_citations = len(invalid_citations) > 0
            has_uncited_factual = len(factual_sentences) > 0
            has_valid_citations = len(valid_citations) > 0
            
            # Critérios para aprovação
            if has_invalid_citations:
                status = "invalid_citations"
                grounded = False
                reason = f"Citações inválidas encontradas: [C{'], [C'.join(invalid_citations)}]"
            elif has_uncited_factual and not has_valid_citations:
                status = "missing_citations"
                grounded = False
                reason = "Afirmações factuais sem citação encontradas"
            elif has_valid_citations:
                status = "grounded"
                grounded = True
                reason = f"Resposta com {len(valid_citations)} citações válidas"
            else:
                # Resposta conversacional sem afirmações factuais
                status = "conversational"
                grounded = True
                reason = "Resposta conversacional sem afirmações factuais"
            
            self.logger.info(f"📊 GROUNDING RESULT: {status} - {reason}")
            
            return {
                "grounded": grounded,
                "status": status,
                "reason": reason,
                "found_citations": found_citations,
                "valid_citations": valid_citations,
                "invalid_citations": invalid_citations,
                "uncited_factual_count": len(factual_sentences),
                "available_citations": available_citations
            }
            
        except Exception as e:
            self.logger.error(f"Error in grounding check: {e}")
            # Em caso de erro, falhar conservadoramente
            return {
                "grounded": False,
                "status": "error",
                "reason": f"Erro no grounding check: {str(e)}",
                "found_citations": [],
                "valid_citations": [],
                "invalid_citations": [],
                "uncited_factual_count": 0,
                "available_citations": evidence_pack.get("citations", [])
            }
    
    async def _regenerate_with_grounding_instructions(self, message: str, context: Dict[str, Any], 
                                                   evidence_pack: Dict[str, Any], agent_type: str) -> str:
        """
        Regenera resposta com instruções explícitas de grounding
        """
        try:
            # Obter system prompt do PromptManager
            system_prompt = self.prompt_manager.get_system_prompt(agent_type)
            
            # Construir prompt com instruções de grounding reforçadas
            grounding_prompt = f"""{system_prompt}

{evidence_pack['context']}

MENSAGEM DO CLIENTE:
{message}

INSTRUÇÕES ESPECIAIS DE GROUNDING:
1. Use APENAS as citações disponíveis: {', '.join(evidence_pack['citations'])}
2. CADA afirmação factual DEVE ter uma citação [C#]
3. NÃO use informações fora do EVIDENCE PACK
4. Se não encontrar informação, diga explicitamente
5. Formato: "Informação [C#]. Outra informação [C#]."

Responda seguindo estritamente estas regras:"""

            # Gerar nova resposta
            llm_response = await self.llm_manager.generate_response(
                prompt=grounding_prompt,
                agent_type=agent_type
            )
            
            return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
        except Exception as e:
            self.logger.error(f"Error in regeneration: {e}")
            # Fallback para resposta segura
            return "Não encontrei informações específicas sobre isso na base do conhecimento do Jota. Posso te ajudar com outra dúvida?"
    
    async def _retry_with_strong_evidence(self, query: str, evidence_pack: Dict[str, Any], llm_response: str, agent_type: str) -> str:
        """
        Retry global com strong evidence quando LLM responde "não encontrei" mas há evidência explícita
        Funciona para qualquer agente
        """
        if not evidence_pack.get("strong_match", {}).get("strong_match", False):
            return llm_response
        
        strong_match = evidence_pack["strong_match"]
        citation_id = strong_match.get("citation_id", "[C1]")
        snippet = strong_match.get("snippet_original", "")
        direct_statement = strong_match.get("direct_statement", "")
        
        self.logger.info(f"🔍 STRONG EVIDENCE OVERRIDE: doc_rank={strong_match['doc_rank']}, citation_id={citation_id}, patterns={strong_match['matched_patterns']}")
        
        # Retry com prompt explícito
        try:
            retry_prompt = f"""Você respondeu 'não encontrei informações específicas', mas existe evidência explícita no {citation_id}.

EVIDÊNCIA EXPLÍCITA:
{snippet[:200]}

Refaça a resposta usando essa evidência e inclua a citação {citation_id}. 
NÃO use 'não encontrei informações específicas'.
Responda de forma curta e direta."""
            
            llm_response = await self.llm_manager.generate_response(
                prompt=retry_prompt,
                agent_type=agent_type
            )
            
            retry_response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Verificar se ainda contém "não encontrei"
            if "não encontrei" in retry_response.lower():
                # Override determinístico
                if "aplicativo" in query.lower():
                    return f"Não há aplicativo separado, o Jota funciona 100% pelo WhatsApp. {citation_id}"
                elif "cartão" in query.lower() and "credito" in query.lower():
                    return f"O Jota não emite cartão de crédito. {citation_id}"
                elif "rendimento" in query.lower():
                    if "100%" in snippet:
                        return f"O saldo na conta Jota rende 100% do CDI. {citation_id}"
                    else:
                        return f"Qualquer saldo na conta Jota rende. Para detalhes específicos sobre percentuais, entre em contato pelo WhatsApp. {citation_id}"
                else:
                    # Fallback genérico com evidência
                    return f"{snippet[:200]} {citation_id}"
            
            return retry_response
            
        except Exception as e:
            self.logger.error(f"Error in strong evidence retry: {e}")
            # Override determinístico como fallback
            if "aplicativo" in query.lower():
                return f"Não há aplicativo separado, o Jota funciona 100% pelo WhatsApp. {citation_id}"
            elif "cartão" in query.lower() and "credito" in query.lower():
                return f"O Jota não emite cartão de crédito. {citation_id}"
            else:
                return f"{snippet[:200]} {citation_id}"
    
    def _create_error_response(self, error_message: str, agent_type: str) -> Dict[str, Any]:
        """Cria resposta de erro padrão"""
        return {
            "agent_type": agent_type,
            "response": "Desculpe, estou com dificuldades para responder no momento. Por favor, tente novamente.",
            "confidence": 0.3,
            "should_delegate": False,
            "needs_escalation": True,
            "reasoning": f"Erro no processamento: {error_message}",
            "llm_generated": False
        }
    
    def _create_trace(
        self, 
        rag_result, 
        evidence_pack, 
        anchor_info=None, 
        llm_response=None,
        prompt_mode="generative",
        kb_fact_override_applied=False,
        override_reasons=None
    ) -> Dict[str, Any]:
        """
        Cria trace padronizado para rastreabilidade
        Implementação centralizada para evitar duplicação
        """
        trace = {
            "retrieval_top_k": len(rag_result.documents) if rag_result else 0,
            "chunks_used": [getattr(doc, 'doc_id', getattr(doc, 'chunk_id', str(doc.get('doc_id', 'unknown')))) 
                           for doc in evidence_pack.get("selected_docs", [])],
            "anchor_chunk_id": anchor_info.get("anchor_chunk_id") if anchor_info else None,
            "anchor_section": anchor_info.get("anchor_section") if anchor_info else None,
            "citations_expected": evidence_pack.get("citations", []),
            "prompt_mode": prompt_mode,
            "kb_fact_override_applied": kb_fact_override_applied,
            "override_reasons": override_reasons or [],
        }
        
        # 🆕 Informações do LLM (se disponível)
        if llm_response:
            trace["provider_used"] = getattr(llm_response, 'provider_used', getattr(llm_response, 'provider', 'unknown'))
            trace["fallback_used"] = getattr(llm_response, 'fallback_used', False)
            # Tentar diferentes campos para request_id
            trace["request_id"] = getattr(llm_response, 'request_id', '') or getattr(llm_response, 'id', '') or ''
            
            # 🆕 NÃO SOBRESCREVER MODEL_USED - usar valor do LLM Manager
            # O model_used já foi definido pelo LLM Manager no trace
            # Removendo lógica conflitante que mascarava o modelo real
        else:
            trace["provider_used"] = 'unknown'
            trace["fallback_used"] = False
            trace["request_id"] = ''
            trace["model_used"] = 'unknown'
            trace["model_tier"] = 'unknown'
        
        # 🆕 Log do trace para debug
        self.logger.info(f"🔍 TRACE: retrieval_top_k={trace['retrieval_top_k']}, rag_used=True")
        
        return trace
