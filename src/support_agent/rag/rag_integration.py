"""
RAG Integration - Connects RAG system with agent orchestrator
Handles RAG trigger logic and prompt assembly
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import asyncio

from .ports import DocumentChunk, RetrievedChunk
from .rag_service import RAGService
from .models import RAGConfig, QualityMetrics
from support_agent.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class RAGIntegration:
    """
    Integration layer for RAG with agent orchestrator
    Handles trigger logic, context injection, and quality guardrails
    """
    
    def __init__(self, rag_service: RAGService, llm_manager: LLMManager = None):
        """Initialize RAG integration"""
        self.rag_service = rag_service
        self.llm_manager = llm_manager or LLMManager()
        self.config = RAGConfig.from_env()
        self.logger = logging.getLogger(f"{__name__}.RAGIntegration")
        
        # Quality metrics
        self.quality_metrics = QualityMetrics()
    
    async def initialize(self) -> None:
        """Initialize RAG integration components"""
        if not hasattr(self.llm_manager, 'resolved_model'):
            await self.llm_manager.initialize()
        self.logger.info("RAG Integration initialized")
    
    async def should_use_rag(self, query: str, agent_type: str, requires_rag: bool = False) -> bool:
        """
        Decide if RAG should be used for this query.
        
        Args:
            query: User query
            agent_type: Type of agent
            requires_rag: Explicit flag from dataset
            
        Returns:
            True if RAG should be used
        """
        return await self.rag_service.should_use_rag(query, agent_type, requires_rag)
    
    async def generate_response_with_rag(
        self,
        prompt: str,
        agent_type: str = "atendimento_geral",
        requires_rag: bool = False,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using RAG when appropriate.
        
        Args:
            prompt: Original prompt
            agent_type: Type of agent
            requires_rag: Explicit RAG flag
            context: Additional context
            **kwargs: Additional LLM parameters
            
        Returns:
            Dictionary with response and RAG metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # RAG Always On - Always proceed with RAG
            self.logger.info(f"RAG Always On: Processing query for agent {agent_type}")
        
            # Process with RAG (always)
            rag_result = await self.rag_service.process_query(
                query=prompt,
                agent_type=agent_type,
                requires_rag=True,  # Always true with RAG Always On
                top_k=kwargs.get('top_k', self.config.rag_top_k),
                filters={'agent_type': agent_type, 'domain': agent_type}
            )
            
            # Assemble prompt with context
            enhanced_prompt = self._assemble_prompt_with_context(
                original_prompt=prompt,
                rag_chunks=rag_result.chunks,
                agent_type=agent_type
            )
            
            # Generate response with context
            llm_response = await self.llm_manager.generate_response(
                prompt=enhanced_prompt,
                agent_type=agent_type,
                context=context,
                **kwargs
            )
            
            # Check quality and retry if needed
            final_response = await self._check_and_retry_quality(
                original_prompt=prompt,
                response=llm_response.content,
                rag_result=rag_result,
                agent_type=agent_type,
                context=context,
                **kwargs
            )
            
            # Update quality metrics
            self.quality_metrics.update_response(
                response_text=final_response['response'],
                was_rag_used=True
            )
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                'response': final_response['response'],
                'rag_used': True,
                'rag_latency_ms': rag_result.rag_latency_ms,
                'rag_hits': len(rag_result.chunks),
                'rag_sources': [chunk.chunk.source_file for chunk in rag_result.chunks],
                'rag_domains': rag_result.metadata.get('domains', []),
                'model_used': final_response.get('model_used', llm_response.model_used),
                'processing_time': processing_time,
                'handoff_needed': final_response.get('handoff_needed', False),
                'quality_issues': final_response.get('quality_issues', []),
                'retry_count': final_response.get('retry_count', 0)
            }
            
        except Exception as e:
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self.logger.error(f"RAG integration failed: {e}")
            
            # Fallback to LLM without RAG
            try:
                llm_response = await self.llm_manager.generate_response(
                    prompt=prompt,
                    agent_type=agent_type,
                    context=context,
                    **kwargs
                )
                
                return {
                    'response': llm_response.content,
                    'rag_used': False,
                    'rag_latency_ms': 0,
                    'rag_hits': 0,
                    'rag_sources': [],
                    'model_used': llm_response.model_used,
                    'processing_time': processing_time,
                    'handoff_needed': True,  # Escalate due to error
                    'error': str(e)
                }
            except Exception as llm_error:
                self.logger.error(f"LLM fallback also failed: {llm_error}")
                return {
                    'response': "Desculpe, estou com dificuldades para responder. Por favor, tente novamente mais tarde.",
                    'rag_used': False,
                    'rag_latency_ms': 0,
                    'rag_hits': 0,
                    'rag_sources': [],
                    'model_used': 'fallback',
                    'processing_time': processing_time,
                    'handoff_needed': True,
                    'error': f"RAG: {e}, LLM: {llm_error}"
                }
    
    def _assemble_prompt_with_context(
        self,
        original_prompt: str,
        rag_chunks: List[RetrievedChunk],
        agent_type: str
    ) -> str:
        """
        Assemble enhanced prompt with RAG context.
        
        Args:
            original_prompt: Original user prompt
            rag_chunks: Retrieved chunks
            agent_type: Type of agent
            
        Returns:
            Enhanced prompt with context
        """
        # Format context from chunks
        context_parts = []
        
        # Protection against empty context
        if not rag_chunks:
            self.logger.warning("No RAG chunks available - using minimal context")
            context_text = "Nenhuma informação específica encontrada na base de conhecimento."
        else:
            for i, chunk in enumerate(rag_chunks):
                context_text = chunk.chunk.content.strip()
                
                # Add citation if debug mode is enabled
                if self.config.debug_citations:
                    citation = f" [KB: {chunk.chunk.section_title}]"
                else:
                    citation = ""
                
                context_parts.append(f"Fonte {i+1}: {context_text}{citation}")
            
            context_text = "\n\n".join(context_parts)
        
        # Build enhanced prompt
        enhanced_prompt = f"""
CONTEXTO DA BASE DE CONHECIMENTO:
{context_text}

PERGUNTA DO USUÁRIO:
{original_prompt}

INSTRUÇÕES PARA RESPOSTA:
- Responda COM BASE nas informações da base de conhecimento fornecida acima
- Se a base de conhecimento não contiver a resposta, use seu conhecimento geral mas seja claro sobre isso
- Não invente informações ou políticas que não estejam na base de conhecimento
- Seja claro e direto na resposta
- Se aplicável, mencione o WhatsApp (11) 4004-8006 como canal oficial
- Use um tom profissional e prestativo

Resposta:
"""
        
        return enhanced_prompt.strip()
    
    async def _check_and_retry_quality(
        self,
        original_prompt: str,
        response: str,
        rag_result: Any,
        agent_type: str,
        context: Optional[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Check response quality and retry if needed.
        
        Args:
            original_prompt: Original prompt
            response: Generated response
            rag_result: RAG result
            agent_type: Agent type
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with final response and metadata
        """
        if not self.config.quality_retry_enabled:
            return {'response': response, 'retry_count': 0}
        
        quality_issues = []
        retry_count = 0
        final_response = response
        
        # Check for quality issues
        if len(response) < self.config.quality_min_chars:
            quality_issues.append("response_too_short")
        
        negative_phrases = ["não sei", "não tenho", "não posso", "desculpe", "infelizmente"]
        if any(phrase.lower() in response.lower() for phrase in negative_phrases):
            quality_issues.append("negative_response")
        
        # Retry if there are issues and we have RAG hits
        if quality_issues and rag_result.chunks and retry_count < self.config.quality_max_retries:
            self.logger.info(f"Quality issues detected: {quality_issues}, retrying with higher top_k")
            
            retry_count += 1
            
            # Try with higher top_k
            retry_result = await self.rag_service.process_query(
                query=original_prompt,
                agent_type=agent_type,
                requires_rag=True,
                top_k=min(rag_result.metadata.get('top_k', 8) * 2, 20),  # Double top_k, max 20
                filters={'agent_type': agent_type, 'domain': agent_type}
            )
            
            if retry_result.chunks:
                # Re-assemble prompt with more context
                enhanced_prompt = self._assemble_prompt_with_context(
                    original_prompt=original_prompt,
                    rag_chunks=retry_result.chunks,
                    agent_type=agent_type
                )
                
                # Add retry instruction
                enhanced_prompt += "\n\nIMPORTANTE: Forneça uma resposta mais detalhada e completa baseada no contexto acima."
                
                # Generate new response
                llm_response = await self.llm_manager.generate_response(
                    prompt=enhanced_prompt,
                    agent_type=agent_type,
                    context=context,
                    **kwargs
                )
                
                final_response = llm_response.content
                
                # Re-check quality
                if len(final_response) >= self.config.quality_min_chars:
                    quality_issues = [issue for issue in quality_issues if issue != "response_too_short"]
                
                if not any(phrase.lower() in final_response.lower() for phrase in negative_phrases):
                    quality_issues = [issue for issue in quality_issues if issue != "negative_response"]
        
        # Check if handoff is needed
        handoff_needed = (
            len(quality_issues) > 0 and 
            retry_count >= self.config.quality_max_retries and
            rag_result.chunks
        )
        
        return {
            'response': final_response,
            'retry_count': retry_count,
            'quality_issues': quality_issues,
            'handoff_needed': handoff_needed,
            'model_used': getattr(self.llm_manager, 'current_model', 'gpt-4o')
        }
    
    def get_quality_metrics(self) -> QualityMetrics:
        """Get quality metrics"""
        return self.quality_metrics
    
    def reset_quality_metrics(self) -> None:
        """Reset quality metrics"""
        self.quality_metrics = QualityMetrics()
        self.logger.info("Quality metrics reset")
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of RAG integration components"""
        rag_health = await self.rag_service.health_check()
        
        return {
            'rag_service': rag_health.get('overall', False),
            'llm_manager': True,  # Assume LLM manager is healthy if we can call it
            'overall': rag_health.get('overall', False)
        }
