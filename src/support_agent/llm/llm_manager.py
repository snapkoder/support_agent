"""
LLM Manager - Versão limpa sem connection pool
Simplificado para performance e estabilidade
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import httpx
import random
import uuid

# Configuração centralizada
from support_agent.config.settings import get_llm_config, get_feature_flags, _get_env_fallback

# 🆕 EXCEÇÃO PERSONALIZADA PARA CONFIGURAÇÃO DE MODELO
class ModelConfigurationError(Exception):
    """Raised when model configuration is missing or invalid"""
    pass

# 🆕 FUNÇÃO ÚNICA DE RESOLUÇÃO DE MODELO COM INSTRUMENTAÇÃO
def resolve_model(context: Optional[Dict[str, Any]] = None) -> str:
    """
    Resolve o modelo a ser usado seguindo hierarquia obrigatória.
    
    Hierarquia:
    1. Se A/B ativo → usar modelo do experimento
    2. Senão → usar modelo primário da config
    3. Senão → usar modelo fallback da config
    4. Se nenhum definido → erro explícito
    
    Args:
        context: Contexto adicional (ex: experimento ativo)
    
    Returns:
        Nome do modelo resolvido
        
    Raises:
        ModelConfigurationError: Se nenhum modelo válido for encontrado
    """
    logger = logging.getLogger(__name__)
    
    # Obter configuração centralizada
    llm_config = get_llm_config()
    features = get_feature_flags()
    
    model_selected = None
    source = None
    
    # 1. Verificar se A/B está ativo
    if context and context.get("ab_testing_active"):
        ab_model = context.get("ab_model")
        if ab_model:
            model_selected = ab_model
            source = "experiment"
    
    # 2. Usar modelo primário da config
    if not model_selected:
        primary_model = llm_config.primary_model
        if primary_model:
            model_selected = primary_model
            source = "primary_config"
        # 3. Usar modelo fallback da config
    if not model_selected:
        fallback_model = llm_config.fallback_model
        if fallback_model:
            model_selected = fallback_model
            source = "fallback_config"
    
    # 4. Validar strict enforcement
    strict_enforcement = features.enable_model_migration  # Usar feature flag
    
    if not model_selected:
        error_msg = "No model configuration found. Check LLM configuration in config.py."
        
        # 🆕 LOG ESTRUTURADO - ERRO DE RESOLUÇÃO
        logger.error(json.dumps({
            "event": "model_resolution_failed",
            "error": error_msg,
            "strict_enforcement": strict_enforcement,
            "migration_enabled": features.enable_model_migration,
            "context_provided": context is not None,
            "timestamp": time.time()
        }))
        
        raise ModelConfigurationError(error_msg)
    
    #  LOG ESTRUTURADO - MODELO RESOLVIDO
    logger.info(json.dumps({
        "event": "model_resolved",
        "model_selected": model_selected,
        "source": source,
        "migration_enabled": features.enable_model_migration,
        "context_provided": context is not None,
        "timestamp": time.time()
    }))
    
    # VALIDAÇÃO STRICT ENFORCEMENT
    if strict_enforcement and source == "default":
        warning_msg = f"Using default model '{model_selected}' with strict enforcement enabled"
        logger.warning(json.dumps({
            "event": "strict_enforcement_warning",
            "warning": warning_msg,
            "model_selected": model_selected,
            "source": source,
            "timestamp": time.time()
        }))
    
    return model_selected

# Carregar variáveis de ambiente
from dotenv import load_dotenv
load_dotenv()

# OpenAI import (condicional)
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Provedores de LLM disponíveis"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    CLAUDE = "claude"
    FALLBACK = "fallback"

@dataclass
class GenerationConfig:
    """Configuração de geração (provider-agnostic defaults)"""
    max_tokens: int = 150
    temperature: float = 0.2
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    stream: bool = False

@dataclass
class LLMProviderConfig:
    """Configuração do provedor LLM"""
    provider: LLMProvider
    model: str
    base_url: str = ""
    max_tokens: int = 150
    temperature: float = 0.2
    timeout: float = 30.0
    generation_config: Optional[GenerationConfig] = None

@dataclass
class LLMResponse:
    """Resposta do LLM"""
    content: str
    provider: str
    model: str
    processing_time: float
    confidence: float = 0.8
    token_count: int = 0
    error: Optional[str] = None
    # 🆕 Campos de trace para rastreabilidade
    provider_used: str = ""
    fallback_used: bool = False
    request_id: str = ""
    # 🆕 CAMPOS NOVOS DE TRACE (corrigidos)
    model_selected: str = ""  # Modelo selecionado pelo resolve_model
    model_sent_to_api: str = ""  # Modelo realmente enviado à API
    model_used: str = ""  # Igual ao enviado à API (consistência garantida)
    # 🆕 CAMPOS DE INSTRUMENTAÇÃO
    trace_id: str = ""  # ID único para rastrear a requisição

class BaseLLMProvider:
    """Classe base para provedores LLM"""
    
    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def generate(self, prompt: str, context: Optional[str] = None, config: Optional[GenerationConfig] = None, model: Optional[str] = None) -> LLMResponse:
        """Gera resposta"""
        pass
    
    async def health_check(self) -> bool:
        """Verifica se o provedor está saudável"""
        pass
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Obtém informações do modelo"""
        pass

class OllamaProvider(BaseLLMProvider):
    """Provedor Ollama (local) - Simplificado sem pool"""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "")
        if not self.base_url:
            raise ValueError("OLLAMA_BASE_URL deve ser configurado no .env")
        # Usar timeout do .env
        actual_timeout = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
        self.client = httpx.AsyncClient(timeout=actual_timeout)
    
    async def generate(self, prompt: str, context: Optional[str] = None, config: Optional[GenerationConfig] = None, model: Optional[str] = None) -> LLMResponse:
        """Gera resposta direta sem pool de conexões"""
        start_time = time.time()
        
        # Usar modelo passado ou config default
        effective_model = model or self.config.model
        
        # Usar config otimizada se não fornecida
        if config is None:
            config = self.config.generation_config or GenerationConfig()
        
        # Combinar prompt e contexto
        full_prompt = prompt
        if context:
            full_prompt = f"CONTEXTO:\n{context}\n\nPERGUNTA:\n{prompt}"
        
        self.logger.info("🚀 OLLAMA REQUEST STARTED")
        self.logger.info(f"🔗 URL: {self.base_url}/api/generate")
        self.logger.info(f"📝 Model: {effective_model}")
        self.logger.info(f"⚙️ Config: tokens={config.max_tokens}, temp={config.temperature}")
        
        try:
            # Requisição direta ao Ollama
            payload = {
                "model": effective_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "repeat_penalty": config.repeat_penalty,
                    "num_predict": config.max_tokens
                }
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                processing_time = time.time() - start_time
                self.logger.info(f"✅ OLLAMA RESPONSE SUCCESS - {processing_time:.3f}s")
                
                return LLMResponse(
                    content=response_text.strip(),
                    model=effective_model,
                    provider="ollama",
                    processing_time=processing_time,
                    token_count=len(response_text.split()),
                    confidence=0.8,
                    # 🆕 CAMPOS DE TRACE COMPLETOS
                    trace_id=f"ollama_{uuid.uuid4().hex[:12]}",
                    model_selected=effective_model,
                    model_sent_to_api=effective_model,
                    model_used=effective_model
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self.logger.error(f"❌ OLLAMA RESPONSE ERROR: {error_msg}")
                raise Exception(f"Ollama request failed: {error_msg}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ OLLAMA REQUEST FAILED: {e} - {processing_time:.3f}s")
            raise Exception(f"Failed to generate response: {e}")
    
    async def health_check(self) -> bool:
        """Verifica se o Ollama está saudável"""
        try:
            response = await self.client.get(f"{self.base_url}/api/version")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Obtém informações do modelo Ollama"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/show",
                json={"name": self.config.model}
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {}

class FallbackProvider(BaseLLMProvider):
    """Provedor Fallback baseado em regras"""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.responses = self._load_fallback_responses()
    
    def _load_fallback_responses(self) -> Dict[str, List[str]]:
        """Carrega respostas fallback"""
        return {
            "atendimento_geral": [
                "Olá! Sou o assistente do Jota. Como posso te ajudar hoje?",
                "Olá! Esse é o canal oficial de atendimento do Jota. Estou aqui para ajudar!",
                "Olá! Para te ajudar melhor, preciso entender sua necessidade. O que você precisa?"
            ],
            "criacao_conta": [
                "Olá! Para abrir sua conta no Jota, entre em contato conosco pelo WhatsApp no número (11) 4004-8006.",
                "Olá! Sou o especialista em criação de contas do Jota. Vou te ajudar a abrir sua conta!",
                "Olá! Para abrir sua conta, envie uma mensagem para o Jota no WhatsApp e siga as instruções."
            ],
            "open_finance": [
                "Olá! Sou o especialista em Open Finance do Jota. Como posso te ajudar com sua conexão bancária?",
                "Olá! Para conectar seu banco, vou te ajudar com o processo. Primeiro, qual banco você quer conectar?",
                "Olá! Vou te orientar passo a passo na conexão do seu banco com o Jota."
            ],
            "golpe_med": [
                "Olá! Sinto muito pelo ocorrido. Vou te ajudar com isso. Para começar, preciso de algumas informações.",
                "Olá! Entendo sua situação. Vou te ajudar com o processo de contestação do PIX.",
                "Olá! Para te ajudar com o golpe, preciso saber o que aconteceu. Me conte mais detalhes."
            ]
        }
    
    async def generate(self, prompt: str, context: Optional[str] = None, config: Optional[GenerationConfig] = None, model: Optional[str] = None) -> LLMResponse:
        """Gera resposta baseada em regras"""
        start_time = time.time()
        
        # Model é opcional para fallback (ignorado internamente)
        _ = model  # Ignorar modelo, usar rule-based
        
        # Tentar identificar o tipo de agente pelo prompt
        agent_type = "atendimento_geral"  # Default
        
        if "golpe" in prompt.lower() or "fraude" in prompt.lower() or "pix" in prompt.lower():
            agent_type = "golpe_med"
        elif "conta" in prompt.lower() or "abrir" in prompt.lower() or "criar" in prompt.lower():
            agent_type = "criacao_conta"
        elif "banco" in prompt.lower() or "conectar" in prompt.lower() or "open finance" in prompt.lower():
            agent_type = "open_finance"
        
        # RNG local para determinismo sem afetar estado global (thread-safe)
        rng = random.Random(hash(prompt) % (2**32))
        responses = self.responses.get(agent_type, self.responses["atendimento_geral"])
        response = rng.choice(responses)
        
        return LLMResponse(
            content=response,
            provider="fallback",
            model="rule_based",
            processing_time=time.time() - start_time,
            confidence=0.5,
            # 🆕 CAMPOS DE TRACE COMPLETOS
            trace_id=f"fallback_{uuid.uuid4().hex[:12]}",
            model_selected="rule_based",
            model_sent_to_api="rule_based",
            model_used="rule_based"
        )
    
    async def health_check(self) -> bool:
        """Fallback sempre está saudável"""
        return True
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Informações do modelo fallback"""
        return {"name": "rule_based", "provider": "fallback", "status": "active"}

class OpenAIProvider(BaseLLMProvider):
    """Provedor OpenAI (nuvem) - Primário com fallback para Ollama"""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        # Usar configuração centralizada
        llm_config = get_llm_config()
        
        # 🚨 VALIDAÇÃO: API key obrigatória
        api_key = llm_config.openai_api_key
        if not api_key or api_key.startswith("YOUR_"):
            raise ModelConfigurationError("OPENAI_API_KEY is required and cannot be a placeholder")
        
        # 🚨 CORREÇÃO CRÍTICA: Inicializar self.client corretamente
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=llm_config.openai_base_url,
            timeout=float(llm_config.request_timeout)  # Timeout no client, não no payload
        )
        
        self.default_temperature = llm_config.temperature
        self.default_max_tokens = llm_config.max_tokens
        
        self.logger.info(f"🚀 OpenAI Provider initialized: client configured with timeout={llm_config.request_timeout}s")
    
    async def generate(self, prompt: str, context: Optional[str] = None, config: Optional[GenerationConfig] = None, model: Optional[str] = None) -> LLMResponse:
        """Gera resposta usando OpenAI API
        
        Args:
            prompt: Texto do prompt
            context: Contexto adicional (system message)
            config: Configuração de geração
            model: Modelo a ser usado (OBRIGATÓRIO - provider é stateless)
        """
        start_time = time.time()
        
        # 🚨 VALIDAÇÃO: Modelo é obrigatório pois provider é stateless
        if not model:
            raise ModelConfigurationError("Model parameter is required for stateless provider")
        
        # 🚨 CORREÇÃO DE CONCORRÊNCIA: Usar UUID4 para garantir unicidade
        trace_id = f"openai_{uuid.uuid4().hex[:12]}"
        
        try:
            # Construir mensagens
            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": prompt})
            
            # Parâmetros da requisição
            generation_config = config or GenerationConfig()
            
            # 🆕 CORREÇÃO PARA MODELOS MAIS RECENTES (gpt-5.1)
            # gpt-5.1 usa max_completion_tokens em vez de max_tokens
            if model.startswith("gpt-5"):
                request_params = {
                    "model": model,  # 🚨 USAR PARÂMETRO RECEBIDO
                    "messages": messages,
                    "temperature": float(generation_config.temperature),
                    "max_completion_tokens": int(generation_config.max_tokens)
                }
                max_tokens_param = "max_completion_tokens"
            else:
                request_params = {
                    "model": model,  # 🚨 USAR PARÂMETRO RECEBIDO
                    "messages": messages,
                    "temperature": float(generation_config.temperature),  # 🆕 Converter para float
                    "max_tokens": int(generation_config.max_tokens)  # 🆕 Converter para int
                }
                max_tokens_param = "max_tokens"
            
            # 🆕 LOG ESTRUTURADO - ANTES DA CHAMADA API
            self.logger.info(json.dumps({
                "event": "llm_request",
                "provider": "openai",
                "model_sent_to_api": model,  # 🚨 USAR PARÂMETRO RECEBIDO
                "trace_id": trace_id,
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "context_provided": context is not None,
                "temperature": generation_config.temperature,
                max_tokens_param: generation_config.max_tokens,
                "timestamp": time.time()
            }))
            
            # Chamada à API
            response = await self.client.chat.completions.create(**request_params)
            
            # Extrair resposta
            content = response.choices[0].message.content or ""
            
            processing_time = time.time() - start_time
            
            # Token usage (se disponível)
            token_count = 0
            if response.usage:
                token_count = response.usage.total_tokens or 0
            
            # 🆕 LOG ESTRUTURADO - APÓS RESPOSTA API
            self.logger.info(json.dumps({
                "event": "llm_response",
                "provider": "openai",
                "model_used": model,  # 🚨 USAR PARÂMETRO RECEBIDO
                "trace_id": trace_id,
                "response_length": len(content),
                "processing_time": processing_time,
                "token_count": token_count,
                "timestamp": time.time()
            }))
            
            return LLMResponse(
                content=content,
                model=model,  # 🚨 USAR PARÂMETRO RECEBIDO
                provider="openai",
                processing_time=processing_time,
                token_count=token_count,
                # 🆕 CAMPOS DE TRACE COMPLETOS
                trace_id=trace_id,
                model_selected=model,  # Selecionado = enviado (stateless)
                model_sent_to_api=model,  # Enviado à API
                model_used=model  # Retornado pela API
            )
            
        except asyncio.TimeoutError:
            error_msg = f"OpenAI timeout after {self.client.timeout}s"
            self.logger.error(error_msg)
            raise
            
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {str(e)}"
            self.logger.error(error_msg)
            raise
            
        except Exception as e:
            error_msg = f"OpenAI unexpected error: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    async def health_check(self, model: str = "gpt-4o") -> bool:
        """Verifica se OpenAI está acessível
        
        Args:
            model: Modelo a ser usado no health check (padrão: gpt-4o)
        """
        try:
            response = await self.client.chat.completions.create(
                model=model,  # 🚨 USAR PARÂMETRO RECEBIDO
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=float(os.getenv("LLM_HEALTH_CHECK_TIMEOUT", "5.0"))
            )
            return True
        except Exception as e:
            self.logger.warning(f"OpenAI health check failed: {e}")
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Obtém informações do provider OpenAI (stateless)"""
        return {
            "name": "stateless_openai_provider",
            "provider": "openai",
            "status": "active",
            "model": "passed_per_call",  # 🚨 STATELESS - modelo passado a cada chamada
            "api_base": self.client.base_url
        }

# LLM Manager simplificado sem pool
class LLMManager:
    """Gerenciador de LLM simplificado sem connection pool"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.LLMManager")
        self.providers = {}
        self.primary_provider = None
        self.fallback_provider = None
        self.fallback_enabled = True
        
        # Métricas
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "total_processing_time": 0.0
        }
    
    async def initialize(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """Inicializa os provedores LLM com modelo resolvido explicitamente"""
        try:
            # 🆕 USAR FUNÇÃO ÚNICA DE RESOLUÇÃO DE MODELO
            self.resolved_model = resolve_model(context)
            self.logger.info(f"🎯 MODEL_RESOLVED: {self.resolved_model}")
            
            # 🚨 ARMAZENAR CONTEXTO para uso em chamadas subsequentes
            self._context = context
            
            # Manter flags para compatibilidade, mas não mais usar para seleção
            self.migration_enabled = _get_env_fallback("LLM_MIGRATION_ENABLED", "false").lower() == "true"
            self.canary_percent = int(_get_env_fallback("LLM_CANARY_PERCENT", "0"))
            
            # Validação de configuração
            if self.canary_percent < 0 or self.canary_percent > 100:
                self.logger.warning(f"LLM_CANARY_PERCENT inválido: {self.canary_percent}, usando 0")
                self.canary_percent = 0
            
            # Log de configuração
            self.logger.info(f"🚀 CONFIG: resolved_model={self.resolved_model}, migration_enabled={self.migration_enabled}, canary={self.canary_percent}%")
            
            # Configuração efetiva
            effective_config = {
                "primary_provider": _get_env_fallback("LLM_PRIMARY_PROVIDER", "openai"),
                "fallback_provider": _get_env_fallback("LLM_FALLBACK_PROVIDER", "ollama"),
                "generation_model": _get_env_fallback("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
                "classification_model": _get_env_fallback("OLLAMA_CLASSIFICATION_MODEL", "qwen2.5:7b-instruct"),
                "max_tokens": _get_env_fallback("OLLAMA_MAX_TOKENS", "150"),
                "timeout": _get_env_fallback("OLLAMA_TIMEOUT", "45.0"),
                "temperature": _get_env_fallback("OLLAMA_TEMPERATURE", "0.2"),
                "retrieval_top_k": _get_env_fallback("RETRIEVAL_TOP_K", "8"),
                "context_top_k": _get_env_fallback("CONTEXT_TOP_K_FINAL", "3"),
                "base_url": _get_env_fallback("OLLAMA_BASE_URL", "http://localhost:11434"),
                # 🆕 MODELO RESOLVIDO EXPLICITAMENTE
                "resolved_model": self.resolved_model,
                "openai_api_key": _get_env_fallback("OPENAI_API_KEY", ""),
                "openai_base_url": _get_env_fallback("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "openai_timeout": _get_env_fallback("OPENAI_TIMEOUT", "30.0"),
                "openai_max_tokens": _get_env_fallback("OPENAI_MAX_TOKENS", "200"),
                "openai_temperature": _get_env_fallback("OPENAI_TEMPERATURE", "0.2"),
                # Configurações de migração (mantidas para compatibilidade)
                "migration_enabled": self.migration_enabled,
                "canary_percent": self.canary_percent
            }
            _SENSITIVE = ("key", "token", "secret", "password", "authorization")
            safe_config = {
                k: ("***" if any(s in k.lower() for s in _SENSITIVE) and "max_" not in k.lower() else v)
                for k, v in effective_config.items()
            }
            self.logger.info(f"🔧 CONFIGURAÇÃO EFETIVA: {safe_config}")
            
            # Configurar provedor primário (OpenAI se disponível)
            primary_provider_name = effective_config["primary_provider"].lower()
            
            if primary_provider_name == "openai" and effective_config["openai_api_key"]:
                try:
                    # Configuração OpenAI
                    generation_config = GenerationConfig(
                        max_tokens=int(effective_config["openai_max_tokens"]),
                        temperature=float(effective_config["openai_temperature"]),
                        top_p=float(_get_env_fallback("OLLAMA_TOP_P", "0.9")),
                        repeat_penalty=float(_get_env_fallback("OLLAMA_REPEAT_PENALTY", "1.1")),
                        stream=False
                    )
                    
                    openai_config = LLMProviderConfig(
                        provider=LLMProvider.OPENAI,
                        model=self.resolved_model,  # 🆕 USAR MODELO RESOLVIDO
                        base_url=effective_config["openai_base_url"],
                        max_tokens=int(effective_config["openai_max_tokens"]),
                        temperature=float(effective_config["openai_temperature"]),
                        timeout=float(effective_config["openai_timeout"]),
                        generation_config=generation_config
                    )
                    
                    self.logger.info(f"🚀 OpenAI configured: model={openai_config.model}, timeout={openai_config.timeout}s, tokens={openai_config.max_tokens}")
                    openai_provider = OpenAIProvider(openai_config)
                    self.providers[LLMProvider.OPENAI] = openai_provider
                    self.primary_provider = openai_provider
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize OpenAI: {e}")
                    primary_provider_name = "ollama"  # Fallback para Ollama
            
            # Fallback para Ollama se OpenAI falhou ou não é primário
            if not self.primary_provider and primary_provider_name == "ollama":
                if os.getenv("OLLAMA_ENABLED", "true").lower() == "true":
                    # Configuração otimizada para performance local
                    generation_config = GenerationConfig(
                        max_tokens=int(effective_config["max_tokens"]),
                        temperature=float(effective_config["temperature"]),
                        top_p=float(_get_env_fallback("OLLAMA_TOP_P", "0.9")),
                        repeat_penalty=float(_get_env_fallback("OLLAMA_REPEAT_PENALTY", "1.1")),
                        stream=False
                    )
                    
                    ollama_config = LLMProviderConfig(
                        provider=LLMProvider.OLLAMA,
                        model=effective_config["generation_model"],
                        base_url=effective_config["base_url"],
                        max_tokens=int(effective_config["max_tokens"]),
                        temperature=float(effective_config["temperature"]),
                        timeout=float(effective_config["timeout"]),
                        generation_config=generation_config
                    )
                    
                    self.logger.info(f"🚀 OLLAMA configured: model={ollama_config.model}, timeout={ollama_config.timeout}s, tokens={ollama_config.max_tokens}")
                    ollama_provider = OllamaProvider(ollama_config)
                    self.providers[LLMProvider.OLLAMA] = ollama_provider
                    self.primary_provider = ollama_provider
            
            # Configurar provedor de fallback (Ollama se OpenAI é primário)
            if self.primary_provider and self.primary_provider.config.provider != LLMProvider.OLLAMA:
                if os.getenv("OLLAMA_ENABLED", "true").lower() == "true":
                    # Configurar Ollama como fallback
                    generation_config = GenerationConfig(
                        max_tokens=int(effective_config["max_tokens"]),
                        temperature=float(effective_config["temperature"]),
                        top_p=float(_get_env_fallback("OLLAMA_TOP_P", "0.9")),
                        repeat_penalty=float(_get_env_fallback("OLLAMA_REPEAT_PENALTY", "1.1")),
                        stream=False
                    )
                    
                    fallback_config = LLMProviderConfig(
                        provider=LLMProvider.OLLAMA,
                        model=effective_config["generation_model"],
                        base_url=effective_config["base_url"],
                        max_tokens=int(effective_config["max_tokens"]),
                        temperature=float(effective_config["temperature"]),
                        timeout=float(effective_config["timeout"]),
                        generation_config=generation_config
                    )
                    
                    ollama_fallback = OllamaProvider(fallback_config)
                    self.providers[LLMProvider.OLLAMA] = ollama_fallback
                    self.fallback_provider = ollama_fallback
                    self.logger.info("🔄 OLLAMA configured as fallback")
            
            # Fallback rule-based (sempre disponível)
            if os.getenv("FALLBACK_ENABLED", "true").lower() == "true":
                fallback_config = LLMProviderConfig(
                    provider=LLMProvider.FALLBACK,
                    model="rule_based",
                    timeout=float(os.getenv("LLM_FALLBACK_TIMEOUT", "1.0"))
                )
                self.fallback_rule_provider = FallbackProvider(fallback_config)
                self.logger.info("🔄 Rule-based fallback configured")
            
            # Verificar se temos pelo menos um provedor
            if not self.primary_provider:
                self.logger.error("❌ No primary LLM provider available")
                return False
            
            self.logger.info(f"✅ LLM Manager initialized: primary={self.primary_provider.config.provider.value}")
            
            # 🆕 PARTE 2: Validar modelo disponível antes do deploy
            await self._validate_model_availability()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing LLM Manager: {e}")
            return False
    
    async def _validate_model_availability(self):
        """Valida se o modelo primário está acessível"""
        if not self.migration_enabled:
            self.logger.info("🔍 MIGRATION_DISABLED: Usando apenas fallback model")
            return
        
        try:
            # Testar modelo primário
            # 🆕 CORREÇÃO: Usar resolved_model em vez de primary_model
            self.logger.info(f"🔍 VALIDATING_PRIMARY_MODEL: {self.resolved_model}")
            
            # Criar provider temporário para validação
            temp_config = LLMProviderConfig(
                provider=LLMProvider.OPENAI,
                model=self.resolved_model,  # Usar resolved_model
                base_url=_get_env_fallback("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                max_tokens=1,
                timeout=10.0  # Timeout curto para validação
            )
            
            temp_provider = OpenAIProvider(temp_config)
            validation_response = await temp_provider.generate("test", model=self.resolved_model)
            
            if validation_response.error:
                self.logger.error(f"❌ PRIMARY_MODEL_UNAVAILABLE: {self.resolved_model} - {validation_response.error}")
                self.logger.warning(f"🔄 FALLBACK_TO: {self.fallback_model}")
                self.migration_enabled = False
            else:
                self.logger.info(f"✅ PRIMARY_MODEL_AVAILABLE: {self.resolved_model}")
                
        except Exception as e:
            self.logger.error(f"❌ PRIMARY_MODEL_VALIDATION_FAILED: {self.resolved_model} - {e}")
            self.logger.warning(f"🔄 FALLBACK_TO: {self.fallback_model}")
            self.migration_enabled = False
    
    def _select_model_for_request(self, context: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
        """
        🆕 RETORNA MODELO RESOLVIDO - sempre atualizado
        Returns: (model_name, model_type)
        """
        # 🚨 CRÍTICO: Chamar resolve_model() a cada requisição com contexto
        # Isso garante que mudanças de environment e A/B testing sejam respeitadas
        current_model = resolve_model(context)
        return current_model, "RESOLVED"
    
    async def warmup_models(self) -> bool:
        """
        Warmup obrigatório para eliminar cold start do Ollama
        Executa geração mínima dummy para manter modelo residente
        """
        try:
            self.logger.info("🔥 Iniciando LLM warmup...")
            
            # Modelo de geração principal
            main_model = os.getenv("OLLAMA_MODEL")
            classification_model = os.getenv("OLLAMA_CLASSIFICATION_MODEL")
            
            warmup_success = True
            
            # Warmup modelo principal
            try:
                if self.primary_provider:
                    warmup_config = GenerationConfig(
                        max_tokens=1,  # Mínimo possível
                        temperature=0.0,
                        top_p=0.9,
                        repeat_penalty=1.1,
                        stream=False
                    )
                    
                    start_time = time.time()
                    warmup_response = await self.primary_provider.generate("warmup", config=warmup_config, model=main_model)
                    warmup_time = time.time() - start_time
                    
                    self.logger.info(f"✅ LLM warmup concluído para: {main_model} ({warmup_time:.2f}s)")
                    
            except Exception as e:
                self.logger.error(f"❌ Falha no warmup do modelo principal {main_model}: {e}")
                warmup_success = False
            
            # Warmup modelo de classificação
            try:
                classification_config = GenerationConfig(
                    max_tokens=1,  # Mínimo possível
                    temperature=0.0,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    stream=False
                )
                
                # Criar provider temporário para classificação
                temp_config = LLMProviderConfig(
                    provider=LLMProvider.OLLAMA,
                    model=classification_model,
                    base_url=os.getenv("OLLAMA_BASE_URL"),
                    max_tokens=1,
                    temperature=0.0,
                    timeout=float(os.getenv("OLLAMA_CLASSIFICATION_TIMEOUT", "15.0")),
                    generation_config=classification_config
                )
                
                temp_provider = OllamaProvider(temp_config)
                start_time = time.time()
                warmup_response = await temp_provider.generate("warmup", config=classification_config, model=classification_model)
                warmup_time = time.time() - start_time
                
                self.logger.info(f"✅ LLM warmup concluído para: {classification_model} ({warmup_time:.2f}s)")
                
            except Exception as e:
                self.logger.error(f"❌ Falha no warmup do modelo de classificação {classification_model}: {e}")
                warmup_success = False
            
            # Validar se modelo está residente
            try:
                if self.primary_provider:
                    health_ok = await self.primary_provider.health_check()
                    if health_ok:
                        self.logger.info("🎯 Modelo verificado como residente via health check")
                    else:
                        self.logger.warning("⚠️ Health check falhou após warmup")
            except Exception as e:
                self.logger.warning(f"⚠️ Não foi possível verificar residência do modelo: {e}")
            
            return warmup_success
            
        except Exception as e:
            self.logger.error(f"Erro no warmup dos modelos: {e}")
            return False
    
    async def generate_classification(self, prompt: str) -> LLMResponse:
        """Gera classificação usando provider primário (OpenAI ou fallback)"""
        start_time = time.time()
        self.metrics["requests_total"] += 1
        
        try:
            # Criar config específica para classificação
            classification_config = GenerationConfig(
                max_tokens=5,  # Permitir resposta completa
                temperature=0.0,  # Máximo determinismo
                top_p=1.0,  # Sem amostragem
                repeat_penalty=1.0,  # Sem penalidade
                stream=False
            )
            
            # 🆕 Usar provider primário em vez de Ollama hardcoded
            if not self.primary_provider:
                raise Exception("No primary provider available")
            
            # 🚨 CORREÇÃO DE CONCORRÊNCIA: Usar UUID4 para garantir unicidade
            request_id = f"class_{uuid.uuid4().hex[:12]}"
            self.logger.info(json.dumps({
                "event": "classification_start",
                "request_id": request_id,
                "provider": self.primary_provider.config.provider.value,
                "prompt_preview": prompt[:50]
            }))
            
            # Gerar usando provider primário
            response = await self.primary_provider.generate(prompt, config=classification_config, model=self.resolved_model)
            
            self.logger.info(json.dumps({
                "event": "classification_success",
                "request_id": request_id,
                "provider": self.primary_provider.config.provider.value,
                "response": response.content,
                "processing_time": response.processing_time
            }))
            
            # 🆕 Retornar com trace fields
            return LLMResponse(
                content=response.content,
                model=response.model,
                provider=response.provider,
                processing_time=response.processing_time,
                token_count=getattr(response, 'token_count', 0),
                provider_used=self.primary_provider.config.provider.value,
                fallback_used=False,
                request_id=request_id
            )
            
        except Exception as e:
            self.logger.warning(f"Classification provider failed: {e}")
            
            # Fallback para provider secundário se disponível
            if self.fallback_provider:
                try:
                    fallback_response = await self.fallback_provider.generate(prompt, config=classification_config, model=self.resolved_model)
                    self.metrics["requests_successful"] += 1
                    self.metrics["total_processing_time"] += time.time() - start_time
                    
                    # Retornar fallback com trace
                    return LLMResponse(
                        content=fallback_response.content,
                        model=fallback_response.model,
                        provider=fallback_response.provider,
                        processing_time=fallback_response.processing_time,
                        token_count=getattr(fallback_response, 'token_count', 0),
                        provider_used=self.fallback_provider.config.provider.value,
                        fallback_used=True,
                        request_id=request_id
                    )
                except Exception as fallback_error:
                    self.logger.error(f"Fallback classification also failed: {fallback_error}")
            
            # Retornar erro
            self.metrics["requests_failed"] += 1
            self.metrics["total_processing_time"] += time.time() - start_time
            
            return LLMResponse(
                content="atendimento_geral",  # Fallback seguro
                provider="error",
                model="error",
                error=str(e),
                processing_time=time.time() - start_time,
                provider_used="error",
                fallback_used=False,
                request_id=request_id
            )
    
    async def generate_response(self, prompt: str, agent_type: str = "atendimento_geral", context: Optional[Dict[str, Any]] = None, **kwargs) -> LLMResponse:
        """Gera resposta usando provedor primário com fallback controlado e seleção de modelo"""
        start_time = time.time()
        self.metrics["requests_total"] += 1
        
        # 🆕 PARTE 1: Selecionar modelo para este request
        # Usar contexto passado ou contexto armazenado na inicialização
        effective_context = context or getattr(self, '_context', None)
        selected_model, model_type = self._select_model_for_request(effective_context)
        
        # Circuit breaker state
        circuit_breaker_state = getattr(self, '_circuit_breaker_state', {
            'openai_failures': 0,
            'ollama_failures': 0,
            'openai_unhealthy_until': 0,
            'ollama_unhealthy_until': 0
        })
        
        # 🚨 CORREÇÃO DE CONCORRÊNCIA: Usar UUID4 para garantir unicidade
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        
        # 🆕 Armazenar request_id e model_source para uso em tratamento de erro
        self._current_request_id = request_id
        
        # Determinar source do modelo para logs de erro
        # Usar informação já armazenada em _model_source ou determinar aqui
        if not hasattr(self, '_model_source'):
            if os.getenv("LLM_MODEL_PRIMARY"):
                self._model_source = "primary"
            elif os.getenv("LLM_MODEL_DEFAULT"):
                self._model_source = "default"
            else:
                self._model_source = "unknown"
        
        self.logger.info(json.dumps({
            "event": "llm_request_start",
            "request_id": request_id,
            "agent_type": agent_type,
            "primary_provider": self.primary_provider.config.provider.value if self.primary_provider else "none",
            "has_fallback": self.fallback_provider is not None,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            # 🆕 Campos de migração
            "model_used": selected_model,
            "model_type": model_type,
            "model_source": self._model_source,
            "migration_enabled": self.migration_enabled,
            "canary_percent": self.canary_percent
        }))
        
        # Tentar provedor primário
        primary_provider_name = self.primary_provider.config.provider.value
        fallback_used = False
        provider_used = primary_provider_name
        error_type = None
        
        try:
            # Verificar circuit breaker
            if primary_provider_name == "openai":
                if circuit_breaker_state['openai_unhealthy_until'] > time.time():
                    self.logger.warning(f"⚠️ OpenAI circuit breaker open, using fallback")
                    raise Exception("OpenAI unhealthy - circuit breaker")
            
            # Filtrar parâmetros incompatíveis
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'max_tokens'}
            
            # 🚨 PASSAR MODELO EXPLICITAMENTE - Provider agora é stateless
            # Usar config do provider ativo (respeita OPENAI_MAX_TOKENS / OLLAMA_MAX_TOKENS)
            generation_config = kwargs.get('config') or getattr(self.primary_provider.config, 'generation_config', None) or GenerationConfig()
            response = await self.primary_provider.generate(prompt, model=selected_model, config=generation_config, **filtered_kwargs)
            
            # Reset circuit breaker em sucesso
            if primary_provider_name == "openai":
                circuit_breaker_state['openai_failures'] = 0
                circuit_breaker_state['openai_unhealthy_until'] = 0
            
            self.metrics["requests_successful"] += 1
            self.metrics["total_processing_time"] += response.processing_time
            
            # 🆕 VALIDAÇÃO AUTOMÁTICA DE MODELO
            strict_enforcement = os.getenv("LLM_STRICT_MODEL_ENFORCEMENT", "false").lower() == "true"
            
            if strict_enforcement:
                # Validar consistência entre modelo selecionado e modelo usado
                if response.model_used and response.model_used != selected_model:
                    # 🚨 ERRO CRÍTICO - MODELO MISMATCH
                    error_msg = f"MODEL MISMATCH: selected={selected_model}, used={response.model_used}"
                    
                    self.logger.error(json.dumps({
                        "event": "model_mismatch",
                        "error": error_msg,
                        "selected_model": selected_model,
                        "model_sent_to_api": response.model_sent_to_api,
                        "model_used": response.model_used,
                        "trace_id": getattr(response, 'trace_id', ''),
                        "provider_used": provider_used,
                        "request_id": request_id,
                        "timestamp": time.time()
                    }))
                    
                    raise ModelConfigurationError(error_msg)
            
            # Log estruturado do sucesso
            self.logger.info(json.dumps({
                "event": "llm_request_success",
                "request_id": request_id,
                "provider_used": provider_used,
                "fallback_used": fallback_used,
                "response_length": len(response.content),
                "processing_time": response.processing_time,
                "token_count": getattr(response, 'token_count', 0),
                # 🆕 CAMPOS DE VALIDAÇÃO
                "selected_model": selected_model,
                "model_sent_to_api": getattr(response, 'model_sent_to_api', ''),
                "model_used": getattr(response, 'model_used', ''),
                "trace_id": getattr(response, 'trace_id', ''),
                "strict_enforcement": strict_enforcement
            }))
            
            # 🆕 Retornar com trace fields atualizados
            return LLMResponse(
                content=response.content,
                model=response.model,
                provider=response.provider,
                processing_time=response.processing_time,
                token_count=getattr(response, 'token_count', 0),
                provider_used=provider_used,
                fallback_used=fallback_used,
                request_id=request_id,
                # 🆕 CAMPOS NOVOS DE TRACE
                model_selected=selected_model,
                model_sent_to_api=response.model,
                model_used=response.model  # Igual ao enviado à API
            )
            
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e).lower()
            self.logger.warning(f"⚠️ Primary provider ({primary_provider_name}) failed: {e}")
            
            # 🚨 ELIMINAR FALLBACK PARA ERROS DE MODELO
            # Classificar erros de modelo inválido/inexistente
            model_error_indicators = [
                "model_not_found", "invalid_request_error", "invalid model",
                "does not exist", "unrecognized", "unknown", "404", "400"
            ]
            
            # Verificar se é erro de modelo baseado no tipo e mensagem
            is_model_error = (
                any(indicator in error_message for indicator in model_error_indicators) or
                error_type in ["APIError"] and any(indicator in error_message for indicator in model_error_indicators)
            )
            
            if is_model_error:
                # 🚨 ERRO DE MODELO - NÃO FAZER FALLBACK
                request_id = getattr(self, '_current_request_id', 'unknown')
                
                # Log estruturado do erro de modelo
                self.logger.error(json.dumps({
                    "event": "model_resolution_failed",
                    "error_type": "invalid_model",
                    "model_solicited": selected_model,
                    "error_message": str(e),
                    "request_id": request_id,
                    "provider": primary_provider_name,
                    "timestamp": time.time(),
                    "fallback_disabled": True
                }))
                
                # Lançar erro explícito sem fallback
                raise ModelConfigurationError(
                    f"Invalid model '{selected_model}' (source: {self._model_source}). "
                    f"API Error: {str(e)}. Request ID: {request_id}"
                )
            
            # Verificar se deve fazer fallback (apenas para erros transitórios)
            should_fallback = False
            
            if primary_provider_name == "openai":
                # Incrementar falhas
                circuit_breaker_state['openai_failures'] += 1
                
                # Tipos de erro que permitem fallback
                fallback_errors = [
                    "TimeoutError", "APIError", "ConnectionError", 
                    "HTTPError", "RateLimitError", "openai unhealthy"
                ]
                
                if any(err in error_type for err in fallback_errors):
                    should_fallback = True
                    
                    # Circuit breaker: 3 falhas seguidas = 60s sem tentar OpenAI
                    if circuit_breaker_state['openai_failures'] >= 3:
                        circuit_breaker_state['openai_unhealthy_until'] = time.time() + 60
                        self.logger.error("🚨 OpenAI circuit breaker OPEN for 60s")
            
            # Tentar fallback se disponível e apropriado
            if should_fallback and self.fallback_provider:
                fallback_used = True
                provider_used = self.fallback_provider.config.provider.value
                
                try:
                    self.logger.info(f"🔄 Attempting fallback to {provider_used}")
                    
                    # Criar config default se não fornecida
                    generation_config = kwargs.get('config') or GenerationConfig()
                    response = await self.fallback_provider.generate(prompt, model=selected_model, config=generation_config)
                    
                    # Reset counter do fallback em sucesso
                    if provider_used == "ollama":
                        circuit_breaker_state['ollama_failures'] = 0
                    
                    self.metrics["requests_successful"] += 1
                    self.metrics["total_processing_time"] += response.processing_time
                    
                    # Log estruturado do fallback
                    self.logger.info(json.dumps({
                        "event": "llm_fallback_success",
                        "request_id": request_id,
                        "primary_provider": primary_provider_name,
                        "fallback_provider": provider_used,
                        "primary_error": error_type,
                        "response_length": len(response.content),
                        "processing_time": response.processing_time
                    }))
                    
                    # 🆕 Retornar fallback com trace completo
                    return LLMResponse(
                        content=response.content,
                        model=response.model,
                        provider=response.provider,
                        processing_time=response.processing_time,
                        token_count=getattr(response, 'token_count', 0),
                        provider_used=provider_used,
                        fallback_used=True,
                        request_id=request_id,
                        # 🆕 CAMPOS DE TRACE
                        model_selected=selected_model,
                        model_sent_to_api=response.model,
                        model_used=response.model
                    )
                    
                except Exception as fallback_error:
                    self.logger.error(f"❌ Fallback provider also failed: {fallback_error}")
                    
                    # Incrementar falhas do fallback
                    if provider_used == "ollama":
                        circuit_breaker_state['ollama_failures'] += 1
            
            # Último recurso: rule-based fallback
            if hasattr(self, 'fallback_rule_provider'):
                self.logger.warning("🆘 Using rule-based fallback as last resort")
                try:
                    response = await self.fallback_rule_provider.generate(prompt)
                    
                    self.logger.info(json.dumps({
                        "event": "llm_rule_based_success",
                        "request_id": request_id,
                        "primary_provider": primary_provider_name,
                        "fallback_provider": "rule_based",
                        "primary_error": error_type
                    }))
                    
                    # 🆕 Retornar rule-based com trace
                    return LLMResponse(
                        content=response.content,
                        model=response.model,
                        provider=response.provider,
                        processing_time=response.processing_time,
                        token_count=getattr(response, 'token_count', 0),
                        provider_used="rule_based",
                        fallback_used=True,
                        request_id=request_id
                    )
                    
                except Exception as rule_error:
                    self.logger.error(f"❌ Even rule-based fallback failed: {rule_error}")
            
            # Se chegou aqui, tudo falhou
            self.metrics["requests_failed"] += 1
            self.metrics["total_processing_time"] += time.time() - start_time
            
            self.logger.error(json.dumps({
                "event": "llm_request_failed",
                "request_id": request_id,
                "primary_provider": primary_provider_name,
                "error_type": error_type,
                "error_message": str(e),
                "total_failures": 1
            }))
            
            # Retornar resposta de erro
            return LLMResponse(
                content="Desculpe, estou com dificuldades técnicas no momento. Por favor, tente novamente em alguns instantes.",
                model="error",
                provider="none",
                processing_time=time.time() - start_time,
                token_count=0,
                provider_used="none",
                fallback_used=False,
                request_id=request_id
            )
        
        finally:
            # Salvar estado do circuit breaker
            self._circuit_breaker_state = circuit_breaker_state
    
    async def health_check(self) -> bool:
        """Verifica saúde dos provedores"""
        if self.primary_provider:
            return await self.primary_provider.health_check()
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do LLM Manager"""
        return self.metrics.copy()

# Função global para obter LLM Manager
_llm_manager_instance = None

async def get_llm_manager(preferred_provider: str = "ollama") -> LLMManager:
    """Obtém instância do LLM Manager"""
    global _llm_manager_instance
    if _llm_manager_instance is None:
        _llm_manager_instance = LLMManager()
        await _llm_manager_instance.initialize()
    return _llm_manager_instance
