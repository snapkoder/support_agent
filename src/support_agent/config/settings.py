"""
Configuration Layer - Single Source of Truth
Centralized, typed, validated configuration management
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dataclasses import dataclass
import logging
from support_agent.security.redact import redact_dict, redact_secrets

logger = logging.getLogger(__name__)


def _get_env_fallback(param: str, default):
    """Canonical env-with-fallback helper (type-safe, fail-safe).

    When the env var exists, its string value is coerced to match the type
    of *default* (bool → truthy parse, int → int(), float → float()).
    On conversion failure the *default* is returned and a DEBUG line is logged.
    """
    value = os.getenv(param)
    if value is None:
        logger.debug("CONFIG_FALLBACK: %s not set, using default=%s", param, default)
        return default

    # If default is None or str, return the raw string (backward-compat)
    if default is None or isinstance(default, str):
        return value

    # Bool must be checked before int (bool is a subclass of int in Python)
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")

    if isinstance(default, int):
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.debug("CONFIG_FALLBACK: %s=%r not convertible to int, using default=%s", param, value, default)
            return default

    if isinstance(default, float):
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.debug("CONFIG_FALLBACK: %s=%r not convertible to float, using default=%s", param, value, default)
            return default

    return value


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

class EnvironmentConfig(BaseSettings):
    """Environment-specific configuration"""
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    max_file_size: str = Field(default="10MB", env="LOG_MAX_FILE_SIZE")
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    @validator('level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

class LLMConfig(BaseSettings):
    """LLM configuration"""
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    openai_organization: Optional[str] = Field(default=None, env="OPENAI_ORGANIZATION")
    
    # Model Configuration
    primary_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    fallback_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_FALLBACK_MODEL")
    temperature: float = Field(default=0.2, env="OPENAI_TEMPERATURE")
    max_tokens: int = Field(default=150, env="OPENAI_MAX_TOKENS")
    
    # LLM Manager Configuration
    enable_fallback: bool = Field(default=True, env="LLM_ENABLE_FALLBACK")
    enable_circuit_breaker: bool = Field(default=True, env="LLM_ENABLE_CIRCUIT_BREAKER")
    circuit_breaker_threshold: int = Field(default=5, env="LLM_CIRCUIT_BREAKER_THRESHOLD")
    circuit_breaker_timeout: int = Field(default=60, env="LLM_CIRCUIT_BREAKER_TIMEOUT")
    
    # Performance Configuration
    request_timeout: int = Field(default=30, env="LLM_REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, env="LLM_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="LLM_RETRY_DELAY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

class RAGConfig(BaseSettings):
    """RAG System configuration"""
    # Knowledge Base Configuration
    knowledge_file: str = Field(default="assets/knowledge_base/jota_knowledge_base.md", env="RAG_KNOWLEDGE_FILE")
    chunk_size: int = Field(default=1000, env="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="RAG_CHUNK_OVERLAP")
    
    # Retrieval Configuration
    top_k: int = Field(default=5, env="RAG_TOP_K")
    similarity_threshold: float = Field(default=0.7, env="RAG_SIMILARITY_THRESHOLD")
    max_context_length: int = Field(default=4000, env="RAG_MAX_CONTEXT_LENGTH")
    
    # Embedding Configuration
    embedding_model: str = Field(default="text-embedding-ada-002", env="RAG_EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=100, env="RAG_EMBEDDING_BATCH_SIZE")
    
    # Reranking Configuration
    enable_llm_rerank: bool = Field(default=False, env="RAG_ENABLE_LLM_RERANK")
    rerank_model: str = Field(default="gpt-4o", env="RAG_RERANK_MODEL")
    rerank_top_k: int = Field(default=3, env="RAG_RERANK_TOP_K")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

class AgentConfig(BaseSettings):
    """Agent configuration"""
    # General Configuration
    confidence_threshold: float = Field(default=0.7, env="AGENT_CONFIDENCE_THRESHOLD")
    enable_caching: bool = Field(default=True, env="AGENT_ENABLE_CACHING")
    cache_ttl: int = Field(default=3600, env="AGENT_CACHE_TTL")
    
    # Response Configuration
    max_response_length: int = Field(default=500, env="AGENT_MAX_RESPONSE_LENGTH")
    enable_context_truncation: bool = Field(default=True, env="AGENT_ENABLE_CONTEXT_TRUNCATION")
    
    # Grounding Configuration
    enable_grounding_check: bool = Field(default=True, env="AGENT_ENABLE_GROUNDING_CHECK")
    grounding_threshold: float = Field(default=0.8, env="AGENT_GROUNDING_THRESHOLD")
    max_grounding_attempts: int = Field(default=2, env="AGENT_MAX_GROUNDING_ATTEMPTS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

class ConcurrencyConfig(BaseSettings):
    """Concurrency configuration"""
    # Thread Pool Configuration
    max_workers: int = Field(default=10, env="CONCURRENCY_MAX_WORKERS")
    thread_pool_timeout: int = Field(default=30, env="CONCURRENCY_THREAD_POOL_TIMEOUT")
    
    # Rate Limiting
    enable_rate_limiting: bool = Field(default=True, env="CONCURRENCY_ENABLE_RATE_LIMITING")
    requests_per_second: float = Field(default=10.0, env="CONCURRENCY_REQUESTS_PER_SECOND")
    burst_size: int = Field(default=20, env="CONCURRENCY_BURST_SIZE")
    
    # Queue Configuration
    max_queue_size: int = Field(default=100, env="CONCURRENCY_MAX_QUEUE_SIZE")
    queue_timeout: int = Field(default=60, env="CONCURRENCY_QUEUE_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

class SecurityConfig(BaseSettings):
    """Security configuration"""
    # API Security
    enable_api_key_validation: bool = Field(default=True, env="SECURITY_ENABLE_API_KEY_VALIDATION")
    enable_request_signing: bool = Field(default=False, env="SECURITY_ENABLE_REQUEST_SIGNING")
    
    # Data Security
    enable_data_encryption: bool = Field(default=False, env="SECURITY_ENABLE_DATA_ENCRYPTION")
    encryption_key: Optional[str] = Field(default=None, env="SECURITY_ENCRYPTION_KEY")
    
    # Logging Security
    enable_secret_redaction: bool = Field(default=True, env="SECURITY_ENABLE_SECRET_REDACTION")
    log_sensitive_data: bool = Field(default=False, env="SECURITY_LOG_SENSITIVE_DATA")
    
    # Access Control
    enable_ip_whitelist: bool = Field(default=False, env="SECURITY_ENABLE_IP_WHITELIST")
    allowed_ips: List[str] = Field(default_factory=list, env="SECURITY_ALLOWED_IPS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

class FeatureFlags(BaseSettings):
    """Feature flags configuration"""
    # LLM Features
    enable_model_migration: bool = Field(default=False, env="FEATURE_ENABLE_MODEL_MIGRATION")
    enable_canary_models: bool = Field(default=False, env="FEATURE_ENABLE_CANARY_MODELS")
    canary_model_percentage: float = Field(default=0.0, env="FEATURE_CANARY_MODEL_PERCENTAGE")
    
    # RAG Features
    enable_hybrid_search: bool = Field(default=True, env="FEATURE_ENABLE_HYBRID_SEARCH")
    enable_query_rewriting: bool = Field(default=True, env="FEATURE_ENABLE_QUERY_REWRITING")
    enable_context_filtering: bool = Field(default=True, env="FEATURE_ENABLE_CONTEXT_FILTERING")
    
    # Agent Features
    enable_risk_analysis: bool = Field(default=True, env="FEATURE_ENABLE_RISK_ANALYSIS")
    enable_escalation: bool = Field(default=True, env="FEATURE_ENABLE_ESCALATION")
    enable_auto_resolution: bool = Field(default=True, env="FEATURE_ENABLE_AUTO_RESOLUTION")
    
    # Monitoring Features
    enable_detailed_logging: bool = Field(default=False, env="FEATURE_ENABLE_DETAILED_LOGGING")
    enable_performance_metrics: bool = Field(default=True, env="FEATURE_ENABLE_PERFORMANCE_METRICS")
    enable_error_tracking: bool = Field(default=True, env="FEATURE_ENABLE_ERROR_TRACKING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

class AppConfig(BaseSettings):
    """Main application configuration"""
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = False

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

_config_instance: Optional[AppConfig] = None

def load_config() -> AppConfig:
    """
    Load and return the application configuration.
    Implements singleton pattern with lazy loading.
    """
    global _config_instance
    
    if _config_instance is None:
        try:
            _config_instance = AppConfig()
            logger.info("Configuration loaded successfully")
            
            # Validate critical configuration
            _validate_critical_config(_config_instance)
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    return _config_instance

def _validate_critical_config(config: AppConfig) -> None:
    """Validate critical configuration values"""
    errors = []
    
    # Validate OpenAI API key
    if not config.llm.openai_api_key:
        errors.append("OPENAI_API_KEY is required")
    elif config.llm.openai_api_key.startswith("YOUR_") or config.llm.openai_api_key == "OPENAI_USER_API_KEY":
        errors.append("OPENAI_API_KEY must be a real OpenAI API key, not a placeholder")
    
    # Validate model configuration
    if not config.llm.primary_model:
        errors.append("OPENAI_MODEL (primary_model) is required")
    
    # Validate knowledge file
    if not config.rag.knowledge_file:
        errors.append("RAG_KNOWLEDGE_FILE is required")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_config() -> AppConfig:
    """Get the current configuration instance"""
    if _config_instance is None:
        return load_config()
    return _config_instance

def reload_config() -> AppConfig:
    """Force reload of configuration"""
    global _config_instance
    _config_instance = None
    return load_config()

# ============================================================================
# SAFE CONFIGURATION ACCESS
# ============================================================================

def safe_config_dict() -> Dict[str, Any]:
    """
    Return a safe dictionary representation of configuration
    with secrets redacted for logging purposes.
    """
    config = get_config()
    config_dict = config.dict()
    
    # Redact secrets
    if config.security.enable_secret_redaction:
        config_dict = redact_dict(config_dict)
    
    return config_dict

def log_config() -> None:
    """Log configuration safely (with secrets redacted)"""
    config_dict = safe_config_dict()
    logger.info("Configuration loaded", extra={"config": config_dict})

# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def get_llm_config() -> LLMConfig:
    """Get LLM configuration"""
    return get_config().llm

def get_rag_config() -> RAGConfig:
    """Get RAG configuration"""
    return get_config().rag

def get_agent_config() -> AgentConfig:
    """Get agent configuration"""
    return get_config().agent

def get_concurrency_config() -> ConcurrencyConfig:
    """Get concurrency configuration"""
    return get_config().concurrency

def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return get_config().security

def get_feature_flags() -> FeatureFlags:
    """Get feature flags"""
    return get_config().features

def is_development() -> bool:
    """Check if running in development environment"""
    return get_config().environment.environment.lower() == "development"

def is_debug_enabled() -> bool:
    """Check if debug mode is enabled"""
    return get_config().environment.debug

# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================

def validate_environment() -> bool:
    """
    Validate that the environment is properly configured.
    Returns True if valid, False otherwise.
    """
    try:
        config = load_config()
        
        # Check critical environment variables
        critical_vars = [
            "OPENAI_API_KEY",
        ]
        
        missing_vars = []
        for var in critical_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing critical environment variables: {missing_vars}")
            return False
        
        logger.info("Environment validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

# ============================================================================
# CONFIGURATION MIGRATION HELPERS
# ============================================================================

def migrate_from_legacy() -> Dict[str, Any]:
    """
    Helper function to migrate from legacy configuration.
    Returns mapping of legacy variables to new config structure.
    """
    legacy_mappings = {
        # Legacy OpenAI variables
        "OPENAI_API_KEY": "llm.openai_api_key",
        "OPENAI_MODEL": "llm.primary_model",
        "OPENAI_FALLBACK_MODEL": "llm.fallback_model",
        "OPENAI_TEMPERATURE": "llm.temperature",
        "OPENAI_MAX_TOKENS": "llm.max_tokens",
        
        # Legacy RAG variables
        "RAG_KNOWLEDGE_FILE": "rag.knowledge_file",
        "RAG_TOP_K": "rag.top_k",
        "RAG_CHUNK_SIZE": "rag.chunk_size",
        
        # Legacy Agent variables
        "AGENT_CONFIDENCE_THRESHOLD": "agent.confidence_threshold",
        "AGENT_MAX_RESPONSE_LENGTH": "agent.max_response_length",
        
        # Legacy Environment variables
        "ENVIRONMENT": "environment.environment",
        "DEBUG": "environment.debug",
        "LOG_LEVEL": "logging.level",
    }
    
    return legacy_mappings

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "_get_env_fallback",
    "AppConfig",
    "EnvironmentConfig",
    "LoggingConfig", 
    "LLMConfig",
    "RAGConfig",
    "AgentConfig",
    "ConcurrencyConfig",
    "SecurityConfig",
    "FeatureFlags",
    "load_config",
    "get_config",
    "reload_config",
    "safe_config_dict",
    "log_config",
    "get_llm_config",
    "get_rag_config", 
    "get_agent_config",
    "get_concurrency_config",
    "get_security_config",
    "get_feature_flags",
    "is_development",
    "is_debug_enabled",
    "validate_environment",
    "migrate_from_legacy",
]
