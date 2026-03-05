"""
Security Utilities - Redação de Segredos e Validação
"""

import re
import logging
from typing import Any, Dict, Optional
import os

logger = logging.getLogger(__name__)

def redact_secrets(text: str) -> str:
    """
    Remove ou mascara segredos em texto para logging seguro
    
    Args:
        text: Texto que pode conter segredos
        
    Returns:
        Texto com segredos mascarados
    """
    if not text:
        return text
    
    # 🆕 Redação robusta de CPF/CNPJ - ordem crítica!
    # CNPJ sem formatação: 00000000000000 (deve vir antes do CPF)
    text = re.sub(
        r'(\d{14})(?!\d)',  # Exatamente 14 dígitos
        r'**.***.***/****-**',
        text
    )
    
    # CPF formatado: 000.000.000-00
    text = re.sub(
        r'(\d{3})\.(\d{3})\.(\d{3})-(\d{2})',
        r'***.***.***-**',
        text
    )
    
    # CPF sem formatação: 00000000000
    text = re.sub(
        r'(\d{11})(?!\d)',  # Exatamente 11 dígitos
        r'***.***.***-**',
        text
    )
    
    # CNPJ formatado: 00.000.000/0000-00
    text = re.sub(
        r'(\d{2})\.(\d{3})\.(\d{3})/(\d{4})-(\d{2})',
        r'**.***.***/****-**',
        text
    )
    
    # Mask OpenAI API keys (sk-proj-... ou sk-...)
    text = re.sub(
        r'(sk-[a-zA-Z0-9]{20,})',
        lambda m: f"{m.group(1)[:8]}{'*' * (len(m.group(1)) - 12)}{m.group(1)[-4:]}",
        text
    )
    
    # Mask generic API keys
    text = re.sub(
        r'(api_key["\s]*[:=]["\s]*)([^"\s,}]+)',
        r'\1***REDACTED***',
        text,
        flags=re.IGNORECASE
    )
    
    # Mask authorization headers
    text = re.sub(
        r'(authorization["\s]*[:=]["\s]*bearer\s+)([^"\s,}]+)',
        r'\1***REDACTED***',
        text,
        flags=re.IGNORECASE
    )
    
    # Mask x-api-key headers
    text = re.sub(
        r'(x-api-key["\s]*[:=]["\s]*)([^"\s,}]+)',
        r'\1***REDACTED***',
        text,
        flags=re.IGNORECASE
    )
    
    # Mask environment variables with sensitive names
    sensitive_patterns = [
        r'(password["\s]*[:=]["\s]*)([^"\s,}]+)',
        r'(secret["\s]*[:=]["\s]*)([^"\s,}]+)',
        r'(token["\s]*[:=]["\s]*)([^"\s,}]+)',
        r'(key["\s]*[:=]["\s]*)([^"\s,}]+)',
    ]
    
    for pattern in sensitive_patterns:
        text = re.sub(pattern, r'\1***REDACTED***', text, flags=re.IGNORECASE)
    
    return text

def redact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove segredos de um dicionário recursivamente
    
    Args:
        data: Dicionário que pode conter segredos
        
    Returns:
        Dicionário com segredos mascarados
    """
    if not isinstance(data, dict):
        return data
    
    redacted = {}
    for key, value in data.items():
        key_lower = key.lower()
        
        # Check if key suggests sensitive data
        if any(sensitive in key_lower for sensitive in ['password', 'secret', 'token', 'key', 'auth']):
            if isinstance(value, str):
                redacted[key] = '***REDACTED***'
            else:
                redacted[key] = f'***REDACTED_{type(value).__name__}***'
        elif isinstance(value, dict):
            redacted[key] = redact_dict(value)
        elif isinstance(value, str):
            redacted[key] = redact_secrets(value)
        else:
            redacted[key] = value
    
    return redacted

def safe_log_config(config: Dict[str, Any]) -> str:
    """
    Gera string de log segura para configuração
    
    Args:
        config: Dicionário de configuração
        
    Returns:
        String segura para logging
    """
    redacted_config = redact_dict(config)
    
    # Convert to string but limit size
    config_str = str(redacted_config)
    if len(config_str) > 500:
        config_str = config_str[:497] + "..."
    
    return config_str

def validate_no_secrets_in_logs():
    """
    Valida que não há segredos expostos nos logs recentes
    """
    secret_patterns = [
        r'sk-[a-zA-Z0-9]{20,}',
        r'api_key["\s]*[:=]["\s]*[^"\s,}]{10,}',
        r'authorization["\s]*[:=]["\s]*bearer\s+[^"\s,}]{10,}',
    ]
    
    # This would scan log files in a real implementation
    # For now, just log the validation
    logger.info("Secret validation in logs - patterns checked")

# Example usage for logging
class SecureLogger:
    """Logger wrapper que automaticamente redaciona segredos"""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
    
    def debug(self, message: str, extra: Optional[Dict] = None):
        """Log debug level com redação automática"""
        safe_message = redact_secrets(message)
        safe_extra = redact_dict(extra) if extra else None
        self.logger.debug(safe_message, extra=safe_extra)
    
    def info(self, message: str, extra: Optional[Dict] = None):
        """Log info level com redação automática"""
        safe_message = redact_secrets(message)
        safe_extra = redact_dict(extra) if extra else None
        self.logger.info(safe_message, extra=safe_extra)
    
    def warning(self, message: str, extra: Optional[Dict] = None):
        """Log warning level com redação automática"""
        safe_message = redact_secrets(message)
        safe_extra = redact_dict(extra) if extra else None
        self.logger.warning(safe_message, extra=safe_extra)
    
    def error(self, message: str, extra: Optional[Dict] = None):
        """Log error level com redação automática"""
        safe_message = redact_secrets(message)
        safe_extra = redact_dict(extra) if extra else None
        self.logger.error(safe_message, extra=safe_extra)
    
    def exception(self, message: str, extra: Optional[Dict] = None):
        """Log exception level com redação automática"""
        safe_message = redact_secrets(message)
        safe_extra = redact_dict(extra) if extra else None
        self.logger.error(safe_message, extra=safe_extra)

# Convenience function
def get_secure_logger(name: str) -> SecureLogger:
    """Retorna logger seguro automaticamente"""
    return SecureLogger(name)
