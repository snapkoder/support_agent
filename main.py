#!/usr/bin/env python3
"""
JOTA SUPPORT AGENT - NOVA ARQUITETURA ENTERPRISE
Sistema completo pronto para escala de 800M usuários
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Carregar configuração centralizada
from support_agent.config.settings import get_config, log_config

# Carregar variáveis de ambiente do .env (legado - será removido)
try:
    from dotenv import load_dotenv
    # Tentar carregar .env do diretório atual
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Carregado .env de: {env_path}")
except ImportError:
    print("⚠️ python-dotenv não instalado, usando variáveis de ambiente do sistema")

from support_agent.app.service.jota_agent_service import JotaAgentService, ServiceConfig

# Carregar configuração centralizada
config = get_config()

# Configurar logging usando config centralizada
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format=config.logging.format
)

# Log da configuração de forma segura
log_config()

# Configuração de API usando config centralizada
API_HOST = os.getenv("API_HOST", "localhost")  # Mantém compatibilidade
API_PORT = int(os.getenv("API_PORT", "8000"))  # Mantém compatibilidade

logger = logging.getLogger(__name__)

async def main():
    """Função principal"""
    try:
        # Configuração do serviço
        service_config = ServiceConfig(
            host=API_HOST,
            port=API_PORT,
            debug=config.environment.debug,
            log_level=config.logging.level
        )
        
        # Criar e inicializar serviço
        service = JotaAgentService(service_config)
        
        if await service.initialize():
            logger.info(f"Jota Agent Service starting on {API_HOST}:{API_PORT}")
            await service.run()
        else:
            logger.error("Failed to initialize Jota Agent Service")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
