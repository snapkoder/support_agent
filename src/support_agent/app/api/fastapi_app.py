from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import register_routes


def create_app(service) -> FastAPI:
    is_prod = os.getenv("JOTA_ENV", "development").lower() == "production"

    app = FastAPI(
        title="Jota Support Agent API",
        description="API para o sistema de agentes de suporte do Jota",
        version="1.0.0",
        docs_url=None if is_prod else "/docs",
        redoc_url=None if is_prod else "/redoc",
    )

    allowed_origins = os.getenv("JOTA_CORS_ORIGINS", "").split(",")
    allowed_origins = [o.strip() for o in allowed_origins if o.strip()] or (["*"] if not is_prod else [])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins if allowed_origins else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    register_routes(app, service)
    return app


async def main() -> int:
    from support_agent.app.service.jota_agent_service import JotaAgentService, ServiceConfig

    host = os.getenv("API_HOST", "localhost")
    port = int(os.getenv("API_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "INFO")

    service = JotaAgentService(ServiceConfig(host=host, port=port, log_level=log_level))
    ok = await service.initialize()
    if not ok:
        return 1
    await service.run()
    return 0


def async_main() -> int:
    """Entry point for setuptools gui-scripts"""
    return asyncio.run(main())

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
