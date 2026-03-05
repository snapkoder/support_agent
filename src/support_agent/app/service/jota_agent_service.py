from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

import uvicorn

from support_agent.app.api.fastapi_app import create_app


logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0


class JotaAgentService:
    def __init__(self, config: ServiceConfig = None):
        self.config = config or ServiceConfig()
        self.logger = logging.getLogger("jota_agent_service")
        self.orchestrator = None
        self.app = None
        self.start_time = datetime.now()

        self.service_metrics: Dict[str, Any] = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "avg_response_time": 0.0,
            "uptime_seconds": 0.0,
        }

        self.logger.info("JotaAgentService initialized")

    async def initialize(self) -> bool:
        try:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

            from support_agent.orchestrator.agent_orchestrator import get_agent_orchestrator

            self.orchestrator = await get_agent_orchestrator()
            self.app = create_app(self)

            self.logger.info("JotaAgentService initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing service: {e}")
            return False

    def _update_service_metrics(self, success: bool, processing_time: float):
        self.service_metrics["requests_total"] += 1

        if success:
            self.service_metrics["requests_successful"] += 1
        else:
            self.service_metrics["requests_failed"] += 1

        total_requests = self.service_metrics["requests_total"]
        current_avg = self.service_metrics["avg_response_time"]
        self.service_metrics["avg_response_time"] = (
            (current_avg * (total_requests - 1)) + processing_time
        ) / total_requests

    async def run(self):
        try:
            if not self.app:
                raise RuntimeError("Service not initialized")

            self.logger.info(f"Starting Jota Agent Service on {self.config.host}:{self.config.port}")

            config = uvicorn.Config(
                app=self.app,
                host=self.config.host,
                port=self.config.port,
                log_level=self.config.log_level.lower(),
                access_log=True,
            )

            server = uvicorn.Server(config)
            await server.serve()

        except Exception as e:
            self.logger.error(f"Error running service: {e}")
            raise
